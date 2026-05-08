"""Auth + setup HTTP endpoints.

Mounted at ``/api/auth/*`` and ``/api/me``.

Bootstrap rule: ``POST /api/auth/setup`` only succeeds when the
``users`` table is empty. After that, the regular login flow applies.
"""

from __future__ import annotations

import os
from typing import Literal

from email_validator import EmailNotValidError, validate_email
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session as SqlaSession

from pgw.auth.bootstrap import create_user, has_any_user, setup_lock
from pgw.auth.csrf import CSRF_COOKIE, issue_csrf_token, verify_csrf
from pgw.auth.deps import (
    SESSION_COOKIE,
    current_user,
    current_user_optional,
    mark_bootstrap_ended,
)
from pgw.auth.passwords import verify_password
from pgw.auth.sessions import create_session, revoke_session
from pgw.db.models.user import User
from pgw.db.session import get_session
from pgw.errors import Err, envelope

router = APIRouter(prefix="/api", tags=["auth"])


class SetupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=256)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str
    password: str


class AuthStateResponse(BaseModel):
    has_admin: bool
    authenticated: bool


class MeResponse(BaseModel):
    id: int
    email: str
    is_admin: bool


def _set_session_cookies(
    response: Response,
    *,
    raw_session: str,
    csrf: str,
    secure: bool,
) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        raw_session,
        max_age=60 * 60 * 24 * 30,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )
    response.set_cookie(
        CSRF_COOKIE,
        csrf,
        max_age=60 * 60 * 24 * 30,
        httponly=False,
        samesite="lax",
        secure=secure,
        path="/",
    )


def _is_https(request: Request) -> bool:
    """Decide whether to set the cookie ``Secure`` flag.

    Priority:
      1. ``PGW_SECURE_COOKIES=1`` — operator-controlled, always wins.
      2. The actual request scheme (``https``).
      3. ``X-Forwarded-Proto: https`` — only honoured if
         ``PGW_TRUST_PROXY_HEADERS=1`` is set, since otherwise any
         client could spoof it on a direct HTTP deployment.
    """
    if os.environ.get("PGW_SECURE_COOKIES") == "1":
        return True
    if request.url.scheme == "https":
        return True
    if os.environ.get("PGW_TRUST_PROXY_HEADERS") == "1":
        return request.headers.get("x-forwarded-proto") == "https"
    return False


def _validate_email_or_400(raw: str) -> str:
    try:
        info = validate_email(raw, check_deliverability=False)
        return info.normalized.lower()
    except EmailNotValidError as exc:
        raise HTTPException(
            status_code=400,
            detail=envelope(Err.AUTH_INVALID_EMAIL, f"invalid email: {exc}"),
        ) from exc


@router.get("/auth/state", response_model=AuthStateResponse)
def auth_state(
    user: User | None = Depends(current_user_optional),
    db: SqlaSession = Depends(get_session),
) -> AuthStateResponse:
    return AuthStateResponse(
        has_admin=has_any_user(db),
        authenticated=user is not None,
    )


@router.post("/auth/setup", status_code=status.HTTP_201_CREATED)
def setup(
    payload: SetupRequest,
    request: Request,
    response: Response,
    db: SqlaSession = Depends(get_session),
) -> dict[str, Literal["ok"]]:
    # Validate email outside the lock so we don't hold it during slow
    # client work (e.g. a full DNS validation if it's ever turned on).
    email = _validate_email_or_400(payload.email)
    # ``setup_lock`` serialises within-process via a threading.Lock and
    # across processes (Postgres) via pg_advisory_xact_lock — closes the
    # has_any_user → create_user TOCTOU that otherwise lets two
    # concurrent setup requests both create an admin.
    with setup_lock(db):
        if has_any_user(db):
            raise HTTPException(
                status_code=409,
                detail=envelope(Err.SETUP_ALREADY_COMPLETE, "setup already complete"),
            )
        user = create_user(db, email=email, password=payload.password, is_admin=True)
        # Lock the bootstrap door immediately so a concurrent racer can't
        # exploit a brief no-users-in-DB window via a different request path.
        mark_bootstrap_ended()
    raw = create_session(db, user, user_agent=request.headers.get("user-agent"))
    csrf = issue_csrf_token()
    _set_session_cookies(response, raw_session=raw, csrf=csrf, secure=_is_https(request))
    return {"status": "ok"}


@router.post("/auth/login", status_code=status.HTTP_204_NO_CONTENT)
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: SqlaSession = Depends(get_session),
) -> None:
    email = payload.email.lower().strip()
    user = db.scalar(select(User).where(User.email == email))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail=envelope(Err.AUTH_INVALID_CREDENTIALS, "invalid credentials"),
        )
    raw = create_session(db, user, user_agent=request.headers.get("user-agent"))
    csrf = issue_csrf_token()
    _set_session_cookies(response, raw_session=raw, csrf=csrf, secure=_is_https(request))


@router.post(
    "/auth/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_csrf)],
)
def logout(
    response: Response,
    db: SqlaSession = Depends(get_session),
    pgw_sid: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> None:
    if pgw_sid:
        revoke_session(db, pgw_sid)
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")


@router.get("/me", response_model=MeResponse)
def me(user: User = Depends(current_user)) -> MeResponse:
    return MeResponse(id=user.id, email=user.email, is_admin=user.is_admin)
