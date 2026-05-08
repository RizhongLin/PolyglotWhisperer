"""FastAPI dependencies for auth.

Three dep flavors:

- ``current_user``: requires a real session; 401 otherwise.
- ``current_user_optional``: returns ``None`` if no session — for
  endpoints serving both authed and guest contexts.
- ``current_user_or_bootstrap``: returns a synthetic SYSTEM user when
  no users exist in the DB AND bootstrap mode hasn't ended. Once any
  user has been created — or ``PGW_DISABLE_BOOTSTRAP=1`` is set — the
  bootstrap path is permanently closed for this process and behaves
  like ``current_user`` (401 on missing session). This makes "DB wiped
  while server runs" no longer re-open guest access.

Usage:

    @app.get("/api/me")
    def me(user: User = Depends(current_user)):
        ...
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import cast

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session as SqlaSession

from pgw.auth.sessions import lookup_session
from pgw.db.models.user import User
from pgw.db.session import get_session
from pgw.errors import Err, envelope

SESSION_COOKIE = "pgw_sid"


@dataclass(frozen=True)
class _SystemUser:
    """Sentinel returned by ``current_user_or_bootstrap`` pre-setup.

    A frozen dataclass — not a ``User`` ORM instance — so it can't be
    accidentally added to a ``Session`` and persisted. Shape-compatible
    with ``User`` for the attributes callers actually read.
    """

    id: int | None = None
    email: str = "system"
    is_admin: bool = True


#: Process-wide sticky flag: once we've ever seen a real user, we don't
#: re-enter bootstrap mode in this process even if the DB is later
#: emptied. ``Event`` is thread-safe and lock-free for is_set/set.
_bootstrap_ended = threading.Event()
if os.environ.get("PGW_DISABLE_BOOTSTRAP") == "1":
    _bootstrap_ended.set()

#: Public name kept for backwards compat with imports; instance has
#: ``id=None``, ``is_admin=True``. Cast for type-checker compatibility
#: with downstream consumers that expect ``User``-shaped objects.
SYSTEM_USER = cast(User, _SystemUser())


def mark_bootstrap_ended() -> None:
    """Called by routes that create the first user to lock the door."""
    _bootstrap_ended.set()


def current_user_optional(
    pgw_sid: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    db: SqlaSession = Depends(get_session),
) -> User | None:
    if not pgw_sid:
        return None
    pair = lookup_session(db, pgw_sid)
    if pair is None:
        return None
    return pair[1]


def current_user(
    user: User | None = Depends(current_user_optional),
) -> User:
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=envelope(Err.AUTH_NOT_AUTHENTICATED, "not authenticated"),
        )
    return user


def current_user_or_bootstrap(
    user: User | None = Depends(current_user_optional),
    db: SqlaSession = Depends(get_session),
) -> User:
    """Allow guest access when the DB has no users yet (pre-setup).

    Once a real user logs in OR ``PGW_DISABLE_BOOTSTRAP=1`` is set,
    bootstrap mode is permanently closed for this process. A wiped
    ``users`` table will NOT reopen guest access.
    """
    if user is not None:
        # First successful auth in this process locks the bootstrap door
        # forever — even if the DB is later emptied or corrupted.
        _bootstrap_ended.set()
        return user
    if _bootstrap_ended.is_set():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=envelope(Err.AUTH_NOT_AUTHENTICATED, "not authenticated"),
        )
    if db.scalar(select(User.id).limit(1)) is None:
        return SYSTEM_USER
    # Real users exist but caller isn't authenticated — close the door
    # and reject.
    _bootstrap_ended.set()
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="not authenticated")


def require_admin(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=envelope(Err.AUTH_ADMIN_REQUIRED, "admin required"),
        )
    return user
