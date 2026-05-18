"""Per-user credential storage and preferences.

Mounted at ``/api/auth/*``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete as sqla_delete
from sqlalchemy import select
from sqlalchemy.orm import Session as SqlaSession

from pgw.auth.csrf import verify_csrf
from pgw.auth.deps import current_user
from pgw.auth.passwords import hash_password, verify_password
from pgw.crypto.encryption import encrypt
from pgw.db.models.credential import UserCredential
from pgw.db.models.user import User
from pgw.db.session import get_session

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _ok(**extra: object) -> dict[str, object]:
    return {"ok": True, **extra}


# ── Credential models ─────────────────────────────────────────────────


class CredentialCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    service: str = Field(min_length=1, max_length=20)
    provider: str = Field(min_length=1, max_length=50)
    api_key: str = Field(min_length=1)
    api_base: str | None = Field(default=None, max_length=500)
    api_model: str | None = Field(default=None, max_length=200)


class CredentialOut(BaseModel):
    id: int
    service: str
    provider: str
    masked_key: str
    api_base: str | None
    api_model: str | None
    created_at: str


# ── Password change ───────────────────────────────────────────────────


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_password: str
    new_password: str = Field(min_length=8, max_length=256)


@router.put("/password", dependencies=[Depends(verify_csrf)])
def change_password(
    body: ChangePasswordRequest,
    user: User = Depends(current_user),
    db: SqlaSession = Depends(get_session),
):
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    user.password_hash = hash_password(body.new_password)
    db.commit()
    return _ok()


# ── Credential CRUD ───────────────────────────────────────────────────


@router.get("/credentials")
def list_credentials(
    user: User = Depends(current_user),
    db: SqlaSession = Depends(get_session),
) -> list[dict]:
    rows = db.scalars(select(UserCredential).where(UserCredential.user_id == user.id)).all()
    result = []
    for r in rows:
        result.append(
            {
                "id": r.id,
                "service": r.service,
                "provider": r.provider,
                "masked_key": "****",
                "api_base": r.api_base,
                "api_model": r.api_model,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
        )
    return result


@router.post(
    "/credentials",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_csrf)],
)
def create_credential(
    body: CredentialCreate,
    user: User = Depends(current_user),
    db: SqlaSession = Depends(get_session),
):
    encrypted = encrypt(body.api_key)
    cred = UserCredential(
        user_id=user.id,
        service=body.service.lower(),
        provider=body.provider.lower(),
        encrypted_value=encrypted,
        api_base=body.api_base or None,
        api_model=body.api_model or None,
    )
    db.add(cred)
    db.commit()
    db.refresh(cred)
    return _ok(id=cred.id)


@router.delete(
    "/credentials/{credential_id}",
    dependencies=[Depends(verify_csrf)],
)
def delete_credential(
    credential_id: int,
    user: User = Depends(current_user),
    db: SqlaSession = Depends(get_session),
):
    stmt = sqla_delete(UserCredential).where(
        UserCredential.id == credential_id,
        UserCredential.user_id == user.id,
    )
    result = db.execute(stmt)
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Credential not found")
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── Preferences ───────────────────────────────────────────────────────


class PreferencesUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: str | None = None
    translate: str | None = None
    backend: str | None = None
    llm_backend: str | None = None


@router.get("/preferences")
def get_preferences(user: User = Depends(current_user)) -> dict:
    return user.preferences or {}


@router.put("/preferences", dependencies=[Depends(verify_csrf)])
def update_preferences(
    body: PreferencesUpdate,
    user: User = Depends(current_user),
    db: SqlaSession = Depends(get_session),
):
    prefs = dict(user.preferences or {})
    for key in ("language", "translate", "backend", "llm_backend"):
        val = getattr(body, key, None)
        if val is not None:
            prefs[key] = val
    user.preferences = prefs
    db.commit()
    return _ok()
