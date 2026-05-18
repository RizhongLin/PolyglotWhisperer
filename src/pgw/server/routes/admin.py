"""Admin user management endpoints.

Mounted at ``/api/admin/*`` (admin-only).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session as SqlaSession

from pgw.auth.bootstrap import create_user
from pgw.auth.csrf import verify_csrf
from pgw.auth.deps import current_user
from pgw.auth.passwords import hash_password
from pgw.db.models.user import User
from pgw.db.session import get_session
from pgw.errors import Err, envelope

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _require_admin(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=envelope(Err.AUTH_ADMIN_REQUIRED, "admin required"),
        )
    return user


def _ok(**extra: object) -> dict[str, object]:
    return {"ok": True, **extra}


# ── Models ─────────────────────────────────────────────────────────────


class CreateUserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=256)
    is_admin: bool = False


class ResetPasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    password: str = Field(min_length=8, max_length=256)


# ── Endpoints ──────────────────────────────────────────────────────────


@router.get("/users")
def list_users(
    _admin: User = Depends(_require_admin),
    db: SqlaSession = Depends(get_session),
) -> list[dict]:
    rows = db.scalars(select(User).order_by(User.created_at.desc())).all()
    return [
        {
            "id": u.id,
            "email": u.email,
            "is_admin": u.is_admin,
            "created_at": u.created_at.isoformat() if u.created_at else "",
        }
        for u in rows
    ]


@router.post(
    "/users",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_csrf)],
)
def create(
    body: CreateUserRequest,
    _admin: User = Depends(_require_admin),
    db: SqlaSession = Depends(get_session),
):
    try:
        user = create_user(db, email=body.email, password=body.password, is_admin=body.is_admin)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _ok(id=user.id, email=user.email)


@router.put(
    "/users/{user_id}/password",
    dependencies=[Depends(verify_csrf)],
)
def reset_password(
    user_id: int,
    body: ResetPasswordRequest,
    _admin: User = Depends(_require_admin),
    db: SqlaSession = Depends(get_session),
):
    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    target.password_hash = hash_password(body.password)
    db.commit()
    return _ok()


@router.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_csrf)],
)
def delete_user(
    user_id: int,
    _admin: User = Depends(_require_admin),
    db: SqlaSession = Depends(get_session),
):
    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    if target.id == _admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    if target.is_admin:
        n_admins = len(db.scalars(select(User.id).where(User.is_admin.is_(True))).all())
        if n_admins <= 1:
            raise HTTPException(status_code=400, detail="Cannot delete the last admin")
    db.delete(target)
    db.commit()
