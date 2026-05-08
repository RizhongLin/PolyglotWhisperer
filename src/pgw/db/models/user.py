"""User and session ORM models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pgw.db.base import Base
from pgw.db.types import BigIntPK

if TYPE_CHECKING:
    from pgw.db.models.workspace import Workspace, WorkspaceJob


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    # BigInteger to match alembic migration 0002. SQLite treats both
    # Integer and BigInteger as INTEGER, but Postgres differs — keeping
    # ORM and Alembic in sync prevents `Base.metadata.create_all()`
    # diverging from migration-driven schema.
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    # CITEXT on Postgres; TEXT on SQLite. Store lower-cased emails to
    # keep behavior portable.
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    sessions: Mapped[list["Session"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    workspaces: Mapped[list["Workspace"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )
    jobs: Mapped[list["WorkspaceJob"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )


class Session(Base):
    __tablename__ = "sessions"

    # 64-char hex sha256 of the cookie token. Cookie carries the raw
    # value; we never store the raw token server-side.
    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip: Mapped[str | None] = mapped_column(String(45), nullable=True)

    user: Mapped[User] = relationship(back_populates="sessions")
