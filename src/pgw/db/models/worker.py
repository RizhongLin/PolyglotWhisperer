"""Worker token + connection-session ORM.

A ``WorkerToken`` is a long-lived credential. The user gets the raw
value once at issue time; we only persist sha256(raw). Each token can
be revoked.

A ``WorkerSession`` records a single WebSocket lifetime — opened on
connect, ``disconnected_at`` set on close. ``capabilities`` is a JSONB
blob the worker advertises in its ``ready`` frame (``{ gpu, mlx, ffmpeg,
pgw_version, supported_langs }``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pgw.db.base import Base
from pgw.db.types import BigIntPK

if TYPE_CHECKING:
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class WorkerToken(Base):
    __tablename__ = "worker_tokens"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    sessions: Mapped[list["WorkerSession"]] = relationship(
        back_populates="token", cascade="all, delete-orphan"
    )


class WorkerSession(Base):
    __tablename__ = "worker_sessions"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    token_id: Mapped[int] = mapped_column(
        ForeignKey("worker_tokens.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    connected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    disconnected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    hostname: Mapped[str | None] = mapped_column(Text, nullable=True)
    pgw_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    capabilities: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    token: Mapped[WorkerToken] = relationship(back_populates="sessions")
