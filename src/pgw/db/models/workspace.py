"""Workspace and job ORM models.

The DB stores metadata, vocabulary, and summary stats.  Video/audio/VTT
blobs and PDF/EPUB exports remain on the filesystem and are served from
there via ``/ws/...`` passthrough.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pgw.db.base import Base
from pgw.db.types import BigIntPK, JSONText

if TYPE_CHECKING:
    from pgw.db.models.user import User
    from pgw.db.models.vocab import VocabOccurrence


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Workspace(Base):
    __tablename__ = "workspaces"
    __table_args__ = (
        UniqueConstraint("owner_id", "slug", "timestamp", name="uq_workspace_owner_path"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    owner_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    slug: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_language: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_language: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    embed_provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    embed_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    embed_blocked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    fs_path: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONText, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )

    owner: Mapped["User"] = relationship(back_populates="workspaces")
    jobs: Mapped[list["WorkspaceJob"]] = relationship(
        back_populates="workspace",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    vocab_occurrences: Mapped[list["VocabOccurrence"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )


class WorkspaceJob(Base):
    __tablename__ = "workspace_jobs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    owner_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    workspace_id: Mapped[int | None] = mapped_column(
        BigIntPK, ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True
    )
    state: Mapped[str] = mapped_column(Text, nullable=False)
    inputs: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stage: Mapped[str | None] = mapped_column(Text, nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    owner: Mapped["User"] = relationship(back_populates="jobs")
    workspace: Mapped["Workspace | None"] = relationship(back_populates="jobs")
