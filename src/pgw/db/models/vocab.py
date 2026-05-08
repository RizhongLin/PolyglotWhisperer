"""Cross-workspace vocab knowledge base.

``VocabEntry`` is the per-user lemma index — one row per
``(user_id, language, lemma, pos)``. ``VocabOccurrence`` records every
sighting in a workspace at a given segment / time range.

Backfill happens during P2's first-boot import; runtime updates land in
P5 (cross-workspace vocab dashboard) and are populated incrementally
from each pipeline run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pgw.db.base import Base
from pgw.db.types import BigIntPK

if TYPE_CHECKING:
    from pgw.db.models.workspace import Workspace


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class VocabEntry(Base):
    __tablename__ = "vocab_entries"
    __table_args__ = (
        UniqueConstraint("user_id", "language", "lemma", "pos", name="uq_vocab_entry_natural_key"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    language: Mapped[str] = mapped_column(Text, nullable=False)
    lemma: Mapped[str] = mapped_column(Text, nullable=False)
    pos: Mapped[str | None] = mapped_column(Text, nullable=True)
    zipf: Mapped[float | None] = mapped_column(Float, nullable=True)
    cefr: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    occurrences: Mapped[list["VocabOccurrence"]] = relationship(
        back_populates="entry", cascade="all, delete-orphan"
    )


class VocabOccurrence(Base):
    __tablename__ = "vocab_occurrences"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    entry_id: Mapped[int] = mapped_column(
        ForeignKey("vocab_entries.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[int] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    surface: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[str | None] = mapped_column(Text, nullable=True)
    translation: Mapped[str | None] = mapped_column(Text, nullable=True)

    entry: Mapped[VocabEntry] = relationship(back_populates="occurrences")
    workspace: Mapped["Workspace"] = relationship(back_populates="vocab_occurrences")
