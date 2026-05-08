"""Flashcards + FSRS review log.

``Flashcard`` is one card a user has saved from a workspace transcript
(or vocab occurrence). ``FlashcardReview`` is the per-grade log used to
re-derive FSRS state if the algorithm is ever retuned.

FSRS state is materialised on the card itself (``fsrs_stability``,
``fsrs_difficulty``, ``fsrs_due``) so the review queue is a single
indexed scan ``WHERE user_id = ? AND fsrs_due <= now() ORDER BY fsrs_due``.

Audio range is optional — cards built from a vocab word default to the
segment's start/end and the audio-clip endpoint slices the workspace
audio on demand.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pgw.db.base import Base
from pgw.db.types import BigIntPK

if TYPE_CHECKING:
    from pgw.db.models.user import User
    from pgw.db.models.vocab import VocabEntry
    from pgw.db.models.workspace import Workspace


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Flashcard(Base):
    __tablename__ = "flashcards"
    __table_args__ = (
        Index("ix_flashcards_user_due", "user_id", "fsrs_due"),
        Index("ix_flashcards_workspace", "workspace_id"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    workspace_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    )
    vocab_entry_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("vocab_entries.id", ondelete="SET NULL"),
        nullable=True,
    )

    front: Mapped[str] = mapped_column(Text, nullable=False)
    back: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(Text, nullable=False)

    audio_start_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    audio_end_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── LLM-refined content ──
    # Populated asynchronously after card creation by ``pgw.server.flashcard_refine``.
    # Until ``refine_status`` flips to ``'done'``, the SPA renders the
    # original ``front``/``back`` fields above as a fallback.
    lemma: Mapped[str | None] = mapped_column(Text, nullable=True)
    pos: Mapped[str | None] = mapped_column(String(16), nullable=True)
    definition: Mapped[str | None] = mapped_column(Text, nullable=True)
    example_source: Mapped[str | None] = mapped_column(Text, nullable=True)
    example_target: Mapped[str | None] = mapped_column(Text, nullable=True)
    mnemonic: Mapped[str | None] = mapped_column(Text, nullable=True)
    refine_status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    refine_attempts: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    refine_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    refined_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # FSRS state — fsrs.Card maps to (stability, difficulty, state, due, ...).
    # We keep the trio that drives the review queue on the row; the rest is
    # re-derivable from the review log if we ever retune.
    fsrs_stability: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fsrs_difficulty: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fsrs_due: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    user: Mapped["User"] = relationship()
    workspace: Mapped["Workspace"] = relationship()
    vocab_entry: Mapped["VocabEntry | None"] = relationship()
    reviews: Mapped[list["FlashcardReview"]] = relationship(
        back_populates="flashcard",
        cascade="all, delete-orphan",
        order_by="FlashcardReview.created_at",
    )


class FlashcardReview(Base):
    __tablename__ = "flashcard_reviews"
    __table_args__ = (Index("ix_flashcard_reviews_card_time", "flashcard_id", "created_at"),)

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    flashcard_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("flashcards.id", ondelete="CASCADE"),
        nullable=False,
    )
    # FSRS rating: 1=Again, 2=Hard, 3=Good, 4=Easy.
    rating: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    elapsed_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fsrs_stability_after: Mapped[float] = mapped_column(Float, nullable=False)
    fsrs_difficulty_after: Mapped[float] = mapped_column(Float, nullable=False)
    fsrs_due_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    flashcard: Mapped[Flashcard] = relationship(back_populates="reviews")


class FlashcardRefinement(Base):
    """Cache of LLM-produced ``(definition, mnemonic)`` per ``(language, lemma, pos)``.

    Refinement is dictionary-grade for definition/POS — re-running on
    the same lemma is wasted spend. Example sentences are NOT cached
    (they're context-dependent on the workspace's transcript).
    """

    __tablename__ = "flashcard_refinements"
    __table_args__ = (
        UniqueConstraint("language", "lemma", "pos", name="uq_flashcard_refinements_natural_key"),
        Index("ix_flashcard_refinements_lookup", "language", "lemma", "pos"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    language: Mapped[str] = mapped_column(String(8), nullable=False)
    lemma: Mapped[str] = mapped_column(Text, nullable=False)
    pos: Mapped[str | None] = mapped_column(String(16), nullable=True)
    definition: Mapped[str] = mapped_column(Text, nullable=False)
    mnemonic: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
