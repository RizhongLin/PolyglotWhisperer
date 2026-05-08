"""Bulk-create flashcards from existing vocab entries.

For users who came from a pre-P6 install: every ``VocabEntry`` already
records a per-(user, language, lemma) row, often with translations and
context lifted from ``vocabulary.*.json`` during the P2 backfill. This
module walks those entries and creates one fresh ``Flashcard`` per
entry, linked via ``vocab_entry_id`` so re-running is idempotent.

A flashcard needs a ``workspace_id``. We use the **first occurrence**
of each entry as its source workspace — that's also the one with the
context sentence we want on the back of the card. Audio range is
copied from the occurrence's ``(start_seconds, end_seconds)`` when
the values describe a real span; the P2 backfill writes ``(0, 0)``
placeholders for entries that didn't carry timing, and we leave audio
unset for those.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as SqlaSession

from pgw.db.models.flashcard import Flashcard
from pgw.db.models.user import User
from pgw.db.models.vocab import VocabEntry, VocabOccurrence
from pgw.srs.fsrs_engine import initial_state

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlashcardBackfillReport:
    cards_created: int
    skipped_already_exists: int
    skipped_no_translation: int
    skipped_no_occurrence: int


def run(
    db: SqlaSession,
    *,
    owner: User,
    language: str | None = None,
    require_translation: bool = True,
    limit: int | None = None,
) -> FlashcardBackfillReport:
    """Create one flashcard per VocabEntry the user has not yet carded.

    Args:
        owner: User whose entries we materialise as cards.
        language: Optional ISO code filter (``"fr"``). ``None`` walks all.
        require_translation: When True (default), entries whose first
            occurrence lacks a translation are skipped — a card with a
            blank back is useless for review.
        limit: Optional cap on cards created (for spot-testing).
    """
    if owner.id is None:
        raise ValueError("backfill requires a persisted owner with a real id")

    created = 0
    skipped_existing = 0
    skipped_no_translation = 0
    skipped_no_occurrence = 0

    # Build a single set of vocab_entry_ids the user has already carded
    # so the per-row check below is in-memory, not N more queries.
    already_carded: set[int] = {
        entry_id
        for entry_id in db.scalars(
            select(Flashcard.vocab_entry_id).where(
                Flashcard.user_id == owner.id,
                Flashcard.vocab_entry_id.is_not(None),
            )
        )
        if entry_id is not None
    }

    stmt = select(VocabEntry).where(VocabEntry.user_id == owner.id).order_by(VocabEntry.id)
    if language is not None:
        stmt = stmt.where(VocabEntry.language == language)

    for entry in db.scalars(stmt):
        if limit is not None and created >= limit:
            break
        if entry.id in already_carded:
            skipped_existing += 1
            continue

        # First occurrence — order_by id is fine since occurrences are
        # inserted in workspace-creation order during the P2 backfill.
        occ = db.scalar(
            select(VocabOccurrence)
            .where(VocabOccurrence.entry_id == entry.id)
            .order_by(VocabOccurrence.id)
            .limit(1)
        )
        if occ is None:
            skipped_no_occurrence += 1
            continue
        translation = (occ.translation or "").strip()
        if require_translation and not translation:
            skipped_no_translation += 1
            continue

        front = (occ.surface or entry.lemma).strip()
        back = translation or entry.lemma
        # Use the occurrence range only if it describes a real span;
        # the P2 backfill uses (0, 0) placeholders for entries with no
        # known timing — leaving audio_*_ms as None is preferable to
        # storing a zero-length range that will fail validation later.
        audio_start_ms: int | None = None
        audio_end_ms: int | None = None
        if occ.end_seconds > occ.start_seconds:
            audio_start_ms = int(occ.start_seconds * 1000)
            audio_end_ms = int(occ.end_seconds * 1000)

        state = initial_state()
        card = Flashcard(
            user_id=owner.id,
            workspace_id=occ.workspace_id,
            vocab_entry_id=entry.id,
            front=front,
            back=back,
            language=entry.language,
            audio_start_ms=audio_start_ms,
            audio_end_ms=audio_end_ms,
            fsrs_stability=state.stability,
            fsrs_difficulty=state.difficulty,
            fsrs_due=state.due,
        )
        db.add(card)
        try:
            db.flush()
        except IntegrityError:
            # Race with another backfill / FK violation — roll back this
            # one row and keep going.
            db.rollback()
            logger.warning("flashcards backfill: skipping entry %s (integrity error)", entry.id)
            skipped_existing += 1
            continue
        created += 1

    db.commit()
    report = FlashcardBackfillReport(
        cards_created=created,
        skipped_already_exists=skipped_existing,
        skipped_no_translation=skipped_no_translation,
        skipped_no_occurrence=skipped_no_occurrence,
    )
    logger.info("flashcard backfill complete: %s", report)
    return report
