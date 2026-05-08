"""Bulk-create flashcards from existing vocab entries."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from pgw.db import Base, SessionLocal, get_engine
from pgw.db.models.flashcard import Flashcard
from pgw.db.models.user import User
from pgw.db.models.vocab import VocabEntry, VocabOccurrence
from pgw.db.models.workspace import Workspace
from pgw.maintenance.flashcards_backfill import run as run_backfill


@pytest.fixture
def seeded(tmp_path: Path) -> Iterator[tuple[User, Workspace]]:
    """User + workspace + 3 vocab entries with varying occurrence shapes."""
    Base.metadata.create_all(get_engine())
    with SessionLocal() as db:
        user = User(email="u@example.com", password_hash="x", is_admin=True)
        db.add(user)
        db.commit()
        db.refresh(user)
        ws = Workspace(
            owner_id=user.id,
            slug="demo",
            timestamp="20260508_120000",
            title="Demo",
            fs_path=str(tmp_path),
        )
        db.add(ws)
        db.commit()
        db.refresh(ws)

        # Three entries:
        # 1. has translation + real timing
        # 2. has translation + placeholder (0,0) timing
        # 3. no translation
        for i, (lemma, translation, start_s, end_s) in enumerate(
            [
                ("bonjour", "hello", 1.0, 3.5),
                ("merci", "thanks", 0.0, 0.0),
                ("zut", "", 0.0, 0.0),
            ]
        ):
            entry = VocabEntry(
                user_id=user.id,
                language="fr",
                lemma=lemma,
                pos="NOUN",
                zipf=4.5,
                cefr="A2",
            )
            db.add(entry)
            db.flush()
            db.add(
                VocabOccurrence(
                    entry_id=entry.id,
                    workspace_id=ws.id,
                    segment_index=i,
                    start_seconds=start_s,
                    end_seconds=end_s,
                    surface=lemma,
                    context=f"... {lemma} ...",
                    translation=translation or None,
                )
            )
        db.commit()
        yield user, ws


def test_backfill_creates_cards_for_translated_entries(seeded) -> None:
    user, _ = seeded
    with SessionLocal() as db:
        owner = db.merge(user)
        report = run_backfill(db, owner=owner)
    # 2 entries have translations (bonjour, merci); 1 lacks one (zut).
    assert report.cards_created == 2
    assert report.skipped_no_translation == 1
    assert report.skipped_already_exists == 0
    assert report.skipped_no_occurrence == 0

    with SessionLocal() as db:
        cards = list(db.query(Flashcard).filter_by(user_id=user.id).all())
    assert len(cards) == 2
    fronts = sorted(c.front for c in cards)
    assert fronts == ["bonjour", "merci"]
    by_front = {c.front: c for c in cards}
    # bonjour had real timing → audio range populated
    assert by_front["bonjour"].audio_start_ms == 1000
    assert by_front["bonjour"].audio_end_ms == 3500
    # merci had (0, 0) placeholder → audio range left None
    assert by_front["merci"].audio_start_ms is None
    assert by_front["merci"].audio_end_ms is None


def test_backfill_idempotent(seeded) -> None:
    user, _ = seeded
    with SessionLocal() as db:
        owner = db.merge(user)
        run_backfill(db, owner=owner)
        report2 = run_backfill(db, owner=owner)
    assert report2.cards_created == 0
    assert report2.skipped_already_exists == 2  # 2 already-carded entries


def test_backfill_language_filter(seeded) -> None:
    user, _ = seeded
    with SessionLocal() as db:
        owner = db.merge(user)
        # Filter to Italian — none match → zero cards.
        report = run_backfill(db, owner=owner, language="it")
    assert report.cards_created == 0


def test_backfill_allow_empty_back(seeded) -> None:
    """``require_translation=False`` materialises even untranslated entries."""
    user, _ = seeded
    with SessionLocal() as db:
        owner = db.merge(user)
        report = run_backfill(db, owner=owner, require_translation=False)
    # All 3 entries get cards now (the untranslated one falls back to the lemma).
    assert report.cards_created == 3


def test_backfill_limit(seeded) -> None:
    user, _ = seeded
    with SessionLocal() as db:
        owner = db.merge(user)
        report = run_backfill(db, owner=owner, limit=1)
    assert report.cards_created == 1
