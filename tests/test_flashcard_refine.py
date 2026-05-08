"""Server-side flashcard LLM-refinement orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from pgw.db import Base, SessionLocal, get_engine
from pgw.db.models.flashcard import Flashcard, FlashcardRefinement
from pgw.db.models.user import User
from pgw.db.models.workspace import Workspace
from pgw.llm.flashcard import RefineOutput
from pgw.srs.fsrs_engine import initial_state


@pytest.fixture
def seeded(tmp_path: Path) -> Iterator[tuple[User, Workspace]]:
    """User + workspace + 2 cards in pending state."""
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
            source_language="fr",
            target_language="en",
        )
        db.add(ws)
        db.commit()
        db.refresh(ws)

        for surface, back in [("courir", "to run"), ("manger", "to eat")]:
            state = initial_state()
            db.add(
                Flashcard(
                    user_id=user.id,
                    workspace_id=ws.id,
                    front=surface,
                    back=back,
                    language="fr",
                    fsrs_stability=state.stability,
                    fsrs_difficulty=state.difficulty,
                    fsrs_due=state.due,
                    refine_status="pending",
                )
            )
        db.commit()
        yield user, ws


def test_refine_skips_when_disabled(seeded, monkeypatch) -> None:
    """When refinement is disabled, pending cards flip to 'skipped'."""
    monkeypatch.setenv("PGW_FLASHCARD_REFINE", "0")
    from pgw.server.flashcard_refine import refine_card_ids

    with SessionLocal() as db:
        ids = [r.id for r in db.scalars(__import__("sqlalchemy").select(Flashcard))]
    report = refine_card_ids(ids)
    assert report.skipped_disabled == 2
    assert report.refined == 0
    with SessionLocal() as db:
        statuses = {r.refine_status for r in db.scalars(__import__("sqlalchemy").select(Flashcard))}
    assert statuses == {"skipped"}


def test_refine_writes_results_and_populates_cache(seeded, monkeypatch) -> None:
    """Successful refine_batch updates card fields + writes cache rows."""
    monkeypatch.setenv("PGW_FLASHCARD_REFINE", "1")
    monkeypatch.setenv("PGW_LLM__API_BASE", "https://example.invalid/v1")
    monkeypatch.setenv("PGW_LLM__API_KEY", "test-key")

    captured: list[list] = []

    def fake_batch(items, config, *, want_mnemonic=False):
        captured.append(items)
        return [
            RefineOutput(
                lemma=it.surface,
                pos="VERB",
                definition=f"polished gloss for {it.surface}",
                example_source=f"Je {it.surface} chaque jour.",
                example_target=f"I {it.surface} every day.",
                mnemonic=None,
            )
            for it in items
        ]

    monkeypatch.setattr("pgw.server.flashcard_refine.refine_batch", fake_batch)

    from sqlalchemy import select

    from pgw.server.flashcard_refine import refine_card_ids

    with SessionLocal() as db:
        ids = [r.id for r in db.scalars(select(Flashcard))]
    report = refine_card_ids(ids)
    assert report.refined == 2
    assert report.failed == 0
    assert len(captured) == 1  # all 2 cards in one batch

    with SessionLocal() as db:
        cards = list(db.scalars(select(Flashcard).order_by(Flashcard.id)))
        cache = list(db.scalars(select(FlashcardRefinement).order_by(FlashcardRefinement.lemma)))

    for card in cards:
        assert card.refine_status == "done"
        assert card.lemma == card.front
        assert card.pos == "VERB"
        assert card.definition.startswith("polished gloss")
        assert card.example_source.startswith("Je ")
        assert card.refine_model
    assert {c.lemma for c in cache} == {"courir", "manger"}


def test_refine_uses_cache_on_repeat(seeded, monkeypatch) -> None:
    """Second refine call for an identical surface hits the cache."""
    monkeypatch.setenv("PGW_FLASHCARD_REFINE", "1")
    monkeypatch.setenv("PGW_LLM__API_BASE", "https://example.invalid/v1")
    monkeypatch.setenv("PGW_LLM__API_KEY", "test-key")

    call_count = {"n": 0}

    def fake_batch(items, config, *, want_mnemonic=False):
        call_count["n"] += 1
        return [
            RefineOutput(
                lemma=it.surface.lower(),
                pos="VERB",
                definition="cached gloss",
                example_source="example",
                example_target="example",
                mnemonic=None,
            )
            for it in items
        ]

    monkeypatch.setattr("pgw.server.flashcard_refine.refine_batch", fake_batch)

    from sqlalchemy import select

    from pgw.server.flashcard_refine import refine_card_ids

    user, ws = seeded
    # Add a third card with the SAME surface as the first.
    with SessionLocal() as db:
        state = initial_state()
        third = Flashcard(
            user_id=user.id,
            workspace_id=ws.id,
            front="courir",
            back="another gloss",
            language="fr",
            fsrs_stability=state.stability,
            fsrs_difficulty=state.difficulty,
            fsrs_due=state.due,
            refine_status="pending",
        )
        db.add(third)
        db.commit()
        first_two = [r.id for r in db.scalars(select(Flashcard).order_by(Flashcard.id).limit(2))]
        third_id = third.id

    # First batch: refines the original two and populates cache.
    refine_card_ids(first_two)
    # Second batch: only the third card. Should be a cache hit.
    report = refine_card_ids([third_id])
    assert report.cache_hits == 1
    assert report.refined == 0
    assert call_count["n"] == 1  # LLM called only on the first batch


def test_refine_failure_bumps_attempt_counter(seeded, monkeypatch) -> None:
    """LLM raising → attempt counter increments, status remains pending until cap."""
    monkeypatch.setenv("PGW_FLASHCARD_REFINE", "1")
    monkeypatch.setenv("PGW_LLM__API_BASE", "https://example.invalid/v1")
    monkeypatch.setenv("PGW_LLM__API_KEY", "test-key")

    def fake_batch(items, config, *, want_mnemonic=False):
        raise RuntimeError("LLM 500")

    monkeypatch.setattr("pgw.server.flashcard_refine.refine_batch", fake_batch)

    from sqlalchemy import select

    from pgw.server.flashcard_refine import refine_card_ids

    with SessionLocal() as db:
        ids = [r.id for r in db.scalars(select(Flashcard))]
    report = refine_card_ids(ids)
    assert report.failed == 2

    with SessionLocal() as db:
        cards = list(db.scalars(select(Flashcard)))
    # First failure → attempts=1, status stays 'pending'.
    for c in cards:
        assert c.refine_attempts == 1
        assert c.refine_status == "pending"

    # Two more failures should trip the max-attempts cap.
    refine_card_ids(ids)
    refine_card_ids(ids)
    with SessionLocal() as db:
        cards = list(db.scalars(select(Flashcard)))
    for c in cards:
        assert c.refine_attempts >= 3
        assert c.refine_status == "failed"
