"""FSRS engine wrapper sanity checks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pgw.srs.fsrs_engine import CardState, initial_state, review


def test_initial_state_is_due_now() -> None:
    s = initial_state()
    delta = abs((s.due - datetime.now(timezone.utc)).total_seconds())
    assert delta < 5.0
    assert s.stability >= 0.0
    assert s.difficulty >= 0.0


def test_review_invalid_rating_raises() -> None:
    with pytest.raises(ValueError, match="rating must be 1..4"):
        review(initial_state(), rating=0)
    with pytest.raises(ValueError):
        review(initial_state(), rating=5)


def test_review_again_keeps_card_due_soon() -> None:
    """Rating 1 (Again) should leave the card due in <= 24h."""
    now = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)
    next_state = review(initial_state(), rating=1, now=now)
    assert next_state.due <= now + timedelta(days=1)


def test_review_easy_pushes_due_further_than_good() -> None:
    """Easy schedules later than Good. Smoke check that ratings differentiate."""
    now = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)
    state = initial_state()
    good = review(state, rating=3, now=now)
    easy = review(state, rating=4, now=now)
    assert easy.due >= good.due


def test_review_round_trip_persists_state() -> None:
    """Rehydrating a CardState into review() must produce a stable next state."""
    now = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)
    s1 = review(initial_state(), rating=3, now=now)
    later = now + timedelta(days=1)
    # Re-feed the same state and grade — should advance the due date.
    s2 = review(s1, rating=3, now=later)
    assert s2.due > s1.due
    assert isinstance(s2, CardState)


def test_review_treats_only_double_zero_as_unreviewed() -> None:
    """A non-zero stability paired with zero difficulty must NOT be coerced
    to "unreviewed" — that bug would silently reset a card mid-progression.

    Regression test for the ``state.stability or None`` coercion that
    existed in fsrs_engine until 2026-05-08.
    """
    now = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)
    # Pretend the card was reviewed once: real stability, but a difficulty
    # that happened to land at exactly 0.0 (the FSRS algorithm clamps in
    # rare cases). The next review must NOT throw away ``state.stability``.
    state = CardState(stability=4.5, difficulty=0.0, due=now)
    next_state = review(state, rating=3, now=now)
    # If the bug were present, fsrs would treat this as a brand-new card
    # and produce stability close to its bootstrap weights (~3.0). Real
    # rehydration produces a stability that builds on the prior 4.5.
    assert next_state.stability >= state.stability, (
        f"stability regressed from {state.stability} to {next_state.stability} — "
        "looks like the unreviewed-sentinel coercion is back"
    )
