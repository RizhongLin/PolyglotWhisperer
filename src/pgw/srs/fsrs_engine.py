"""Wraps the ``fsrs`` library so the rest of the codebase doesn't depend
on its API directly.

We persist three FSRS fields on each ``Flashcard`` row — stability,
difficulty, due. The library's ``Card`` carries more state (state enum,
last-review timestamp, learning step counters) but the trio above is
enough to reproduce the next-due time on subsequent reviews when fed
back into ``Scheduler.review_card``.

Defaults are ``Scheduler()`` — generic 17-parameter weights that are
fine for v1. Per-user retuning lands later by persisting a
``Scheduler.parameters`` blob on ``users``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from fsrs import Card, Rating, Scheduler

# Singleton scheduler — Scheduler is stateless beyond its parameter
# weights, so re-using one across requests is correct.
_SCHEDULER = Scheduler()

# Allowed user-supplied ratings. Maps to fsrs.Rating enum values.
_RATING_MAP: dict[int, Rating] = {
    1: Rating.Again,
    2: Rating.Hard,
    3: Rating.Good,
    4: Rating.Easy,
}


@dataclass(frozen=True)
class CardState:
    """Persisted FSRS state on a Flashcard row."""

    stability: float
    difficulty: float
    due: datetime


#: Sentinel signalling "this row was never reviewed yet". A fresh
#: ``Flashcard`` row is created with stability=0/difficulty=0; we treat
#: that as "no FSRS state" when rehydrating into ``fsrs.Card``. After
#: the first real review the values are non-zero, but a long
#: Again-chain can in theory return them to zero — so we use an
#: explicit sentinel rather than a falsy check.
_UNREVIEWED_SENTINEL = 0.0


def initial_state() -> CardState:
    """State for a fresh card — due immediately, zero stability/difficulty."""
    c = Card()
    return CardState(
        stability=c.stability if c.stability is not None else _UNREVIEWED_SENTINEL,
        difficulty=c.difficulty if c.difficulty is not None else _UNREVIEWED_SENTINEL,
        due=c.due,
    )


def review(state: CardState, rating: int, *, now: datetime | None = None) -> CardState:
    """Apply a rating (1..4) to a stored state and return the new state.

    ``now`` defaults to ``datetime.now(UTC)`` and exists for tests.
    """
    if rating not in _RATING_MAP:
        raise ValueError(f"rating must be 1..4, got {rating!r}")
    review_at = now or datetime.now(timezone.utc)

    # Re-hydrate a fsrs.Card from our persisted trio so the scheduler
    # sees the same state we last saved. Only treat the sentinel value
    # as "unreviewed"; non-zero stability/difficulty are forwarded
    # verbatim so review history accumulates correctly.
    is_unreviewed = (
        state.stability == _UNREVIEWED_SENTINEL and state.difficulty == _UNREVIEWED_SENTINEL
    )
    card = Card(
        stability=None if is_unreviewed else state.stability,
        difficulty=None if is_unreviewed else state.difficulty,
        due=state.due,
    )
    next_card, _log = _SCHEDULER.review_card(
        card=card,
        rating=_RATING_MAP[rating],
        review_datetime=review_at,
    )
    return CardState(
        stability=next_card.stability if next_card.stability is not None else _UNREVIEWED_SENTINEL,
        difficulty=(
            next_card.difficulty if next_card.difficulty is not None else _UNREVIEWED_SENTINEL
        ),
        due=next_card.due,
    )
