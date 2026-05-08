"""Flashcard CRUD + FSRS review queue.

Mounted at ``/api/flashcards/*``. All routes require an authenticated
user and own scoping by ``user_id`` — no cross-user reads.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session as SqlaSession
from sqlalchemy.orm import joinedload

from pgw.auth.csrf import verify_csrf
from pgw.auth.deps import current_user
from pgw.db.models.flashcard import Flashcard, FlashcardReview
from pgw.db.models.user import User
from pgw.db.models.workspace import Workspace
from pgw.db.session import get_session
from pgw.errors import Err, envelope
from pgw.srs.fsrs_engine import CardState, initial_state
from pgw.srs.fsrs_engine import review as fsrs_review

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/flashcards", tags=["flashcards"])


# ── Request / response models ────────────────────────────────────────────


class CreateFlashcardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace_id: int
    front: str = Field(min_length=1, max_length=1024)
    back: str = Field(min_length=1, max_length=4096)
    language: str = Field(min_length=2, max_length=8)
    audio_start_ms: int | None = Field(default=None, ge=0)
    audio_end_ms: int | None = Field(default=None, ge=0)
    vocab_entry_id: int | None = None


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rating: Literal[1, 2, 3, 4]
    elapsed_ms: int | None = Field(default=None, ge=0)


class FlashcardResponse(BaseModel):
    id: int
    workspace_id: int
    workspace_slug: str
    workspace_timestamp: str
    vocab_entry_id: int | None
    front: str
    back: str
    language: str
    audio_start_ms: int | None
    audio_end_ms: int | None
    fsrs_due: datetime
    fsrs_stability: float
    fsrs_difficulty: float
    created_at: datetime
    updated_at: datetime


def _to_response(card: Flashcard) -> FlashcardResponse:
    # ``card.workspace`` is a relationship; the routes call this inside
    # the request's session scope so SQLAlchemy can lazy-load if it
    # wasn't joined eagerly.
    workspace = card.workspace
    return FlashcardResponse(
        id=card.id,
        workspace_id=card.workspace_id,
        workspace_slug=workspace.slug,
        workspace_timestamp=workspace.timestamp,
        vocab_entry_id=card.vocab_entry_id,
        front=card.front,
        back=card.back,
        language=card.language,
        audio_start_ms=card.audio_start_ms,
        audio_end_ms=card.audio_end_ms,
        fsrs_due=card.fsrs_due,
        fsrs_stability=card.fsrs_stability,
        fsrs_difficulty=card.fsrs_difficulty,
        created_at=card.created_at,
        updated_at=card.updated_at,
    )


# ── Routes ───────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=FlashcardResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_csrf)],
)
def create_flashcard(
    payload: CreateFlashcardRequest,
    user: User = Depends(current_user),
    db: SqlaSession = Depends(get_session),
) -> FlashcardResponse:
    # Workspace must exist and belong to the user (no cross-user cards).
    workspace = db.scalar(
        select(Workspace).where(
            Workspace.id == payload.workspace_id,
            Workspace.owner_id == user.id,
        )
    )
    if workspace is None:
        raise HTTPException(
            status_code=404,
            detail=envelope(Err.WORKSPACE_NOT_FOUND, "workspace not found"),
        )
    if (
        payload.audio_start_ms is not None
        and payload.audio_end_ms is not None
        and payload.audio_end_ms <= payload.audio_start_ms
    ):
        raise HTTPException(
            status_code=400,
            detail=envelope(
                Err.FLASHCARD_INVALID_AUDIO_RANGE,
                "audio_end_ms must be > audio_start_ms",
            ),
        )

    state = initial_state()
    card = Flashcard(
        user_id=user.id,
        workspace_id=workspace.id,
        vocab_entry_id=payload.vocab_entry_id,
        front=payload.front,
        back=payload.back,
        language=payload.language,
        audio_start_ms=payload.audio_start_ms,
        audio_end_ms=payload.audio_end_ms,
        fsrs_stability=state.stability,
        fsrs_difficulty=state.difficulty,
        fsrs_due=state.due,
    )
    db.add(card)
    db.commit()
    db.refresh(card)
    return _to_response(card)


@router.get("/queue", response_model=list[FlashcardResponse])
def review_queue(
    user: User = Depends(current_user),
    db: SqlaSession = Depends(get_session),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[FlashcardResponse]:
    """Cards due now, oldest-due first.

    The index ``ix_flashcards_user_due`` makes this an indexed scan
    even at hundreds of thousands of cards per user.
    """
    now = datetime.now(timezone.utc)
    rows = list(
        db.scalars(
            select(Flashcard)
            .options(joinedload(Flashcard.workspace))
            .where(Flashcard.user_id == user.id, Flashcard.fsrs_due <= now)
            .order_by(Flashcard.fsrs_due)
            .limit(limit)
        )
    )
    return [_to_response(r) for r in rows]


@router.get("", response_model=list[FlashcardResponse])
def list_flashcards(
    user: User = Depends(current_user),
    db: SqlaSession = Depends(get_session),
    workspace_id: int | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> list[FlashcardResponse]:
    """List the user's cards. Optional ``workspace_id`` filter."""
    stmt = (
        select(Flashcard)
        .options(joinedload(Flashcard.workspace))
        .where(Flashcard.user_id == user.id)
    )
    if workspace_id is not None:
        stmt = stmt.where(Flashcard.workspace_id == workspace_id)
    stmt = stmt.order_by(Flashcard.created_at.desc()).limit(limit)
    return [_to_response(r) for r in db.scalars(stmt)]


@router.post(
    "/{card_id}/review",
    response_model=FlashcardResponse,
    dependencies=[Depends(verify_csrf)],
)
def submit_review(
    card_id: int,
    payload: ReviewRequest,
    user: User = Depends(current_user),
    db: SqlaSession = Depends(get_session),
) -> FlashcardResponse:
    card = db.scalar(select(Flashcard).where(Flashcard.id == card_id, Flashcard.user_id == user.id))
    if card is None:
        raise HTTPException(
            status_code=404,
            detail=envelope(Err.FLASHCARD_NOT_FOUND, "flashcard not found"),
        )

    next_state: CardState = fsrs_review(
        CardState(
            stability=card.fsrs_stability,
            difficulty=card.fsrs_difficulty,
            due=card.fsrs_due,
        ),
        rating=payload.rating,
    )
    card.fsrs_stability = next_state.stability
    card.fsrs_difficulty = next_state.difficulty
    card.fsrs_due = next_state.due

    log = FlashcardReview(
        flashcard_id=card.id,
        rating=payload.rating,
        elapsed_ms=payload.elapsed_ms,
        fsrs_stability_after=next_state.stability,
        fsrs_difficulty_after=next_state.difficulty,
        fsrs_due_after=next_state.due,
    )
    db.add(log)
    db.commit()
    db.refresh(card)
    return _to_response(card)


@router.delete(
    "/{card_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_csrf)],
)
def delete_flashcard(
    card_id: int,
    user: User = Depends(current_user),
    db: SqlaSession = Depends(get_session),
) -> None:
    card = db.scalar(select(Flashcard).where(Flashcard.id == card_id, Flashcard.user_id == user.id))
    if card is None:
        raise HTTPException(
            status_code=404,
            detail=envelope(Err.FLASHCARD_NOT_FOUND, "flashcard not found"),
        )
    db.delete(card)
    db.commit()
