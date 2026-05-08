"""Server-side orchestrator for flashcard LLM refinement.

Called from ``BackgroundTasks`` after a card is created and from
``pgw maintenance refine-flashcards`` for retroactive sweeps.

Flow per batch:
  1. Load N pending cards in (language, pos) groups.
  2. For each card check the ``flashcard_refinements`` cache; cache
     hits fill ``definition`` + ``mnemonic`` directly. Misses go to
     the LLM.
  3. Misses are sent to ``llm.flashcard.refine_batch`` 20-at-a-time.
  4. Results are written back to ``flashcards`` and the language-
     agnostic parts populate ``flashcard_refinements``.
  5. Card-level ``refine_status`` flips ``pending`` -> ``done`` /
     ``failed`` (after 3 attempts).

This module does not import FastAPI. The HTTP route schedules the call
via ``BackgroundTasks(refine_card_ids, ...)`` so unit tests can drive
the orchestrator directly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session as SqlaSession

from pgw.core.config import LLMConfig, load_config
from pgw.db.models.flashcard import Flashcard, FlashcardRefinement
from pgw.db.session import SessionLocal
from pgw.llm.flashcard import RefineInput, refine_batch

logger = logging.getLogger(__name__)


_BATCH_SIZE = 20
_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class RefineReport:
    refined: int
    cache_hits: int
    failed: int
    skipped_disabled: int


def is_enabled() -> bool:
    """Soft-toggle for the entire feature.

    Set ``PGW_FLASHCARD_REFINE=0`` to disable. Default is enabled when
    an LLM API base + key are configured; otherwise we skip silently
    (the card still saves with its original ``back`` text).
    """
    if os.environ.get("PGW_FLASHCARD_REFINE") == "0":
        return False
    cfg = _llm_config()
    return bool(cfg.api_base and (cfg.api_key or cfg.backend == "local"))


def want_mnemonic() -> bool:
    """Default-off; opt in via ``PGW_FLASHCARD_REFINE_MNEMONIC=1``.

    Mnemonics roughly double output tokens and quality varies a lot by
    model; a deliberate toggle is the right default.
    """
    return os.environ.get("PGW_FLASHCARD_REFINE_MNEMONIC") == "1"


def _llm_config() -> LLMConfig:
    """Load the project's LLM config — env vars + pgw.toml + defaults."""
    return load_config().llm


def refine_card_ids(card_ids: list[int]) -> RefineReport:
    """Refine the listed cards. Safe to call from a BackgroundTask.

    Opens its own DB session — the caller's session is closed by the
    time BackgroundTasks runs, so we cannot piggyback on it.
    """
    if not card_ids:
        return RefineReport(0, 0, 0, 0)
    if not is_enabled():
        with SessionLocal() as db:
            for cid in card_ids:
                card = db.get(Flashcard, cid)
                if card and card.refine_status == "pending":
                    card.refine_status = "skipped"
            db.commit()
        return RefineReport(0, 0, 0, len(card_ids))

    cfg = _llm_config()
    mnem = want_mnemonic()
    refined = cache_hits = failed = 0

    with SessionLocal() as db:
        cards = list(
            db.scalars(
                select(Flashcard).where(
                    Flashcard.id.in_(card_ids),
                    Flashcard.refine_status.in_(("pending", "failed")),
                )
            )
        )
        # Group by (source_language, target_language) so the LLM batch
        # is consistent. ``target_language`` lives on the workspace —
        # eager-load via the relationship.
        by_lang: dict[tuple[str, str], list[Flashcard]] = {}
        for card in cards:
            ws = card.workspace
            target = (ws.target_language or "en").strip() or "en"
            by_lang.setdefault((card.language, target), []).append(card)

        for (source_lang, target_lang), group in by_lang.items():
            llm_batch: list[Flashcard] = []
            llm_inputs: list[RefineInput] = []
            for card in group:
                cached = _lookup_cache(db, card)
                if cached is not None:
                    _apply_cache_to_card(card, cached, model=cached.model)
                    cache_hits += 1
                    continue
                llm_batch.append(card)
                llm_inputs.append(
                    RefineInput(
                        surface=card.front,
                        language=source_lang,
                        target_language=target_lang,
                        context=_context_for_card(card),
                        prior_back=card.back,
                    )
                )

            for chunk_start in range(0, len(llm_batch), _BATCH_SIZE):
                chunk_cards = llm_batch[chunk_start : chunk_start + _BATCH_SIZE]
                chunk_inputs = llm_inputs[chunk_start : chunk_start + _BATCH_SIZE]
                try:
                    results = refine_batch(chunk_inputs, cfg, want_mnemonic=mnem)
                except Exception:  # noqa: BLE001 — model failures bubble up here
                    logger.exception(
                        "flashcard refine: LLM call failed for %d cards", len(chunk_cards)
                    )
                    for card in chunk_cards:
                        card.refine_attempts += 1
                        card.refine_status = (
                            "failed" if card.refine_attempts >= _MAX_ATTEMPTS else "pending"
                        )
                    failed += len(chunk_cards)
                    continue

                for card, out in zip(chunk_cards, results):
                    _apply_output_to_card(card, out, model=cfg.model)
                    _populate_cache(db, card, out, model=cfg.model)
                    refined += 1

        db.commit()

    return RefineReport(refined=refined, cache_hits=cache_hits, failed=failed, skipped_disabled=0)


def _context_for_card(card: Flashcard) -> str | None:
    """Use the existing ``back`` field as a context hint when present.

    The vocab-backfilled cards typically carry the gloss as ``back``
    and the surrounding sentence is in the workspace's vocab JSON
    (``card.vocab_entry`` -> first occurrence). We pull the cheap one
    here — full vocab lookup is a future improvement.
    """
    return None  # Cheap default; the LLM still uses ``prior_back``.


def _lookup_cache(db: SqlaSession, card: Flashcard) -> FlashcardRefinement | None:
    """Find a cached entry matching ``(language, lemma_or_surface, pos)``.

    On the first refinement we don't yet know the lemma; we look up by
    the surface form lowercased so identical surfaces hit each other.
    """
    surface_key = (card.front or "").strip().lower()
    if not surface_key:
        return None
    # Try (language, surface, ANY pos) — pos may be NULL in cache too.
    return db.scalar(
        select(FlashcardRefinement)
        .where(
            FlashcardRefinement.language == card.language,
            FlashcardRefinement.lemma == surface_key,
        )
        .limit(1)
    )


def _apply_cache_to_card(card: Flashcard, cached: FlashcardRefinement, *, model: str) -> None:
    """Fill the language-agnostic parts of the card from a cache hit."""
    card.lemma = cached.lemma
    card.pos = cached.pos
    card.definition = cached.definition
    card.mnemonic = cached.mnemonic
    # Examples are context-dependent — leave for a future LLM call. We
    # still mark the card as refined; the SPA falls back to ``back``
    # when example fields are empty.
    card.refine_status = "done"
    card.refine_model = model
    card.refined_at = datetime.now(timezone.utc)


def _apply_output_to_card(card: Flashcard, out, *, model: str) -> None:
    card.lemma = out.lemma
    card.pos = out.pos
    card.definition = out.definition
    card.example_source = out.example_source
    card.example_target = out.example_target
    card.mnemonic = out.mnemonic
    card.refine_status = "done"
    card.refine_attempts += 1
    card.refine_model = model
    card.refined_at = datetime.now(timezone.utc)


def _populate_cache(db: SqlaSession, card: Flashcard, out, *, model: str) -> None:
    """Insert a ``flashcard_refinements`` row, idempotent on the natural key."""
    existing = db.scalar(
        select(FlashcardRefinement).where(
            FlashcardRefinement.language == card.language,
            FlashcardRefinement.lemma == out.lemma.lower(),
            FlashcardRefinement.pos == out.pos,
        )
    )
    if existing is not None:
        return
    db.add(
        FlashcardRefinement(
            language=card.language,
            lemma=out.lemma.lower(),
            pos=out.pos,
            definition=out.definition,
            mnemonic=out.mnemonic,
            model=model,
        )
    )
