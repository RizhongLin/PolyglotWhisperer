"""LLM-based subtitle translation with chunked processing."""

from __future__ import annotations

import json
import re
from typing import Callable

from pgw.core.config import LLMConfig
from pgw.core.models import SubtitleSegment, TranslationResult
from pgw.llm.client import complete
from pgw.llm.prompts import (
    TRANSLATION_SYSTEM,
    TRANSLATION_USER,
    UNTRANSLATED_MARKER,
    filter_empty_segments,
    format_bilingual_context,
    format_history_context,
    format_json_segments,
    parse_json_response,
    parse_numbered_response,
    reconstruct_with_empties,
)
from pgw.utils.console import chunk_progress, debug, warning
from pgw.utils.text import SENTENCE_END_CHARS, TIMING_GAP_THRESHOLD, find_sentence_split

CHUNK_SIZE = 48
OVERLAP = 6  # Forward lookahead — segments translated but discarded, for boundary context
BACK_OVERLAP = 4  # Backward re-translation — later chunk overwrites previous chunk's tail
HISTORY_SIZE = 8  # Number of previous translated pairs to include as context
MAX_RETRY_DEPTH = 3  # Max recursion for binary-split retries
SCAN_RANGE = 5  # How far to scan for sentence boundaries around ideal split point


def _auto_chunk_size(model: str) -> int:
    """Estimate chunk size from model name based on parameter count.

    Smaller models need smaller chunks to produce reliable JSON output.
    Uses a log-scale formula: chunk_size = clamp(8 * log2(params), 10, 48)
    This gives roughly: 0.5B→8, 1B→10, 3B→13, 7B→22, 14B→30, 30B→39, 70B→48
    """
    import math

    # Extract parameter count from model name (e.g. "qwen3.5:9b" → 9, "70b" → 70)
    match = re.search(r"(\d+(?:\.\d+)?)[bB]", model)
    if not match:
        return CHUNK_SIZE  # Unknown size (API models, etc.), use default

    params = float(match.group(1))
    if params <= 0:
        return CHUNK_SIZE

    size = int(8 * math.log2(max(params, 1)))
    return max(10, min(size, CHUNK_SIZE))


def parse_response(response: str, expected_count: int) -> tuple[list[str], bool]:
    """Three-tier response parsing: JSON first, then numbered, then padded fallback."""
    texts, exact = parse_json_response(response, expected_count)
    if exact:
        return texts, True
    # JSON found translations but wrong count — still better than numbered fallback
    if texts:
        return texts, False
    return parse_numbered_response(response, expected_count)


def process_chunk(
    texts: list[str],
    source_lang: str,
    target_lang: str,
    config: LLMConfig,
    system_prompt: str,
    context: str,
    _depth: int = 0,
    _retried: bool = False,
) -> list[str]:
    """Process a single translation chunk with retry-before-split strategy.

    On count mismatch:
    1. First attempt: reask the LLM with error feedback (single retry)
    2. If retry fails: binary split with fresh context for the second half
    3. Max recursion depth before giving up and returning originals
    """
    if _depth >= MAX_RETRY_DEPTH:
        return texts  # Best-effort: return originals to avoid infinite recursion

    json_segments = format_json_segments(texts)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": TRANSLATION_USER.format(
                count=len(texts),
                source_lang=source_lang,
                target_lang=target_lang,
                context=context,
                json_segments=json_segments,
            ),
        },
    ]

    response = complete(messages, config, response_format={"type": "json_object"})
    translated_texts, exact_match = parse_response(response, len(texts))

    if exact_match or len(texts) <= 2:
        return translated_texts

    # --- Debug: dump raw response and source on mismatch ---
    import os

    if os.environ.get("PGW_DEBUG"):
        debug("--- DEBUG: source JSON ---")
        debug(json_segments)
        debug("--- DEBUG: raw LLM response ---")
        debug(response)
        # Show raw key count from response
        try:
            raw = json.loads(response.strip())
            if isinstance(raw, dict):
                debug(f"Raw keys: {len(raw)} (expected {len(texts)})")
        except (ValueError, json.JSONDecodeError):
            debug("Failed to parse response as JSON")

    # --- Issue 5: Single retry before split ---
    if not _retried:
        parsed_count = sum(1 for t in translated_texts if t)
        warning(
            f"Count mismatch ({parsed_count} vs {len(texts)} expected), " f"retrying same chunk..."
        )
        reask_msg = (
            f"You returned {parsed_count} items but I need exactly "
            f"{len(texts)} translations. Please return a JSON object with "
            f'keys "1" through "{len(texts)}", each mapped to its translation.'
        )
        retry_messages = messages + [
            {"role": "assistant", "content": response},
            {"role": "user", "content": reask_msg},
        ]
        response2 = complete(retry_messages, config, response_format={"type": "json_object"})
        translated_texts2, exact_match2 = parse_response(response2, len(texts))
        if exact_match2:
            return translated_texts2
        # Fall through to split

    # --- Issue 4: Binary split with sentence-boundary-aware split point ---
    warning(f"Splitting into smaller batches ({len(texts)} segments)...")
    mid = find_sentence_split(texts)
    first_half = process_chunk(
        texts[:mid],
        source_lang,
        target_lang,
        config,
        system_prompt,
        context,
        _depth=_depth + 1,
        _retried=False,
    )

    # Enrich second half context with first half's translations
    split_context_parts = [context] if context else []
    bilingual_preceding = format_bilingual_context(
        texts[:mid][-HISTORY_SIZE:],
        first_half[-HISTORY_SIZE:],
        label="preceding",
    )
    if bilingual_preceding:
        split_context_parts.append(
            "Surrounding context (for reference only):\n" + bilingual_preceding + "\n"
        )
    second_half_context = "\n".join(p for p in split_context_parts if p)
    if second_half_context and not second_half_context.endswith("\n"):
        second_half_context += "\n"

    second_half = process_chunk(
        texts[mid:],
        source_lang,
        target_lang,
        config,
        system_prompt,
        second_half_context,
        _depth=_depth + 1,
        _retried=False,
    )
    return first_half + second_half


def find_chunk_boundaries(
    segments: list[SubtitleSegment],
    chunk_size: int,
    overlap: int,
    scan_range: int = SCAN_RANGE,
) -> list[int]:
    """Compute chunk start indices aligned to sentence boundaries.

    Instead of splitting at fixed intervals, scans ±scan_range segments
    around each computed split point for:
    1. A sentence-ending punctuation mark (. ! ? etc.)
    2. A large timing gap (> TIMING_GAP_THRESHOLD seconds)

    Returns:
        List of start indices for each chunk. The first is always 0.
    """
    if len(segments) <= chunk_size:
        return [0]

    max_chunk_size = chunk_size + scan_range
    step = max(1, chunk_size - overlap)
    starts = [0]
    pos = 0

    while pos + step < len(segments):
        ideal = pos + step

        best = ideal
        best_score = -1

        lo = max(pos + 1, ideal - scan_range)
        hi = min(len(segments) - 1, ideal + scan_range)

        for candidate in range(lo, hi + 1):
            prev_text = segments[candidate - 1].text.strip()
            score = 0

            # Sentence-ending punctuation is the strongest signal
            if prev_text and prev_text[-1] in SENTENCE_END_CHARS:
                score = 2
            # Timing gap is a secondary signal
            elif (
                candidate < len(segments)
                and segments[candidate].start - segments[candidate - 1].end > TIMING_GAP_THRESHOLD
            ):
                score = 1

            # Among equal scores, prefer closer to ideal
            if score > best_score or (
                score == best_score and abs(candidate - ideal) < abs(best - ideal)
            ):
                best = candidate
                best_score = score

        # Safety: don't let chunk exceed max_chunk_size
        if best - pos > max_chunk_size:
            best = pos + step

        starts.append(best)
        pos = best

    return starts


def translate_subtitles(
    segments: list[SubtitleSegment],
    source_lang: str,
    target_lang: str,
    config: LLMConfig,
    chunk_size: int | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> TranslationResult:
    """Translate subtitle segments using an LLM.

    Uses a sliding window with sentence-boundary-aware chunking:
    - Forward overlap: OVERLAP segments translated as lookahead but discarded
    - Backward overlap: BACK_OVERLAP segments re-translated and overwritten,
      so boundary segments end up mid-window with full bidirectional context
    - Split points prefer sentence endings and timing gaps
    - Preceding context shows bilingual pairs (source + translation)
    - Single retry with error feedback before binary split on mismatch
    - Untranslated segments are marked with [?] prefix

    Args:
        segments: Subtitle segments to translate.
        source_lang: Source language code (e.g. "fr").
        target_lang: Target language code (e.g. "en").
        config: LLM configuration.
        chunk_size: Number of segments per LLM call.
        on_progress: Optional callback receiving progress fraction (0.0–1.0).

    Returns:
        TranslationResult with original and translated segments.
    """
    if chunk_size is None:
        chunk_size = _auto_chunk_size(config.model)

    translated: list[SubtitleSegment] = []

    system_prompt = TRANSLATION_SYSTEM.format(
        source_lang=source_lang,
        target_lang=target_lang,
    )

    # Track recent translations for history context
    recent_source: list[str] = []
    recent_translated: list[str] = []

    # --- Issue 3: Sentence-boundary-aware chunking ---
    boundaries = find_chunk_boundaries(segments, chunk_size, overlap=OVERLAP)
    total_chunks = len(boundaries)

    with chunk_progress() as progress:
        task = progress.add_task("Translating subtitles", total=total_chunks)

        for chunk_idx, keep_start in enumerate(boundaries):
            keep_end = (
                boundaries[chunk_idx + 1] if chunk_idx + 1 < len(boundaries) else len(segments)
            )

            # Extend window backward for non-first chunks — re-translate boundary
            # segments that were at the tail of the previous chunk's window
            if chunk_idx > 0:
                translate_start = max(0, keep_start - BACK_OVERLAP)
            else:
                translate_start = keep_start

            # Extend window forward — lookahead for boundary context
            translate_end = min(keep_end + OVERLAP, len(segments))
            chunk = segments[translate_start:translate_end]

            # Build reference context
            context_parts = []

            # Translation history for style consistency
            history = format_history_context(
                recent_source[-HISTORY_SIZE:],
                recent_translated[-HISTORY_SIZE:],
            )
            if history:
                context_parts.append(history)

            # Bilingual overlap context — segments before the translation window
            overlap_parts = []
            if translate_start > 0:
                before_start = max(0, translate_start - OVERLAP)
                before_source = [segments[j].text for j in range(before_start, translate_start)]
                before_trans = [translated[j].text for j in range(before_start, translate_start)]
                bilingual = format_bilingual_context(before_source, before_trans)
                if bilingual:
                    overlap_parts.append(bilingual)

            # Following segments (not yet translated) — source only
            follow_start = translate_end
            follow_end = min(follow_start + OVERLAP, len(segments))
            if follow_start < follow_end:
                following = segments[follow_start:follow_end]
                following_texts = [" ".join(seg.text.split()) for seg in following]
                overlap_parts.append(
                    "following: " + json.dumps(following_texts, ensure_ascii=False)
                )

            if overlap_parts:
                context_parts.append(
                    "Surrounding context (for reference only):\n" + "\n".join(overlap_parts) + "\n"
                )

            context = "\n".join(context_parts) + "\n" if context_parts else ""

            texts = [seg.text for seg in chunk]

            # Skip empty segments — don't send to LLM
            non_empty_idx, non_empty_texts = filter_empty_segments(texts)

            if non_empty_texts:
                try:
                    result_texts = process_chunk(
                        non_empty_texts,
                        source_lang,
                        target_lang,
                        config,
                        system_prompt,
                        context,
                    )
                except Exception as e:
                    warning(f"Translation failed for chunk, keeping original: {e}")
                    result_texts = non_empty_texts
            else:
                result_texts = []

            # Reconstruct full list with empties preserved
            translated_texts = reconstruct_with_empties(texts, non_empty_idx, result_texts)

            # Backward overlap: overwrite previous chunk's tail with better translations
            back_count = keep_start - translate_start
            if back_count > 0:
                overwrite_start = len(translated) - back_count
                for j in range(back_count):
                    seg = chunk[j]
                    new_text = translated_texts[j]
                    if new_text.strip():
                        final_text = new_text
                    else:
                        final_text = translated[overwrite_start + j].text
                    translated[overwrite_start + j] = SubtitleSegment(
                        text=final_text,
                        start=seg.start,
                        end=seg.end,
                        speaker=seg.speaker,
                    )

            # Append new translations for [keep_start:keep_end]
            keep_count = keep_end - keep_start
            for j in range(back_count, back_count + keep_count):
                seg = chunk[j]
                new_text = translated_texts[j]
                if new_text.strip():
                    final_text = new_text
                else:
                    final_text = f"{UNTRANSLATED_MARKER}{seg.text}"
                translated.append(
                    SubtitleSegment(
                        text=final_text,
                        start=seg.start,
                        end=seg.end,
                        speaker=seg.speaker,
                    )
                )

            # Update history — from overwritten + kept translations
            for j in range(back_count + keep_count):
                seg = chunk[j]
                trans = translated_texts[j]
                if seg.text.strip() and trans.strip() and not trans.startswith(UNTRANSLATED_MARKER):
                    recent_source.append(seg.text)
                    recent_translated.append(trans)

            progress.advance(task)
            if on_progress:
                on_progress((chunk_idx + 1) / total_chunks)

    return TranslationResult(
        original=segments,
        translated=translated,
        source_language=source_lang,
        target_language=target_lang,
    )
