"""LLM-based subtitle refinement — fix ASR errors, fillers, and punctuation.

Aligned with the translation pipeline: sentence-boundary chunking,
forward/backward overlap, JSON I/O, and binary-split retry.
"""

from __future__ import annotations

from typing import Callable

from pgw.core.config import LLMConfig
from pgw.core.models import SubtitleSegment
from pgw.llm.chunking import find_chunk_boundaries, resolve_chunk_params
from pgw.llm.client import complete
from pgw.llm.prompts import (
    REFINE_SYSTEM,
    REFINE_USER,
    build_refine_schema,
    filter_empty_segments,
    format_json_segments,
    parse_json_response,
    parse_numbered_response,
    reconstruct_with_empties,
)
from pgw.utils.console import chunk_progress, warning
from pgw.utils.text import find_sentence_split

# Cloud/API default — refinement preserves wording closely so it can
# run with larger chunks than translation, but stays a step below to
# leave headroom for the 1:1 keyed-JSON constraint.
API_CHUNK_SIZE = 100

# Local Ollama models lag on long structured output. Cap is reached at
# ~64B (formula = int(8 * log2(64)) = 48), so any local model larger
# than that clamps here.
LOCAL_CHUNK_SIZE_CAP = 48

# Floors for boundary context. These dominate the proportional 8%/5%
# scaling for chunk_size below ~50.
MIN_OVERLAP = 4
MIN_BACK_OVERLAP = 3

HISTORY_SIZE = 4  # Preceding refined lines shown as context
MAX_RETRY_DEPTH = 3  # Max recursion for binary-split retries
SCAN_RANGE = 3  # How far to scan for sentence boundaries around ideal split point


def _chunk_params(config: LLMConfig, chunk_size: int | None = None) -> tuple[int, int, int]:
    """Refine wrapper around the shared resolver."""
    return resolve_chunk_params(
        config,
        chunk_size,
        api_default=API_CHUNK_SIZE,
        local_cap=LOCAL_CHUNK_SIZE_CAP,
        min_overlap=MIN_OVERLAP,
        min_back_overlap=MIN_BACK_OVERLAP,
    )


def parse_response(response: str, expected_count: int) -> tuple[list[str], bool]:
    """Parse refine response: JSON first, then numbered fallback."""
    texts, exact = parse_json_response(response, expected_count)
    if exact:
        return texts, True
    if texts:
        return texts, False
    return parse_numbered_response(response, expected_count)


def _process_chunk(
    texts: list[str],
    language: str,
    config: LLMConfig,
    context: str,
    _depth: int = 0,
    _retried: bool = False,
) -> list[str]:
    """Process a single refinement chunk with retry-before-split strategy."""
    if _depth >= MAX_RETRY_DEPTH:
        return texts

    json_segments = format_json_segments(texts)

    messages = [
        {"role": "system", "content": REFINE_SYSTEM},
        {
            "role": "user",
            "content": REFINE_USER.format(
                count=len(texts),
                language=language,
                context=context,
                json_segments=json_segments,
            ),
        },
    ]

    response = complete(
        messages,
        config,
        json_schema=build_refine_schema(len(texts)),
        expected_count=len(texts),
    )
    refined_texts, exact_match = parse_response(response, len(texts))

    if exact_match or len(texts) <= 2:
        return refined_texts

    # Single retry before split
    if not _retried:
        parsed_count = sum(1 for t in refined_texts if t)
        warning(
            f"Refine count mismatch ({parsed_count} vs {len(texts)} expected), "
            f"retrying same chunk..."
        )
        reask_msg = (
            f"You returned {parsed_count} items but I need exactly "
            f"{len(texts)}. Return a JSON object with a "
            f'"refined" array of exactly {len(texts)} items.'
        )
        retry_messages = messages + [
            {"role": "assistant", "content": response},
            {"role": "user", "content": reask_msg},
        ]
        response2 = complete(
            retry_messages,
            config,
            json_schema=build_refine_schema(len(texts)),
            expected_count=len(texts),
        )
        refined_texts2, exact_match2 = parse_response(response2, len(texts))
        if exact_match2:
            return refined_texts2

    # Binary split with sentence-boundary-aware split point
    warning(f"Splitting into smaller batches ({len(texts)} segments)...")
    mid = find_sentence_split(texts)
    first_half = _process_chunk(texts[:mid], language, config, context, _depth=_depth + 1)

    # Build context for second half from first half's results
    second_context_parts = [context] if context else []
    preceding_lines = [f"[preceding] {line}" for line in first_half[-HISTORY_SIZE:]]
    if preceding_lines:
        second_context_parts.append(
            "Previously refined (for reference only):\n" + "\n".join(preceding_lines)
        )
    second_context = "\n".join(p for p in second_context_parts if p)
    if second_context and not second_context.endswith("\n"):
        second_context += "\n"

    second_half = _process_chunk(texts[mid:], language, config, second_context, _depth=_depth + 1)
    return first_half + second_half


def refine_subtitles(
    segments: list[SubtitleSegment],
    language: str,
    config: LLMConfig,
    chunk_size: int | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> list[SubtitleSegment]:
    """Refine subtitle segments using an LLM.

    Uses sentence-boundary-aware chunking with proportional forward/backward
    overlap (~8% / ~5% of chunk_size). JSON I/O for reliable parsing.
    Preserves all timestamps — only text is modified.
    """
    chunk_size, overlap, back_overlap = _chunk_params(config, chunk_size)
    refined: list[SubtitleSegment] = []

    boundaries = find_chunk_boundaries(segments, chunk_size, overlap=overlap, scan_range=SCAN_RANGE)
    total_chunks = len(boundaries)

    with chunk_progress() as progress:
        task = progress.add_task("Refining subtitles", total=total_chunks)

        for chunk_idx, keep_start in enumerate(boundaries):
            keep_end = (
                boundaries[chunk_idx + 1] if chunk_idx + 1 < len(boundaries) else len(segments)
            )

            # Extend backward for non-first chunks
            if chunk_idx > 0:
                translate_start = max(0, keep_start - back_overlap)
            else:
                translate_start = keep_start

            # Extend forward for lookahead context
            translate_end = min(keep_end + overlap, len(segments))
            chunk = segments[translate_start:translate_end]

            # Build context from surrounding segments
            context_parts = []

            # Preceding refined segments as read-only context
            if translate_start > 0:
                before_start = max(0, translate_start - HISTORY_SIZE)
                before_lines = [
                    f"[preceding] {refined[j].text}" for j in range(before_start, translate_start)
                ]
                if before_lines:
                    context_parts.append(
                        "Previously refined (for reference only):\n" + "\n".join(before_lines)
                    )

            # Following segments as read-only context
            follow_start = translate_end
            follow_end = min(follow_start + overlap, len(segments))
            if follow_start < follow_end:
                following = segments[follow_start:follow_end]
                after_lines = [f"[following] {seg.text}" for seg in following]
                context_parts.append(
                    "Following segments (for reference only):\n" + "\n".join(after_lines)
                )

            context = "\n".join(context_parts) + "\n" if context_parts else ""

            texts = [seg.text for seg in chunk]

            # Skip empty segments
            non_empty_idx, non_empty_texts = filter_empty_segments(texts)

            if non_empty_texts:
                try:
                    result_texts = _process_chunk(non_empty_texts, language, config, context)
                except Exception as e:
                    warning(f"Refinement failed for chunk, keeping original: {e}")
                    result_texts = non_empty_texts
            else:
                result_texts = []

            refined_texts = reconstruct_with_empties(texts, non_empty_idx, result_texts)

            # Backward overlap: overwrite previous chunk's tail
            back_count = keep_start - translate_start
            if back_count > 0:
                overwrite_start = len(refined) - back_count
                for j in range(back_count):
                    seg = chunk[j]
                    new_text = refined_texts[j]
                    if new_text.strip():
                        final_text = new_text
                    else:
                        final_text = refined[overwrite_start + j].text
                    refined[overwrite_start + j] = SubtitleSegment(
                        text=final_text,
                        start=seg.start,
                        end=seg.end,
                        speaker=seg.speaker,
                    )

            # Append new segments for [keep_start:keep_end]
            keep_count = keep_end - keep_start
            for j in range(back_count, back_count + keep_count):
                seg = chunk[j]
                new_text = refined_texts[j]
                final_text = new_text if new_text.strip() else seg.text
                refined.append(
                    SubtitleSegment(
                        text=final_text,
                        start=seg.start,
                        end=seg.end,
                        speaker=seg.speaker,
                    )
                )

            progress.advance(task)
            if on_progress:
                on_progress((chunk_idx + 1) / total_chunks)

    return refined
