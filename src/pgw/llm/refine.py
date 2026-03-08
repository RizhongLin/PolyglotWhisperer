"""LLM-based subtitle refinement — fix ASR errors, fillers, and punctuation.

Aligned with the translation pipeline: sentence-boundary chunking,
forward/backward overlap, JSON I/O, and binary-split retry.
"""

from __future__ import annotations

from typing import Callable

from pgw.core.config import LLMConfig
from pgw.core.models import SubtitleSegment
from pgw.llm.client import complete
from pgw.llm.prompts import (
    REFINE_SYSTEM,
    REFINE_USER,
    filter_empty_segments,
    format_json_segments,
    parse_json_response,
    parse_numbered_response,
    reconstruct_with_empties,
)
from pgw.utils.console import chunk_progress, warning
from pgw.utils.text import SENTENCE_END_CHARS, TIMING_GAP_THRESHOLD, find_sentence_split

CHUNK_SIZE = 30
OVERLAP = 4  # Forward lookahead — refined but discarded, for boundary context
BACK_OVERLAP = 3  # Backward re-refinement — later chunk overwrites previous tail
HISTORY_SIZE = 4  # Preceding refined lines shown as context
MAX_RETRY_DEPTH = 3  # Max recursion for binary-split retries
SCAN_RANGE = 3  # How far to scan for sentence boundaries around ideal split point


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

    response = complete(messages, config, response_format={"type": "json_object"})
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
            f"{len(texts)}. Return a JSON object with keys "
            f'"1" through "{len(texts)}".'
        )
        retry_messages = messages + [
            {"role": "assistant", "content": response},
            {"role": "user", "content": reask_msg},
        ]
        response2 = complete(retry_messages, config, response_format={"type": "json_object"})
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


def find_chunk_boundaries(
    segments: list[SubtitleSegment],
    chunk_size: int,
    overlap: int,
    scan_range: int = SCAN_RANGE,
) -> list[int]:
    """Compute chunk start indices aligned to sentence boundaries."""
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
            if prev_text and prev_text[-1] in SENTENCE_END_CHARS:
                score = 2
            elif (
                candidate < len(segments)
                and segments[candidate].start - segments[candidate - 1].end > TIMING_GAP_THRESHOLD
            ):
                score = 1

            if score > best_score or (
                score == best_score and abs(candidate - ideal) < abs(best - ideal)
            ):
                best = candidate
                best_score = score

        if best - pos > max_chunk_size:
            best = pos + step

        starts.append(best)
        pos = best

    return starts


def refine_subtitles(
    segments: list[SubtitleSegment],
    language: str,
    config: LLMConfig,
    chunk_size: int = CHUNK_SIZE,
    on_progress: Callable[[float], None] | None = None,
) -> list[SubtitleSegment]:
    """Refine subtitle segments using an LLM.

    Uses sentence-boundary-aware chunking with forward/backward overlap:
    - Forward overlap: OVERLAP segments refined as lookahead but discarded
    - Backward overlap: BACK_OVERLAP segments re-refined and overwritten,
      so boundary segments end up mid-window with full context
    - JSON I/O format for reliable parsing
    - Preserves all timestamps — only text is modified
    """
    refined: list[SubtitleSegment] = []

    boundaries = find_chunk_boundaries(segments, chunk_size, overlap=OVERLAP)
    total_chunks = len(boundaries)

    with chunk_progress() as progress:
        task = progress.add_task("Refining subtitles", total=total_chunks)

        for chunk_idx, keep_start in enumerate(boundaries):
            keep_end = (
                boundaries[chunk_idx + 1] if chunk_idx + 1 < len(boundaries) else len(segments)
            )

            # Extend backward for non-first chunks
            if chunk_idx > 0:
                translate_start = max(0, keep_start - BACK_OVERLAP)
            else:
                translate_start = keep_start

            # Extend forward for lookahead context
            translate_end = min(keep_end + OVERLAP, len(segments))
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
            follow_end = min(follow_start + OVERLAP, len(segments))
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
