"""LLM-based subtitle cleanup — fix ASR errors, remove fillers, normalize punctuation."""

from __future__ import annotations

from typing import Callable

from pgw.core.config import LLMConfig
from pgw.core.models import SubtitleSegment
from pgw.llm.client import complete
from pgw.llm.prompts import (
    CLEANUP_SYSTEM,
    CLEANUP_USER,
    filter_empty_segments,
    format_numbered_segments,
    parse_numbered_response,
    reconstruct_with_empties,
)
from pgw.utils.console import chunk_progress, warning

CHUNK_SIZE = 20
OVERLAP = 2  # Context overlap between chunks for coherence
MAX_RETRY_DEPTH = 3  # Max recursion for binary-split retries


def _process_chunk(
    texts: list[str],
    language: str,
    config: LLMConfig,
    context_prefix: str,
    _depth: int = 0,
) -> list[str]:
    """Process a single chunk, retrying with smaller batches on count mismatch."""
    if _depth >= MAX_RETRY_DEPTH:
        return texts  # Best-effort: return originals to avoid infinite recursion

    numbered = format_numbered_segments(texts)
    messages = [
        {"role": "system", "content": CLEANUP_SYSTEM},
        {
            "role": "user",
            "content": CLEANUP_USER.format(
                count=len(texts),
                language=language,
                numbered_segments=context_prefix + numbered,
            ),
        },
    ]

    response = complete(messages, config)
    cleaned_texts, exact_match = parse_numbered_response(response, len(texts))

    if exact_match or len(texts) <= 2:
        return cleaned_texts

    # Count mismatch — retry with smaller batches
    warning(f"Cleanup count mismatch ({len(texts)} expected), " f"retrying with smaller batches...")
    mid = len(texts) // 2
    first_half = _process_chunk(texts[:mid], language, config, context_prefix, _depth=_depth + 1)
    second_half = _process_chunk(texts[mid:], language, config, "", _depth=_depth + 1)
    return first_half + second_half


def cleanup_subtitles(
    segments: list[SubtitleSegment],
    language: str,
    config: LLMConfig,
    chunk_size: int = CHUNK_SIZE,
    on_progress: Callable[[float], None] | None = None,
) -> list[SubtitleSegment]:
    """Clean up subtitle segments using an LLM.

    Processes segments in chunks with overlap for context coherence.
    Retries with smaller batches on count mismatch.
    Preserves all timestamps — only text is modified.

    Args:
        segments: Subtitle segments to clean.
        language: Language of the subtitles (e.g. "fr").
        config: LLM configuration.
        chunk_size: Number of segments per LLM call.
        on_progress: Optional callback receiving progress fraction (0.0–1.0).

    Returns:
        New list of SubtitleSegments with cleaned text.
    """
    cleaned = []
    total_chunks = (len(segments) + chunk_size - 1) // chunk_size

    with chunk_progress() as progress:
        task = progress.add_task("Cleaning subtitles", total=total_chunks)

        for i in range(0, len(segments), chunk_size):
            chunk = segments[i : i + chunk_size]
            texts = [seg.text for seg in chunk]

            # Build context from surrounding segments (not included in output)
            context_parts = []
            before = segments[max(0, i - OVERLAP) : i]
            after = segments[i + chunk_size : i + chunk_size + OVERLAP]
            if before:
                before_lines = [f"[preceding context] {seg.text}" for seg in before]
                context_parts.append("\n".join(before_lines))
            if after:
                after_lines = [f"[following context] {seg.text}" for seg in after]
                context_parts.append("\n".join(after_lines))

            context_prefix = ""
            if context_parts:
                context_prefix = (
                    "For context only (do NOT clean these, they are just for reference):\n"
                    + "\n".join(context_parts)
                    + "\n\nNow clean these:\n"
                )

            # Skip empty segments — don't send to LLM
            non_empty_idx, non_empty_texts = filter_empty_segments(texts)

            if non_empty_texts:
                try:
                    result_texts = _process_chunk(
                        non_empty_texts,
                        language,
                        config,
                        context_prefix,
                    )
                except Exception as e:
                    warning(f"Cleanup failed for chunk, keeping original: {e}")
                    result_texts = non_empty_texts
            else:
                result_texts = []

            # Reconstruct full list with empties preserved
            cleaned_texts = reconstruct_with_empties(texts, non_empty_idx, result_texts)

            for seg, new_text in zip(chunk, cleaned_texts):
                # Fall back to original if LLM returned empty
                final_text = new_text if new_text.strip() else seg.text
                cleaned.append(
                    SubtitleSegment(
                        text=final_text,
                        start=seg.start,
                        end=seg.end,
                        speaker=seg.speaker,
                    )
                )

            progress.advance(task)
            if on_progress:
                chunk_idx = i // chunk_size + 1
                on_progress(chunk_idx / total_chunks)

    return cleaned
