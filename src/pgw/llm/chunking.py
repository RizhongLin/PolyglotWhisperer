"""Shared chunk-boundary computation for LLM translation and refinement.

Both translation and refinement use the same sentence-boundary-aware splitting
algorithm with configurable overlap and scan ranges.
"""

from __future__ import annotations

import math
import re

from pgw.core.config import LLMConfig
from pgw.core.models import SubtitleSegment
from pgw.utils.text import SENTENCE_END_CHARS, TIMING_GAP_THRESHOLD


def _local_chunk_size_from_model(model: str, cap: int) -> int:
    """Estimate a local Ollama model's chunk size from its parameter count.

    chunk_size = clamp(8 * log2(params_b), 10, cap), where ``params_b`` is
    pulled from a ``\\d+b`` suffix in the model name (e.g. "qwen3:8b" → 8).
    Falls back to ``cap`` when the name has no parseable size.
    """
    match = re.search(r"(\d+(?:\.\d+)?)[bB]", model)
    if not match:
        return cap
    params = float(match.group(1))
    if params <= 0:
        return cap
    size = int(8 * math.log2(max(params, 1)))
    return max(10, min(size, cap))


def resolve_chunk_params(
    config: LLMConfig,
    chunk_size: int | None,
    *,
    api_default: int,
    local_cap: int,
    min_overlap: int,
    min_back_overlap: int,
) -> tuple[int, int, int]:
    """Resolve ``(chunk_size, overlap, back_overlap)`` for one LLM run.

    Precedence (highest first): the explicit ``chunk_size`` argument
    (typically a CLI ``--chunk-size`` flag), ``LLMConfig.chunk_size``
    from TOML/env, then the auto-detected default — ``api_default`` for
    cloud backends or a log-scaled estimate from the local model name.

    Overlaps scale at ~8% / ~5% of the resolved chunk size, but never
    drop below the supplied floors. The floors dominate small chunks
    (translator: ``chunk_size < ~75``; refine: ``< ~50``), so a user
    pinning a tiny chunk for debugging will still see meaningful
    boundary context — at the cost of a high overlap-to-payload ratio.
    """
    if chunk_size is None:
        chunk_size = config.chunk_size
    if chunk_size is None:
        chunk_size = (
            api_default
            if config.backend == "api"
            else _local_chunk_size_from_model(config.model, local_cap)
        )
    overlap = max(min_overlap, round(chunk_size * 0.08))
    back_overlap = max(min_back_overlap, round(chunk_size * 0.05))
    return chunk_size, overlap, back_overlap


def find_chunk_boundaries(
    segments: list[SubtitleSegment],
    chunk_size: int,
    overlap: int,
    scan_range: int,
) -> list[int]:
    """Compute chunk start indices aligned to sentence boundaries.

    Instead of splitting at fixed intervals, scans ±*scan_range* segments
    around each computed split point for:

    1. A sentence-ending punctuation mark (. ! ? etc.)
    2. A large timing gap (> TIMING_GAP_THRESHOLD seconds)

    Args:
        segments: Subtitle segments to split.
        chunk_size: Target number of segments per chunk.
        overlap: Forward overlap count (subtracted from step size).
        scan_range: How far to scan for boundaries around the ideal split.

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
