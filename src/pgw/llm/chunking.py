"""Shared chunk-boundary computation for LLM translation and refinement.

Both translation and refinement use the same sentence-boundary-aware splitting
algorithm with configurable overlap and scan ranges.
"""

from __future__ import annotations

from pgw.core.models import SubtitleSegment
from pgw.utils.text import SENTENCE_END_CHARS, TIMING_GAP_THRESHOLD


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
