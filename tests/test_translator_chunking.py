"""Tests for sentence-boundary-aware chunk splitting."""

from pgw.core.models import SubtitleSegment
from pgw.llm.translator import find_chunk_boundaries


def _seg(text: str, start: float, end: float) -> SubtitleSegment:
    return SubtitleSegment(text=text, start=start, end=end)


def test_boundaries_short_input():
    """Input shorter than chunk_size returns a single boundary at 0."""
    segments = [_seg("Hello.", i, i + 1.0) for i in range(5)]
    result = find_chunk_boundaries(segments, chunk_size=10, overlap=2)
    assert result == [0]


def test_boundaries_prefer_sentence_end():
    """Prefer splitting after a segment ending with sentence punctuation."""
    # 20 segments, chunk_size=10, overlap=2 → step=8, ideal split at 8
    # Put a sentence end at index 6 (text at segments[6] ends with '.')
    segments = []
    for i in range(20):
        text = f"Word {i}." if i == 6 else f"Word {i}"
        segments.append(_seg(text, float(i), float(i + 1)))

    result = find_chunk_boundaries(segments, chunk_size=10, overlap=2)
    # Should prefer splitting at index 7 (after segments[6] which ends with '.')
    assert len(result) >= 2
    assert result[1] == 7


def test_boundaries_prefer_timing_gap():
    """Prefer splitting at a large timing gap when no sentence punctuation."""
    # 20 segments, chunk_size=10, overlap=2 → step=8, ideal split at 8
    # No punctuation, but a 2-second gap between segments 7 and 8
    segments = []
    for i in range(20):
        start = float(i) * 1.0
        if i == 8:
            start = 9.5  # 2-second gap after segment 7 (ends at 8.0)
        segments.append(_seg(f"Word {i}", start, start + 0.8))

    result = find_chunk_boundaries(segments, chunk_size=10, overlap=2)
    assert len(result) >= 2
    assert result[1] == 8  # Split at the gap


def test_boundaries_fallback_to_fixed():
    """When no punctuation or gaps, falls back near the ideal stride."""
    segments = [_seg(f"Word {i}", float(i), float(i) + 0.9) for i in range(20)]
    result = find_chunk_boundaries(segments, chunk_size=10, overlap=2)
    assert len(result) >= 2
    # Should be near ideal step of 8
    assert 3 <= result[1] <= 13


def test_boundaries_max_chunk_cap():
    """Never exceeds max_chunk_size from any start position."""
    # All segments have sentence endings — many candidates
    segments = [_seg(f"S{i}.", float(i), float(i) + 0.5) for i in range(100)]
    result = find_chunk_boundaries(segments, chunk_size=10, overlap=2, scan_range=5)
    max_size = 10 + 5  # chunk_size + scan_range
    for i in range(len(result) - 1):
        assert result[i + 1] - result[i] <= max_size


def test_boundaries_covers_all_segments():
    """The last chunk (from last boundary to end) includes all remaining segments."""
    segments = [_seg(f"Word {i}", float(i), float(i) + 0.9) for i in range(50)]
    result = find_chunk_boundaries(segments, chunk_size=10, overlap=2)
    # Last boundary should leave enough room for the remaining segments
    assert result[-1] < len(segments)
    # All segments from 0 to len-1 should be covered
    assert result[0] == 0
