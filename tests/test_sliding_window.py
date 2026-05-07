"""End-to-end checks for the translator's sliding-window commit logic.

The translator commits each segment exactly once but may translate boundary
segments twice — first as forward-overlap lookahead (discarded) in chunk N,
and again as backward-overlap re-translation in chunk N+1, where it ends up
mid-window with full bidirectional context. These tests pin that behaviour
by tagging each LLM call with a unique chunk index and asserting which
chunk's output ended up in the final ``translated`` list.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from pgw.core.config import LLMConfig
from pgw.core.models import SubtitleSegment
from pgw.llm import translator


def _make_segments(n: int) -> list[SubtitleSegment]:
    """Punctuation-free segments so chunk boundaries fall on the fixed stride."""
    return [SubtitleSegment(text=f"src{i}", start=float(i), end=float(i) + 0.9) for i in range(n)]


@pytest.fixture
def chunk_tagging_mock():
    """Fake LLM that returns a translation tagged with the call's chunk index.

    Reads the segment count straight from ``expected_count`` (the kwarg
    that ``complete()`` always receives) instead of parsing the prompt
    body — so prompt-format changes can never make these tests pass
    vacuously with empty arrays.
    """
    call_count = {"n": 0}

    def fake_complete(messages, config, **kwargs):
        call_count["n"] += 1
        chunk_idx = call_count["n"]
        n = kwargs.get("expected_count", 0)
        assert n > 0, "translator should always pass expected_count > 0"
        return json.dumps({str(i + 1): f"chunk{chunk_idx}" for i in range(n)})

    return fake_complete, call_count


def test_each_segment_translated_exactly_once(chunk_tagging_mock):
    fake, calls = chunk_tagging_mock
    segments = _make_segments(40)
    with patch("pgw.llm.translator.complete", side_effect=fake):
        result = translator.translate_subtitles(segments, "fr", "en", LLMConfig(), chunk_size=15)

    assert len(result.translated) == len(segments)
    for orig, trans in zip(segments, result.translated):
        assert trans.start == orig.start
        assert trans.end == orig.end
        assert trans.text.startswith("chunk")  # never falls through to original
    assert calls["n"] >= 2  # confirms multi-chunk path was exercised


def test_boundary_segments_are_overwritten_by_later_chunk(chunk_tagging_mock):
    """Segments in the back-overlap window should carry the LATER chunk's tag.

    This pins the design: chunk N+1 re-translates the last back_overlap
    segments of chunk N's committed window and overwrites them with its own
    output (which has more preceding bilingual context).
    """
    fake, _ = chunk_tagging_mock
    segments = _make_segments(40)
    cfg = LLMConfig()
    chunk_size, overlap, back_overlap = translator._chunk_params(cfg, 15)
    assert back_overlap >= 1, "test requires non-zero back_overlap"

    with patch("pgw.llm.translator.complete", side_effect=fake):
        result = translator.translate_subtitles(segments, "fr", "en", cfg, chunk_size=chunk_size)

    from pgw.llm.chunking import find_chunk_boundaries

    boundaries = find_chunk_boundaries(
        segments, chunk_size, overlap=overlap, scan_range=translator.SCAN_RANGE
    )
    assert len(boundaries) >= 2, "test requires multi-chunk input"

    # For each non-first chunk, the last back_overlap segments of the prior
    # chunk's commit window must be tagged with THIS chunk's index, not the
    # previous chunk's. Chunk indices in the mock are 1-based.
    for chunk_idx in range(1, len(boundaries)):
        keep_start = boundaries[chunk_idx]
        for offset in range(1, back_overlap + 1):
            i = keep_start - offset
            if i < 0:
                continue
            tag = result.translated[i].text
            assert tag == f"chunk{chunk_idx + 1}", (
                f"segment {i} should be overwritten by chunk {chunk_idx + 1}, " f"got {tag!r}"
            )


def test_forward_overlap_lookahead_is_discarded(chunk_tagging_mock):
    """The forward-overlap lookahead from chunk N must NOT end up in the output.

    Chunk N translates [keep_start_N .. keep_end_N + overlap] but only
    commits [keep_start_N .. keep_end_N]. The lookahead translations are
    thrown away — segments past keep_end_N are produced fresh by chunk N+1.
    """
    fake, _ = chunk_tagging_mock
    segments = _make_segments(40)
    cfg = LLMConfig()
    chunk_size, overlap, back_overlap = translator._chunk_params(cfg, 15)

    with patch("pgw.llm.translator.complete", side_effect=fake):
        result = translator.translate_subtitles(segments, "fr", "en", cfg, chunk_size=chunk_size)

    from pgw.llm.chunking import find_chunk_boundaries

    boundaries = find_chunk_boundaries(
        segments, chunk_size, overlap=overlap, scan_range=translator.SCAN_RANGE
    )

    # The first segment AFTER chunk N's commit boundary must carry chunk N+1's
    # tag (proving the lookahead translation from chunk N was discarded).
    for chunk_idx in range(len(boundaries) - 1):
        keep_end = boundaries[chunk_idx + 1]
        if keep_end >= len(segments):
            continue
        tag = result.translated[keep_end].text
        assert tag == f"chunk{chunk_idx + 2}", (
            f"segment {keep_end} should be tagged by chunk {chunk_idx + 2} "
            f"(forward-overlap discard test), got {tag!r}"
        )


def test_single_chunk_no_overlap_logic(chunk_tagging_mock):
    """When all segments fit in one chunk, no overlap re-translation occurs."""
    fake, calls = chunk_tagging_mock
    segments = _make_segments(8)
    with patch("pgw.llm.translator.complete", side_effect=fake):
        result = translator.translate_subtitles(segments, "fr", "en", LLMConfig(), chunk_size=15)

    assert calls["n"] == 1
    assert all(seg.text == "chunk1" for seg in result.translated)


def test_refine_honors_explicit_chunk_size(chunk_tagging_mock):
    """``refine_subtitles`` must respect an explicit ``chunk_size`` argument.

    Regression: the pipeline used to call ``refine_subtitles`` without
    forwarding ``--chunk-size``, silently falling back to the auto default.
    A small chunk_size against many segments should produce many calls.
    """
    from pgw.llm import refine

    fake, calls = chunk_tagging_mock
    segments = _make_segments(40)
    with patch("pgw.llm.refine.complete", side_effect=fake):
        result = refine.refine_subtitles(segments, "fr", LLMConfig(), chunk_size=10)

    assert len(result) == len(segments)
    assert calls["n"] >= 4, f"chunk_size=10 over 40 segments should split, got {calls['n']} calls"
    assert all(seg.text.startswith("chunk") for seg in result)
