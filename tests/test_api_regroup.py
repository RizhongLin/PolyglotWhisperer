"""Tests for API transcription regrouping, response parsing, and postprocessing."""

from pgw.core.models import SubtitleSegment
from pgw.transcriber.api import regroup_words, response_to_segments
from pgw.transcriber.postprocess import fix_overlapping_timestamps


class TestRegroupWords:
    """Test regroup_words (all 5 phases)."""

    def test_empty_input(self):
        assert regroup_words([]) == []

    def test_single_word(self):
        words = [{"word": "Hello", "start": 0.0, "end": 0.5}]
        segments = regroup_words(words)
        assert len(segments) == 1
        assert segments[0].text == "Hello"
        assert segments[0].start == 0.0
        assert segments[0].end == 0.5

    def test_phase1_sentence_punctuation_split(self):
        """Words ending with sentence punctuation should start a new segment.

        Use enough words so phase-5 merge doesn't recombine them.
        """
        words = [
            {"word": "One", "start": 0.0, "end": 0.2},
            {"word": "two", "start": 0.2, "end": 0.4},
            {"word": "three.", "start": 0.4, "end": 0.6},
            {"word": "Four", "start": 0.7, "end": 0.9},
            {"word": "five", "start": 0.9, "end": 1.1},
            {"word": "six", "start": 1.1, "end": 1.3},
        ]
        segments = regroup_words(words)
        assert len(segments) == 2
        assert segments[0].text == "One two three."
        assert segments[1].text == "Four five six"

    def test_phase1_question_mark_split(self):
        words = [
            {"word": "One", "start": 0.0, "end": 0.2},
            {"word": "two", "start": 0.2, "end": 0.4},
            {"word": "three?", "start": 0.4, "end": 0.6},
            {"word": "Four", "start": 0.7, "end": 0.9},
            {"word": "five", "start": 0.9, "end": 1.1},
            {"word": "six", "start": 1.1, "end": 1.3},
        ]
        segments = regroup_words(words)
        assert len(segments) == 2

    def test_phase2_speech_gap_split(self):
        """Words with gaps > 0.5s should be split."""
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 1.5, "end": 2.0},  # 1.0s gap
        ]
        segments = regroup_words(words)
        assert len(segments) == 2
        assert segments[0].text == "Hello"
        assert segments[1].text == "world"

    def test_phase3_clause_punctuation_split(self):
        """Comma/semicolon should split if segment has 4+ words."""
        words = [
            {"word": "One", "start": 0.0, "end": 0.2},
            {"word": "two", "start": 0.2, "end": 0.4},
            {"word": "three", "start": 0.4, "end": 0.6},
            {"word": "four,", "start": 0.6, "end": 0.8},
            # Enough trailing words so the fragment isn't merged back
            {"word": "five", "start": 0.8, "end": 1.0},
            {"word": "six", "start": 1.0, "end": 1.2},
            {"word": "seven", "start": 1.2, "end": 1.4},
        ]
        segments = regroup_words(words)
        texts = [seg.text for seg in segments]
        # "four," ends with comma and has 4 words, so should split
        assert "One two three four," in texts

    def test_phase4_max_chars_split(self):
        """Segments exceeding max_chars + merge slack should stay split."""
        words = [
            {"word": "a" * 40, "start": 0.0, "end": 0.5},
            {"word": "b" * 40, "start": 0.5, "end": 1.0},
        ]
        # Combined = 81 chars, exceeds max_chars (50) + MERGE_CHAR_SLACK (15) = 65
        segments = regroup_words(words, max_chars=50)
        assert len(segments) == 2

    def test_phase4_max_duration_split(self):
        """Segments exceeding max_dur should be split."""
        words = [
            {"word": "One", "start": 0.0, "end": 1.0},
            {"word": "two", "start": 1.0, "end": 2.0},
            {"word": "three", "start": 2.0, "end": 3.0},
            {"word": "four", "start": 3.0, "end": 6.0},  # 6s total, > 5
            {"word": "five", "start": 6.0, "end": 7.0},
            {"word": "six", "start": 7.0, "end": 8.0},
            {"word": "seven", "start": 8.0, "end": 9.0},
        ]
        segments = regroup_words(words, max_dur=5.0)
        assert len(segments) >= 2

    def test_phase5_merge_short_fragments(self):
        """Short fragments (< 3 words) with small gaps should be merged."""
        words = [
            {"word": "Hi.", "start": 0.0, "end": 0.3},
            {"word": "OK", "start": 0.35, "end": 0.5},  # 0.05s gap, short
        ]
        segments = regroup_words(words)
        # "Hi." triggers sentence split, but "OK" is < 3 words with < 0.15s gap
        # Should merge back
        assert len(segments) == 1
        assert segments[0].text == "Hi. OK"

    def test_phase5a_merge_trailing_fragment_backward(self):
        """Short trailing fragments should merge into previous segment."""
        # Simulates: "les entreprises dans" + "le flou." — the trailing 2 words
        # should merge back because previous doesn't end with sentence punctuation.
        words = [
            {"word": "les", "start": 0.0, "end": 0.1},
            {"word": "entreprises", "start": 0.1, "end": 0.3},
            {"word": "dans", "start": 0.3, "end": 0.4},
        ]
        # Force max_chars split by using a very small limit, then re-merge
        # Instead, create words that would naturally split then merge:
        words = []
        # Long segment (fills max_chars)
        for i, w in enumerate(["One", "two", "three", "four", "five", "six"]):
            words.append({"word": w, "start": i * 0.2, "end": (i + 1) * 0.2})
        # Short trailing fragment (continuous, no sentence punctuation before it)
        words.append({"word": "seven.", "start": 1.2, "end": 1.4})
        segments = regroup_words(words, max_chars=35)
        # "One two three four five six" = 26 chars, + " seven." = 34 chars <= 35
        # Should merge the trailing "seven." into previous segment
        texts = [seg.text for seg in segments]
        assert any("seven." in t and "six" in t for t in texts)

    def test_phase5a_no_merge_after_sentence_end(self):
        """Don't merge trailing fragment if previous ends with sentence punct."""
        # Use long enough segments so they don't get merged by Phase 5b
        words = [
            {"word": "One", "start": 0.0, "end": 0.1},
            {"word": "two", "start": 0.1, "end": 0.2},
            {"word": "three.", "start": 0.2, "end": 0.4},
            # Gap separates the sentence-ending segment from the trailing fragment
            {"word": "Four", "start": 1.0, "end": 1.1},
            {"word": "five.", "start": 1.1, "end": 1.3},
        ]
        segments = regroup_words(words)
        # "three." ends with sentence punct, "Four five." shouldn't merge back
        assert len(segments) == 2
        assert segments[0].text == "One two three."
        assert segments[1].text == "Four five."

    def test_phase6_overlapping_timestamps_clamped(self):
        """Adjacent segments with overlapping timestamps get clamped."""
        words = [
            {"word": "Hello", "start": 0.0, "end": 1.0},
            {"word": "world.", "start": 0.8, "end": 1.5},  # overlaps with previous
            {"word": "Goodbye", "start": 1.3, "end": 2.0},  # overlaps with previous
        ]
        segments = regroup_words(words)
        for i in range(1, len(segments)):
            assert segments[i].start >= segments[i - 1].end

    def test_whitespace_only_words_skipped(self):
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "  ", "start": 0.5, "end": 0.6},
            {"word": "world", "start": 0.6, "end": 1.0},
        ]
        segments = regroup_words(words)
        combined_text = " ".join(seg.text for seg in segments)
        assert "  " not in combined_text

    def test_continuous_speech_groups_together(self):
        """Words with no gaps or punctuation should stay together."""
        words = [
            {"word": "I", "start": 0.0, "end": 0.1},
            {"word": "am", "start": 0.1, "end": 0.2},
            {"word": "fine", "start": 0.2, "end": 0.4},
        ]
        segments = regroup_words(words)
        assert len(segments) == 1
        assert segments[0].text == "I am fine"


class TestResponseToSegments:
    """Test response_to_segments with 3 fallback tiers."""

    def test_word_level_timestamps(self):
        """Tier 1: word-level timestamps → regrouped segments."""
        response = _MockResponse(
            {
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "world.", "start": 0.5, "end": 1.0},
                ],
            }
        )
        segments = response_to_segments(response)
        assert len(segments) >= 1
        assert "Hello" in segments[0].text

    def test_segment_level_fallback(self):
        """Tier 2: no words → segment-level timestamps."""
        response = _MockResponse(
            {
                "words": [],
                "segments": [
                    {"text": "Hello world", "start": 0.0, "end": 1.0},
                    {"text": "Goodbye", "start": 1.0, "end": 2.0},
                ],
            }
        )
        segments = response_to_segments(response)
        assert len(segments) == 2
        assert segments[0].text == "Hello world"
        assert segments[1].start == 1.0

    def test_full_text_fallback(self):
        """Tier 3: no words or segments → single segment from full text."""
        response = _MockResponse(
            {
                "text": "Hello world",
            }
        )
        segments = response_to_segments(response)
        assert len(segments) == 1
        assert segments[0].text == "Hello world"

    def test_empty_response(self):
        response = _MockResponse({"text": ""})
        segments = response_to_segments(response)
        assert segments == []

    def test_empty_segments_filtered(self):
        """Segments with only whitespace text should be filtered."""
        response = _MockResponse(
            {
                "segments": [
                    {"text": "  ", "start": 0.0, "end": 0.5},
                    {"text": "Hello", "start": 0.5, "end": 1.0},
                ],
            }
        )
        segments = response_to_segments(response)
        assert len(segments) == 1
        assert segments[0].text == "Hello"


class TestFixOverlappingTimestamps:
    """Test fix_overlapping_timestamps on SubtitleSegments."""

    def test_no_overlap(self):
        segments = [
            SubtitleSegment(text="A", start=0.0, end=1.0),
            SubtitleSegment(text="B", start=1.0, end=2.0),
        ]
        result = fix_overlapping_timestamps(segments)
        assert result[0].start == 0.0
        assert result[1].start == 1.0

    def test_overlap_snapped(self):
        segments = [
            SubtitleSegment(text="A", start=0.0, end=1.5),
            SubtitleSegment(text="B", start=1.2, end=2.0),  # 300ms overlap
        ]
        result = fix_overlapping_timestamps(segments)
        assert result[1].start == 1.5  # snapped to previous end
        assert result[1].end == 2.0

    def test_multiple_overlaps(self):
        segments = [
            SubtitleSegment(text="A", start=0.0, end=1.0),
            SubtitleSegment(text="B", start=0.9, end=2.0),
            SubtitleSegment(text="C", start=1.8, end=3.0),
        ]
        result = fix_overlapping_timestamps(segments)
        for i in range(1, len(result)):
            assert result[i].start >= result[i - 1].end

    def test_single_segment(self):
        segments = [SubtitleSegment(text="A", start=0.0, end=1.0)]
        result = fix_overlapping_timestamps(segments)
        assert len(result) == 1

    def test_empty_list(self):
        assert fix_overlapping_timestamps([]) == []


class _MockResponse:
    """Minimal mock for LiteLLM TranscriptionResponse."""

    def __init__(self, data: dict):
        self._data = data

    def model_dump(self) -> dict:
        return self._data
