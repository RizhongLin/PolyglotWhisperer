"""Tests for API transcription regrouping and response parsing."""

from pgw.transcriber.api import _regroup_words, _response_to_segments


class TestRegroupWords:
    """Test _regroup_words (all 5 phases)."""

    def test_empty_input(self):
        assert _regroup_words([]) == []

    def test_single_word(self):
        words = [{"word": "Hello", "start": 0.0, "end": 0.5}]
        segments = _regroup_words(words)
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
        segments = _regroup_words(words)
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
        segments = _regroup_words(words)
        assert len(segments) == 2

    def test_phase2_speech_gap_split(self):
        """Words with gaps > 0.5s should be split."""
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 1.5, "end": 2.0},  # 1.0s gap
        ]
        segments = _regroup_words(words)
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
            {"word": "five", "start": 0.8, "end": 1.0},
        ]
        segments = _regroup_words(words)
        texts = [seg.text for seg in segments]
        # "four," ends with comma and has 4 words before it, so should split
        assert "One two three four," in texts
        assert "five" in texts

    def test_phase4_max_chars_split(self):
        """Segments exceeding max_chars should be split."""
        words = [
            {"word": "a" * 30, "start": 0.0, "end": 0.5},
            {"word": "b" * 30, "start": 0.5, "end": 1.0},
        ]
        segments = _regroup_words(words, max_chars=50)
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
        segments = _regroup_words(words, max_dur=5.0)
        assert len(segments) >= 2

    def test_phase5_merge_short_fragments(self):
        """Short fragments (< 3 words) with small gaps should be merged."""
        words = [
            {"word": "Hi.", "start": 0.0, "end": 0.3},
            {"word": "OK", "start": 0.35, "end": 0.5},  # 0.05s gap, short
        ]
        segments = _regroup_words(words)
        # "Hi." triggers sentence split, but "OK" is < 3 words with < 0.15s gap
        # Should merge back
        assert len(segments) == 1
        assert segments[0].text == "Hi. OK"

    def test_whitespace_only_words_skipped(self):
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "  ", "start": 0.5, "end": 0.6},
            {"word": "world", "start": 0.6, "end": 1.0},
        ]
        segments = _regroup_words(words)
        combined_text = " ".join(seg.text for seg in segments)
        assert "  " not in combined_text

    def test_continuous_speech_groups_together(self):
        """Words with no gaps or punctuation should stay together."""
        words = [
            {"word": "I", "start": 0.0, "end": 0.1},
            {"word": "am", "start": 0.1, "end": 0.2},
            {"word": "fine", "start": 0.2, "end": 0.4},
        ]
        segments = _regroup_words(words)
        assert len(segments) == 1
        assert segments[0].text == "I am fine"


class TestResponseToSegments:
    """Test _response_to_segments with 3 fallback tiers."""

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
        segments = _response_to_segments(response)
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
        segments = _response_to_segments(response)
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
        segments = _response_to_segments(response)
        assert len(segments) == 1
        assert segments[0].text == "Hello world"

    def test_empty_response(self):
        response = _MockResponse({"text": ""})
        segments = _response_to_segments(response)
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
        segments = _response_to_segments(response)
        assert len(segments) == 1
        assert segments[0].text == "Hello"


class _MockResponse:
    """Minimal mock for LiteLLM TranscriptionResponse."""

    def __init__(self, data: dict):
        self._data = data

    def model_dump(self) -> dict:
        return self._data
