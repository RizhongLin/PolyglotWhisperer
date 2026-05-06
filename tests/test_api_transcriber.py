"""Tests for API transcription word regrouping and response parsing."""

from pgw.transcriber.api import _get, regroup_words, response_to_segments


def test_regroup_words_empty():
    assert regroup_words([]) == []


def test_regroup_words_single_word():
    words = [{"word": "Hello", "start": 0.0, "end": 0.5}]
    result = regroup_words(words)
    assert len(result) == 1
    assert result[0].text == "Hello"
    assert result[0].start == 0.0
    assert result[0].end == 0.5


def test_regroup_words_splits_on_sentence_punctuation():
    """Sentence-ending punctuation creates separate groups, but Phase 5b
    may merge short leading groups back. Use enough words to prevent merge."""
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.2},
        {"word": "there", "start": 0.2, "end": 0.4},
        {"word": "friend", "start": 0.4, "end": 0.6},
        {"word": ".", "start": 0.6, "end": 0.7},
        {"word": "World", "start": 1.5, "end": 1.7},  # big gap prevents merge
        {"word": "greetings", "start": 1.7, "end": 2.0},
        {"word": ".", "start": 2.0, "end": 2.1},
    ]
    result = regroup_words(words)
    assert len(result) == 2
    assert "Hello there friend ." in result[0].text


def test_regroup_words_splits_on_speech_gap():
    words = [
        {"word": "First", "start": 0.0, "end": 0.3},
        {"word": "word", "start": 0.3, "end": 0.5},
        {"word": "Second", "start": 1.5, "end": 1.8},  # 1.0s gap
    ]
    result = regroup_words(words)
    assert len(result) == 2
    assert result[0].text == "First word"


def test_regroup_words_merges_trailing_fragment():
    """Short trailing fragments merged into previous segment."""
    words = [
        {"word": "This", "start": 0.0, "end": 0.2},
        {"word": "is", "start": 0.2, "end": 0.3},
        {"word": "a", "start": 0.3, "end": 0.4},
        {"word": "very", "start": 0.4, "end": 0.5},
        {"word": "long", "start": 0.5, "end": 0.6},
        {"word": "sentence", "start": 0.6, "end": 0.7},
        {"word": "and", "start": 0.7, "end": 0.8},
        {"word": "it", "start": 0.8, "end": 0.9},
    ]
    # Each word ~4 chars, so 8 words is ~36 chars — should stay as one segment
    result = regroup_words(words, max_chars=50, max_dur=10.0)
    assert len(result) == 1
    assert len(result[0].text.split()) == 8


def test_regroup_words_splits_on_max_chars():
    """When text exceeds max_chars, it splits."""
    words = [
        {"word": f"word{i}", "start": float(i) * 0.1, "end": float(i) * 0.1 + 0.08}
        for i in range(20)
    ]
    result = regroup_words(words, max_chars=30, max_dur=60.0)
    assert len(result) >= 2


def test_regroup_words_preserves_timestamps():
    words = [
        {"word": "A", "start": 1.0, "end": 1.2},
        {"word": "B", "start": 1.2, "end": 1.4},
    ]
    result = regroup_words(words)
    assert result[0].start == 1.0
    assert result[0].end == 1.4


def test_regroup_words_clamping_prevents_overlap():
    """Phase 6 clamps start times to avoid overlap between segments."""
    # Need multiple segments with timing gaps to create multiple groups
    words = [
        {"word": "First", "start": 0.0, "end": 0.3},
        {"word": "segment", "start": 0.3, "end": 0.5},
        {"word": ".", "start": 0.5, "end": 0.6},
        # Big gap forces a new segment
        {"word": "Second", "start": 1.5, "end": 1.8},
        {"word": "segment", "start": 1.8, "end": 2.0},
        {"word": ".", "start": 2.0, "end": 2.1},
    ]
    result = regroup_words(words)
    assert len(result) >= 2
    # Segments should not overlap
    for i in range(len(result) - 1):
        assert result[i + 1].start >= result[i].end


def test__get_dict():
    d = {"a": 1, "b": 2}
    assert _get(d, "a") == 1
    assert _get(d, "missing", "default") == "default"


class _FakeResponse:
    """Simulate a LiteLLM transcription response object."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return self.__dict__


def test_response_to_segments_with_words():
    resp = _FakeResponse(
        words=[
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]
    )
    segments = response_to_segments(resp)
    assert len(segments) == 1
    assert segments[0].text == "Hello world"


def test_response_to_segments_from_dict():
    resp = {
        "words": [
            {"word": "Hi", "start": 0.0, "end": 0.3},
            {"word": "there", "start": 0.3, "end": 0.6},
        ]
    }
    segments = response_to_segments(resp)
    assert len(segments) == 1
    assert segments[0].text == "Hi there"


def test_response_to_segments_fallback_segments():
    resp = {
        "segments": [
            {"text": "Hello world", "start": 0.0, "end": 1.0},
            {"text": "How are you", "start": 1.0, "end": 2.0},
        ]
    }
    segments = response_to_segments(resp)
    assert len(segments) == 2


def test_response_to_segments_fallback_text():
    resp = {"text": "Full transcript here"}
    segments = response_to_segments(resp)
    assert len(segments) == 1
    assert segments[0].text == "Full transcript here"


def test_response_to_segments_empty():
    segments = response_to_segments({})
    assert segments == []


def test_regroup_words_skips_empty_word_text():
    words = [
        {"word": "", "start": 0.0, "end": 0.1},
        {"word": "real", "start": 0.1, "end": 0.3},
        {"word": "   ", "start": 0.3, "end": 0.4},
        {"word": "word", "start": 0.4, "end": 0.6},
    ]
    result = regroup_words(words)
    assert len(result) == 1
    assert result[0].text == "real word"
