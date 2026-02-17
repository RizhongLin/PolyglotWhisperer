"""Tests for LLM cleanup and translation with mocked responses."""

from unittest.mock import patch

from pgw.core.config import LLMConfig
from pgw.core.models import SubtitleSegment


def _make_segments(texts: list[str]) -> list[SubtitleSegment]:
    """Create test segments with sequential timestamps."""
    return [
        SubtitleSegment(text=text, start=float(i), end=float(i + 1)) for i, text in enumerate(texts)
    ]


def _mock_complete(messages, config, **kwargs):
    """Mock LLM that returns numbered lines echoing back the input."""
    user_msg = messages[-1]["content"]
    lines = []
    for line in user_msg.splitlines():
        line = line.strip()
        if line and line[0].isdigit() and ". " in line:
            _, text = line.split(". ", 1)
            lines.append(text)
    return "\n".join(f"{i + 1}. [processed] {text}" for i, text in enumerate(lines))


class TestCleanup:
    @patch("pgw.llm.cleanup.complete", side_effect=_mock_complete)
    def test_cleanup_preserves_timestamps(self, _mock):
        from pgw.llm.cleanup import cleanup_subtitles

        segments = _make_segments(["Hello", "World"])
        result = cleanup_subtitles(segments, "fr", LLMConfig())

        for orig, cleaned in zip(segments, result):
            assert cleaned.start == orig.start
            assert cleaned.end == orig.end

    @patch("pgw.llm.cleanup.complete", side_effect=Exception("LLM error"))
    def test_cleanup_fallback_on_error(self, _mock):
        from pgw.llm.cleanup import cleanup_subtitles

        segments = _make_segments(["Keep this", "And this"])
        result = cleanup_subtitles(segments, "fr", LLMConfig())

        assert result[0].text == "Keep this"
        assert result[1].text == "And this"


class TestTranslator:
    @patch("pgw.llm.translator.complete", side_effect=_mock_complete)
    def test_translate_preserves_timestamps(self, _mock):
        from pgw.llm.translator import translate_subtitles

        segments = _make_segments(["Bonjour", "Monde"])
        result = translate_subtitles(segments, "fr", "en", LLMConfig())

        for orig, trans in zip(segments, result.translated):
            assert trans.start == orig.start
            assert trans.end == orig.end

    @patch("pgw.llm.translator.complete", side_effect=Exception("LLM error"))
    def test_translate_fallback_on_error(self, _mock):
        from pgw.llm.translator import translate_subtitles

        segments = _make_segments(["Bonjour", "Monde"])
        result = translate_subtitles(segments, "fr", "en", LLMConfig())

        assert result.translated[0].text == "Bonjour"
        assert result.translated[1].text == "Monde"
