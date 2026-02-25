"""Tests for LLM cleanup and translation with mocked responses."""

from unittest.mock import patch

from conftest import make_segments

from pgw.core.config import LLMConfig


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

        segments = make_segments(["Hello", "World"])
        result = cleanup_subtitles(segments, "fr", LLMConfig())

        for orig, cleaned in zip(segments, result):
            assert cleaned.start == orig.start
            assert cleaned.end == orig.end

    @patch("pgw.llm.cleanup.complete", side_effect=Exception("LLM error"))
    def test_cleanup_fallback_on_error(self, _mock):
        from pgw.llm.cleanup import cleanup_subtitles

        segments = make_segments(["Keep this", "And this"])
        result = cleanup_subtitles(segments, "fr", LLMConfig())

        assert result[0].text == "Keep this"
        assert result[1].text == "And this"


class TestTranslator:
    @patch("pgw.llm.translator.complete", side_effect=_mock_complete)
    def test_translate_preserves_timestamps(self, _mock):
        from pgw.llm.translator import translate_subtitles

        segments = make_segments(["Bonjour", "Monde"])
        result = translate_subtitles(segments, "fr", "en", LLMConfig())

        for orig, trans in zip(segments, result.translated):
            assert trans.start == orig.start
            assert trans.end == orig.end

    @patch("pgw.llm.translator.complete", side_effect=Exception("LLM error"))
    def test_translate_fallback_on_error(self, _mock):
        from pgw.llm.translator import translate_subtitles

        segments = make_segments(["Bonjour", "Monde"])
        result = translate_subtitles(segments, "fr", "en", LLMConfig())

        # On error, originals are returned (not marked since they're non-empty results)
        assert result.translated[0].text == "Bonjour"
        assert result.translated[1].text == "Monde"

    @patch("pgw.llm.translator.complete")
    def test_translate_retry_before_split(self, mock_complete):
        """On count mismatch, retry once before splitting."""
        from pgw.llm.translator import translate_subtitles

        call_count = 0

        def _mock_retry(messages, config, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "1. A\n2. B"  # Wrong count (expected 3)
            else:
                return "1. A\n2. B\n3. C"  # Correct on retry

        mock_complete.side_effect = _mock_retry
        segments = make_segments(["x", "y", "z"])
        result = translate_subtitles(segments, "fr", "en", LLMConfig(), chunk_size=3)
        assert len(result.translated) == 3
        # First attempt + retry = 2 calls (no binary split)
        assert call_count == 2

    @patch("pgw.llm.translator.complete")
    def test_translate_untranslated_marker(self, mock_complete):
        """Segments with empty translations get [?] prefix."""
        from pgw.llm.prompts import UNTRANSLATED_MARKER
        from pgw.llm.translator import translate_subtitles

        # Return only 1 line when 2 expected — retry also returns 1
        mock_complete.return_value = "1. Hello"
        segments = make_segments(["Bonjour", "Monde"])
        result = translate_subtitles(segments, "fr", "en", LLMConfig(), chunk_size=2)
        # Second segment should have untranslated marker
        assert result.translated[1].text.startswith(UNTRANSLATED_MARKER)
