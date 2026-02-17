"""Integration tests requiring Ollama running locally.

Run with: pytest -m integration
Skipped by default in CI and normal test runs.
"""

import shutil
import subprocess

import pytest

from pgw.core.config import LLMConfig
from pgw.core.models import SubtitleSegment

pytestmark = pytest.mark.integration

TEST_MODEL = "qwen3:0.6b"
TEST_CONFIG = LLMConfig(provider=f"ollama_chat/{TEST_MODEL}")


def ollama_available() -> bool:
    """Check if Ollama is running and reachable."""
    if not shutil.which("ollama"):
        return False
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


skip_no_ollama = pytest.mark.skipif(not ollama_available(), reason="Ollama not running")


@pytest.fixture(scope="session", autouse=True)
def _ensure_test_model():
    """Auto-pull the test model before running integration tests."""
    if not ollama_available():
        return
    from pgw.llm.client import ensure_ollama_model

    ensure_ollama_model(TEST_CONFIG.provider)


def _segments(texts: list[str]) -> list[SubtitleSegment]:
    return [SubtitleSegment(text=t, start=float(i), end=float(i + 1)) for i, t in enumerate(texts)]


@skip_no_ollama
def test_cleanup_real():
    from pgw.llm.cleanup import cleanup_subtitles

    segments = _segments(["Bonjour euh le monde", "il fait beau aujoud'hui"])
    result = cleanup_subtitles(segments, "fr", TEST_CONFIG, chunk_size=5)

    assert len(result) == len(segments)
    for orig, cleaned in zip(segments, result):
        assert cleaned.start == orig.start
        assert cleaned.text  # non-empty


@skip_no_ollama
def test_translate_real():
    from pgw.llm.translator import translate_subtitles

    segments = _segments(["Bonjour", "Merci beaucoup"])
    result = translate_subtitles(segments, "fr", "en", TEST_CONFIG, chunk_size=5)

    assert len(result.translated) == len(segments)
    for orig, trans in zip(segments, result.translated):
        assert trans.start == orig.start
        assert trans.text  # non-empty
