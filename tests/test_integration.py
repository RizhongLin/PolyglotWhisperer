"""Integration tests requiring Ollama running locally.

Run with: pytest -m integration
Skipped by default in CI and normal test runs.
"""

import json
import shutil
import subprocess

import pytest
from conftest import make_segments

from pgw.core.config import LLMConfig

pytestmark = pytest.mark.integration

TEST_MODEL = "qwen3:0.6b"
TEST_CONFIG = LLMConfig(local_model=f"ollama_chat/{TEST_MODEL}")

# Segments with sentences split across lines — tests boundary coherence
SPLIT_SENTENCE_SEGMENTS = [
    "Bonsoir, bienvenue dans le journal.",
    "Les taxes douanières plongent les entreprises dans le flou.",
    "Donald Trump a décrété une surtaxe",  # sentence split →
    "de 10% sur toutes les importations.",  # ← continuation
    "La situation pourrait encore évoluer.",
    "Nous tenterons d'y voir plus clair",  # sentence split →
    "dans notre édition spéciale",  # ← three lines →
    "consacrée au commerce international.",  # ← final part
]


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

    ensure_ollama_model(TEST_CONFIG.model)


@skip_no_ollama
def test_refine_real():
    from pgw.llm.refine import refine_subtitles

    segments = make_segments(["Bonjour euh le monde", "il fait beau aujoud'hui"])
    result = refine_subtitles(segments, "fr", TEST_CONFIG, chunk_size=5)

    assert len(result) == len(segments)
    for orig, refined in zip(segments, result):
        assert refined.start == orig.start
        assert refined.text  # non-empty


@skip_no_ollama
def test_translate_real():
    from pgw.llm.translator import translate_subtitles

    segments = make_segments(["Bonjour", "Merci beaucoup"])
    result = translate_subtitles(segments, "fr", "en", TEST_CONFIG, chunk_size=5)

    assert len(result.translated) == len(segments)
    for orig, trans in zip(segments, result.translated):
        assert trans.start == orig.start
        assert trans.text  # non-empty


@skip_no_ollama
def test_json_response_format():
    """Verify the LLM returns valid JSON with the translations key."""
    from pgw.llm.client import complete
    from pgw.llm.prompts import (
        TRANSLATION_SYSTEM,
        TRANSLATION_USER,
        format_json_segments,
    )
    from pgw.llm.translator import parse_response

    segments = SPLIT_SENTENCE_SEGMENTS[:3]  # small batch for speed
    system_prompt = TRANSLATION_SYSTEM.format(
        source_lang="French",
        target_lang="English",
    )
    user_msg = TRANSLATION_USER.format(
        count=len(segments),
        source_lang="French",
        target_lang="English",
        context="",
        json_segments=format_json_segments(segments),
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    response = complete(messages, TEST_CONFIG, response_format={"type": "json_object"})

    # Response should be parseable JSON with numeric keys
    data = json.loads(response.strip())
    assert isinstance(data, dict), f"Expected dict, got {type(data).__name__}"

    # Three-tier parser should extract correct count
    texts, exact = parse_response(response, len(segments))
    assert len(texts) == len(segments)
    assert all(t.strip() for t in texts), f"Empty translations: {texts}"


@skip_no_ollama
def test_refine_json_response_format():
    """Verify refine returns valid keyed JSON."""
    from pgw.llm.client import complete
    from pgw.llm.prompts import REFINE_SYSTEM, REFINE_USER, format_json_segments
    from pgw.llm.refine import parse_response

    segments = SPLIT_SENTENCE_SEGMENTS[:3]
    user_msg = REFINE_USER.format(
        count=len(segments),
        language="French",
        context="",
        json_segments=format_json_segments(segments),
    )
    messages = [
        {"role": "system", "content": REFINE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    response = complete(messages, TEST_CONFIG, response_format={"type": "json_object"})

    data = json.loads(response.strip())
    assert isinstance(data, dict), f"Expected dict, got {type(data).__name__}"

    texts, exact = parse_response(response, len(segments))
    assert len(texts) == len(segments)
    assert all(t.strip() for t in texts), f"Empty refinements: {texts}"


@skip_no_ollama
def test_refine_split_sentences():
    """Refine segments with sentences split across lines.

    Verifies boundary coherence — corrections should use cross-segment
    context (e.g. resolving ambiguous words based on surrounding text).
    """
    from pgw.llm.refine import refine_subtitles

    # Introduce ASR-style errors for refine to fix
    noisy_segments = [
        "Bonsoir, bienvenu dans le journal.",  # bienvenu → bienvenue
        "Les taxes douanière plonge les entreprises dans le flou.",  # agreement errors
        "Donald Trump a décrété une surtaxe",
        "de 10% sur toute les importations.",  # toute → toutes
        "La situation pourrait encore évoluer.",
        "Nous tenterons d'y voir plus clair",
        "dans notre édition spécial",  # spécial → spéciale
        "consacrée au commerce international.",
    ]
    segments = make_segments(noisy_segments)
    result = refine_subtitles(segments, "fr", TEST_CONFIG, chunk_size=20)

    assert len(result) == len(segments)
    for i, (orig, refined) in enumerate(zip(segments, result)):
        assert refined.start == orig.start
        assert refined.text.strip(), f"Segment {i} empty after refine"


@skip_no_ollama
def test_translate_split_sentences():
    """Translate segments with sentences split across lines.

    Verifies that all segments get translated (no empties) and
    that the count matches, exercising the JSON format pipeline
    end-to-end through translate_subtitles().
    """
    from pgw.llm.translator import translate_subtitles

    segments = make_segments(SPLIT_SENTENCE_SEGMENTS)
    result = translate_subtitles(segments, "fr", "en", TEST_CONFIG, chunk_size=20)

    assert len(result.translated) == len(segments)
    for i, (orig, trans) in enumerate(zip(segments, result.translated)):
        assert trans.start == orig.start
        assert trans.text.strip(), f"Segment {i} empty: '{orig.text}' -> '{trans.text}'"


@skip_no_ollama
def test_translate_multi_chunk_overlap():
    """Force multiple chunks with backward overlap via small chunk_size.

    Uses chunk_size=4 on 8 segments to exercise:
    - Sentence-boundary chunking
    - Forward lookahead (translated but discarded)
    - Backward overlap (later chunk overwrites previous tail)
    """
    from pgw.llm.translator import translate_subtitles

    segments = make_segments(SPLIT_SENTENCE_SEGMENTS)
    result = translate_subtitles(segments, "fr", "en", TEST_CONFIG, chunk_size=4)

    assert len(result.translated) == len(segments)
    for i, (orig, trans) in enumerate(zip(segments, result.translated)):
        assert trans.start == orig.start
        assert trans.text.strip(), f"Segment {i} empty: '{orig.text}' -> '{trans.text}'"


@skip_no_ollama
def test_refine_multi_chunk_overlap():
    """Force multiple chunks with backward overlap for refine.

    Uses chunk_size=4 on 8 segments — same as translate test.
    """
    from pgw.llm.refine import refine_subtitles

    segments = make_segments(SPLIT_SENTENCE_SEGMENTS)
    result = refine_subtitles(segments, "fr", TEST_CONFIG, chunk_size=4)

    assert len(result) == len(segments)
    for i, (orig, refined) in enumerate(zip(segments, result)):
        assert refined.start == orig.start
        assert refined.text.strip(), f"Segment {i} empty after refine"
