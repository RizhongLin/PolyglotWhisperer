"""Tests for subtitle converter."""

from pathlib import Path

import pytest

from pgw.core.models import SubtitleSegment
from pgw.subtitles.converter import (
    load_subtitles,
    result_to_segments,
    save_bilingual_vtt,
    save_subtitles,
)


def test_save_and_load_vtt_roundtrip(tmp_path: Path):
    """VTT save/load roundtrip preserves text and segment count."""
    segments = [
        SubtitleSegment(text="Bonjour", start=1.0, end=4.0),
        SubtitleSegment(text="Monde", start=4.5, end=8.0),
    ]
    vtt_path = tmp_path / "test.vtt"
    save_subtitles(segments, vtt_path, fmt="vtt")
    assert vtt_path.exists()

    loaded = load_subtitles(vtt_path)
    assert len(loaded) == len(segments)
    for orig, ld in zip(segments, loaded):
        assert ld.text == orig.text


def test_save_and_load_txt(tmp_path: Path):
    """TXT save/load preserves text content."""
    segments = [
        SubtitleSegment(text="Hello", start=0.0, end=1.0),
        SubtitleSegment(text="World", start=1.0, end=2.0),
    ]
    txt_path = tmp_path / "test.txt"
    save_subtitles(segments, txt_path, fmt="txt")
    assert txt_path.exists()

    loaded = load_subtitles(txt_path)
    assert len(loaded) == len(segments)
    assert loaded[0].text == "Hello"
    assert loaded[1].text == "World"


def test_save_txt_skips_empty(tmp_path: Path):
    """TXT output omits empty segments."""
    segments = [
        SubtitleSegment(text="Line one", start=0.0, end=1.0),
        SubtitleSegment(text="", start=1.0, end=2.0),
        SubtitleSegment(text="Line three", start=2.0, end=3.0),
    ]
    txt_path = tmp_path / "test.txt"
    save_subtitles(segments, txt_path, fmt="txt")
    loaded = load_subtitles(txt_path)
    assert len(loaded) == 2


def test_load_sample_vtt(sample_vtt: Path):
    """Sample VTT fixture loads correctly."""
    segments = load_subtitles(sample_vtt)
    assert len(segments) > 0
    assert all(seg.text for seg in segments)


def test_bilingual_vtt(tmp_path: Path):
    """Bilingual VTT contains both languages with positioning cues."""
    original = [
        SubtitleSegment(text="Bonjour", start=1.0, end=4.0),
        SubtitleSegment(text="Monde", start=4.5, end=8.0),
    ]
    translated = [
        SubtitleSegment(text="Hello", start=1.0, end=4.0),
        SubtitleSegment(text="World", start=4.5, end=8.0),
    ]
    bi_path = tmp_path / "bilingual.fr-en.vtt"
    save_bilingual_vtt(original, translated, bi_path)
    assert bi_path.exists()

    content = bi_path.read_text()
    assert content.startswith("WEBVTT")
    assert "Bonjour" in content
    assert "Hello" in content
    assert "line:85%" in content
    assert "line:5%" in content


def test_load_result_from_json(tmp_path: Path):
    """Load segments from a stable-ts JSON result file via WhisperResult."""
    import json

    stable_whisper = pytest.importorskip("stable_whisper")

    data = {
        "segments": [
            {
                "text": " Bonjour le monde",
                "start": 0.0,
                "end": 2.5,
                "words": [
                    {"start": 0.0, "end": 0.8, "word": " Bonjour", "probability": 0.95},
                    {"start": 0.8, "end": 1.2, "word": " le", "probability": 0.98},
                    {"start": 1.2, "end": 2.5, "word": " monde", "probability": 0.97},
                ],
            },
            {
                "text": " Comment allez-vous",
                "start": 2.5,
                "end": 5.0,
                "words": [
                    {"start": 2.5, "end": 3.2, "word": " Comment", "probability": 0.96},
                    {"start": 3.2, "end": 4.0, "word": " allez", "probability": 0.94},
                    {"start": 4.0, "end": 5.0, "word": "-vous", "probability": 0.93},
                ],
            },
        ],
        "text": " Bonjour le monde Comment allez-vous",
        "language": "fr",
    }
    json_path = tmp_path / "transcription.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")

    result = stable_whisper.WhisperResult(str(json_path))
    segments = result_to_segments(result)
    assert len(segments) == 2
    assert segments[0].text == "Bonjour le monde"  # stripped
    assert segments[0].start == 0.0
    assert segments[0].end == 2.5
    assert segments[1].text == "Comment allez-vous"


# --- Tests for fix_dangling_clitics (spaCy-based, in postprocess) ---

spacy = pytest.importorskip("spacy")


def _ensure_spacy_model(lang: str, model: str):
    """Load spaCy model, downloading if needed."""
    try:
        return spacy.load(model, disable=["parser", "lemmatizer", "ner"])
    except OSError:
        spacy.cli.download(model)
        return spacy.load(model, disable=["parser", "lemmatizer", "ner"])


@pytest.fixture(scope="module")
def fr_nlp():
    return _ensure_spacy_model("fr", "fr_core_news_sm")


def test_fix_dangling_clitics_trailing_apostrophe(fr_nlp):
    """Trailing token ending with apostrophe (l') is moved to next segment."""
    from pgw.transcriber.postprocess import fix_dangling_clitics

    segments = [
        SubtitleSegment(text="C'est de l'", start=0.0, end=2.0),
        SubtitleSegment(text="école primaire", start=2.0, end=4.0),
    ]
    fixed = fix_dangling_clitics(segments, "fr")
    assert len(fixed) == 2
    assert "l'" not in fixed[0].text
    assert fixed[1].text.startswith("l'")


def test_fix_dangling_clitics_trailing_det(fr_nlp):
    """Trailing determiner (le, la, les) is moved to next segment."""
    from pgw.transcriber.postprocess import fix_dangling_clitics

    segments = [
        SubtitleSegment(text="C'est dans le", start=0.0, end=2.0),
        SubtitleSegment(text="jardin", start=2.0, end=4.0),
    ]
    fixed = fix_dangling_clitics(segments, "fr")
    assert fixed[0].text.rstrip() == "C'est dans"
    assert "le" in fixed[1].text


def test_fix_dangling_clitics_no_match(fr_nlp):
    """Segments without dangling function words are unchanged."""
    from pgw.transcriber.postprocess import fix_dangling_clitics

    segments = [
        SubtitleSegment(text="Bonjour tout le monde", start=0.0, end=2.0),
        SubtitleSegment(text="Comment allez-vous", start=2.0, end=4.0),
    ]
    fixed = fix_dangling_clitics(segments, "fr")
    assert fixed[0].text == "Bonjour tout le monde"
    assert fixed[1].text == "Comment allez-vous"


def test_fix_dangling_clitics_empty_after_removal(fr_nlp):
    """Segment that becomes empty after clitic removal is dropped."""
    from pgw.transcriber.postprocess import fix_dangling_clitics

    segments = [
        SubtitleSegment(text="l'", start=0.0, end=0.5),
        SubtitleSegment(text="école", start=0.5, end=2.0),
    ]
    fixed = fix_dangling_clitics(segments, "fr")
    assert len(fixed) == 1
    assert "école" in fixed[0].text


@pytest.fixture(scope="module")
def it_nlp():
    return _ensure_spacy_model("it", "it_core_news_sm")


@pytest.fixture(scope="module")
def ca_nlp():
    return _ensure_spacy_model("ca", "ca_core_news_sm")


def test_fix_dangling_clitics_italian_apostrophe(it_nlp):
    """Italian elision: trailing l' in "dell'" is moved to next segment."""
    from pgw.transcriber.postprocess import fix_dangling_clitics

    segments = [
        SubtitleSegment(text="la storia dell'", start=0.0, end=2.0),
        SubtitleSegment(text="uomo moderno", start=2.0, end=4.0),
    ]
    fixed = fix_dangling_clitics(segments, "it")
    assert len(fixed) == 2
    assert "dell'" not in fixed[0].text
    assert fixed[1].text.startswith("dell'")


def test_fix_dangling_clitics_italian_det(it_nlp):
    """Italian trailing determiner (il, la, lo) is moved to next segment."""
    from pgw.transcriber.postprocess import fix_dangling_clitics

    segments = [
        SubtitleSegment(text="Questo è il", start=0.0, end=2.0),
        SubtitleSegment(text="problema", start=2.0, end=4.0),
    ]
    fixed = fix_dangling_clitics(segments, "it")
    assert "il" not in fixed[0].text
    assert "il" in fixed[1].text


def test_fix_dangling_clitics_catalan_apostrophe(ca_nlp):
    """Catalan elision: trailing l' is moved to next segment."""
    from pgw.transcriber.postprocess import fix_dangling_clitics

    segments = [
        SubtitleSegment(text="Aquesta és l'", start=0.0, end=2.0),
        SubtitleSegment(text="escola", start=2.0, end=4.0),
    ]
    fixed = fix_dangling_clitics(segments, "ca")
    assert len(fixed) == 2
    assert "l'" not in fixed[0].text
    assert fixed[1].text.startswith("l'")


def test_fix_dangling_clitics_no_spacy_model():
    """Unsupported language gracefully returns segments unchanged."""
    from pgw.transcriber.postprocess import fix_dangling_clitics

    segments = [
        SubtitleSegment(text="Test segment de", start=0.0, end=2.0),
        SubtitleSegment(text="suite", start=2.0, end=4.0),
    ]
    # Use a language code with no spaCy model
    fixed = fix_dangling_clitics(segments, "xx")
    assert len(fixed) == 2
    assert fixed[0].text == "Test segment de"
