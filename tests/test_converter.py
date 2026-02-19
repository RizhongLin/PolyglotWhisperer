"""Tests for subtitle converter."""

from pathlib import Path

import pytest

from pgw.core.models import SubtitleSegment
from pgw.subtitles.converter import (
    fix_trailing_clitics,
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


def test_fix_trailing_clitics_basic():
    """Trailing l' is moved to next segment."""
    segments = [
        SubtitleSegment(text="C'est de l'", start=0.0, end=2.0),
        SubtitleSegment(text="école primaire", start=2.0, end=4.0),
    ]
    fixed = fix_trailing_clitics(segments)
    assert len(fixed) == 2
    assert fixed[0].text == "C'est de"
    assert fixed[1].text == "l'école primaire"


def test_fix_trailing_clitics_multiple():
    """Handles d', qu', j' and other clitics."""
    segments = [
        SubtitleSegment(text="Il parle d'", start=0.0, end=1.5),
        SubtitleSegment(text="abord de qu'", start=1.5, end=3.0),
        SubtitleSegment(text="est-ce que", start=3.0, end=4.5),
    ]
    fixed = fix_trailing_clitics(segments)
    assert fixed[0].text == "Il parle"
    assert fixed[1].text == "d'abord de"
    assert fixed[2].text == "qu'est-ce que"


def test_fix_trailing_clitics_no_match():
    """Segments without trailing clitics are unchanged."""
    segments = [
        SubtitleSegment(text="Bonjour tout le monde", start=0.0, end=2.0),
        SubtitleSegment(text="Comment allez-vous", start=2.0, end=4.0),
    ]
    fixed = fix_trailing_clitics(segments)
    assert fixed[0].text == "Bonjour tout le monde"
    assert fixed[1].text == "Comment allez-vous"


def test_fix_trailing_clitics_empty_after_removal():
    """Segment that becomes empty after clitic removal is dropped."""
    segments = [
        SubtitleSegment(text="l'", start=0.0, end=0.5),
        SubtitleSegment(text="école", start=0.5, end=2.0),
    ]
    fixed = fix_trailing_clitics(segments)
    assert len(fixed) == 1
    assert fixed[0].text == "l'école"


def test_fix_trailing_clitics_curly_apostrophe():
    """Curly apostrophe (\u2019) is handled the same as straight."""
    segments = [
        SubtitleSegment(text="C\u2019est de l\u2019", start=0.0, end=2.0),
        SubtitleSegment(text="\u00e9cole", start=2.0, end=4.0),
    ]
    fixed = fix_trailing_clitics(segments)
    assert fixed[0].text == "C\u2019est de"
    assert fixed[1].text == "l\u2019\u00e9cole"


def test_fix_trailing_clitics_split_apostrophe():
    """Case 2: letter at end, apostrophe starts next segment."""
    segments = [
        SubtitleSegment(text="C'est de l", start=0.0, end=2.0),
        SubtitleSegment(text="'\u00e9cole primaire", start=2.0, end=4.0),
    ]
    fixed = fix_trailing_clitics(segments)
    assert fixed[0].text == "C'est de"
    assert fixed[1].text == "l'\u00e9cole primaire"


def test_fix_trailing_clitics_three_way_split():
    """Case 3: letter / apostrophe / word across three segments."""
    segments = [
        SubtitleSegment(text="de l", start=0.0, end=1.0),
        SubtitleSegment(text="'", start=1.0, end=1.2),
        SubtitleSegment(text="\u00e9cole", start=1.2, end=3.0),
    ]
    fixed = fix_trailing_clitics(segments)
    # Pass 1: "l" merges with "'" → ["de", "l'", "école"]
    # Pass 2: "l'" merges with "école" → ["de", "l'école"]
    assert len(fixed) == 2
    assert fixed[0].text == "de"
    assert fixed[1].text == "l'\u00e9cole"
