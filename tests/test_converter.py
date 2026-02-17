"""Tests for subtitle converter."""

from pathlib import Path

from pgw.core.models import SubtitleSegment
from pgw.subtitles.converter import (
    from_whisperx,
    load_subtitles,
    save_subtitles,
    segments_to_subs,
    segments_to_text,
    subs_to_segments,
)


def test_segments_to_subs_roundtrip():
    segments = [
        SubtitleSegment(text="Bonjour", start=1.0, end=4.0),
        SubtitleSegment(text="Au revoir", start=5.0, end=8.0),
    ]
    subs = segments_to_subs(segments)
    assert len(subs.events) == 2
    roundtrip = subs_to_segments(subs)
    assert len(roundtrip) == 2
    assert roundtrip[0].text == "Bonjour"
    assert roundtrip[1].text == "Au revoir"


def test_segments_to_text():
    segments = [
        SubtitleSegment(text="Line one", start=0.0, end=1.0),
        SubtitleSegment(text="", start=1.0, end=2.0),
        SubtitleSegment(text="Line three", start=2.0, end=3.0),
    ]
    text = segments_to_text(segments)
    assert text == "Line one\nLine three"


def test_save_and_load_srt(tmp_path: Path):
    segments = [
        SubtitleSegment(text="Bonjour", start=1.0, end=4.0),
        SubtitleSegment(text="Monde", start=4.5, end=8.0),
    ]
    srt_path = tmp_path / "test.srt"
    save_subtitles(segments, srt_path, fmt="srt")
    assert srt_path.exists()

    loaded = load_subtitles(srt_path)
    assert len(loaded) == 2
    assert loaded[0].text == "Bonjour"
    assert loaded[1].text == "Monde"


def test_save_and_load_txt(tmp_path: Path):
    segments = [
        SubtitleSegment(text="Hello", start=0.0, end=1.0),
        SubtitleSegment(text="World", start=1.0, end=2.0),
    ]
    txt_path = tmp_path / "test.txt"
    save_subtitles(segments, txt_path, fmt="txt")
    assert txt_path.exists()

    loaded = load_subtitles(txt_path)
    assert len(loaded) == 2
    assert loaded[0].text == "Hello"
    assert loaded[0].start == 0.0  # TXT has no timestamps


def test_load_sample_srt(sample_srt: Path):
    segments = load_subtitles(sample_srt)
    assert len(segments) == 3
    assert "Bonjour" in segments[0].text
    assert "titres" in segments[1].text


def test_from_whisperx():
    raw = {
        "segments": [
            {
                "text": "Bonjour le monde",
                "start": 0.0,
                "end": 2.0,
                "words": [
                    {"word": "Bonjour", "start": 0.0, "end": 0.5, "score": 0.95},
                    {"word": "le", "start": 0.6, "end": 0.8, "score": 0.90},
                    {"word": "monde", "start": 0.9, "end": 1.5, "score": 0.92},
                ],
            },
            {
                "text": "Test",
                "start": 3.0,
                "end": 4.0,
                "words": [
                    {"word": "Test"},  # Missing start/end — should be skipped
                ],
            },
        ]
    }
    segments = from_whisperx(raw)
    assert len(segments) == 2
    assert len(segments[0].words) == 3
    assert segments[0].words[0].word == "Bonjour"
    assert len(segments[1].words) == 0  # word without timestamps skipped
