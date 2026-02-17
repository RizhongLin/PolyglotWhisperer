"""Tests for subtitle converter."""

from pathlib import Path

from pgw.core.models import SubtitleSegment
from pgw.subtitles.converter import (
    load_subtitles,
    save_subtitles,
)


def test_save_and_load_srt_roundtrip(tmp_path: Path):
    """SRT save/load roundtrip preserves text and segment count."""
    segments = [
        SubtitleSegment(text="Bonjour", start=1.0, end=4.0),
        SubtitleSegment(text="Monde", start=4.5, end=8.0),
    ]
    srt_path = tmp_path / "test.srt"
    save_subtitles(segments, srt_path, fmt="srt")
    assert srt_path.exists()

    loaded = load_subtitles(srt_path)
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


def test_load_sample_srt(sample_srt: Path):
    """Sample SRT fixture loads correctly."""
    segments = load_subtitles(sample_srt)
    assert len(segments) > 0
    assert all(seg.text for seg in segments)
