"""Tests for core data models."""

from pathlib import Path

from pgw.core.models import (
    SubtitleSegment,
    TranscriptionResult,
    TranslationResult,
    VideoSource,
    WordSegment,
)


def test_word_segment_defaults():
    ws = WordSegment(word="bonjour", start=0.0, end=0.5)
    assert ws.word == "bonjour"
    assert ws.score == 0.0


def test_subtitle_segment_defaults():
    seg = SubtitleSegment(text="Bonjour", start=0.0, end=1.0)
    assert seg.words == []
    assert seg.speaker is None


def test_subtitle_segment_with_words():
    words = [
        WordSegment(word="Bonjour", start=0.0, end=0.3, score=0.95),
        WordSegment(word="monde", start=0.4, end=0.7, score=0.90),
    ]
    seg = SubtitleSegment(text="Bonjour monde", start=0.0, end=0.7, words=words)
    assert len(seg.words) == 2
    assert seg.words[0].word == "Bonjour"


def test_transcription_result():
    seg = SubtitleSegment(text="Test", start=0.0, end=1.0)
    result = TranscriptionResult(
        segments=[seg],
        language="fr",
        audio_path=Path("/tmp/audio.wav"),
        model_used="large-v3",
    )
    assert len(result.segments) == 1
    assert result.language == "fr"


def test_translation_result():
    original = [SubtitleSegment(text="Bonjour", start=0.0, end=1.0)]
    translated = [SubtitleSegment(text="Hello", start=0.0, end=1.0)]
    result = TranslationResult(
        original=original,
        translated=translated,
        source_language="fr",
        target_language="en",
    )
    assert result.translated[0].text == "Hello"


def test_video_source_defaults():
    vs = VideoSource(video_path=Path("/tmp/video.mp4"))
    assert vs.audio_path is None
    assert vs.source_url is None
    assert vs.title == ""
    assert vs.duration is None
