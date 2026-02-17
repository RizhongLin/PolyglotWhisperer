"""Shared data models for PolyglotWhisperer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WordSegment:
    """A single word with precise timing from forced alignment."""

    word: str
    start: float
    end: float
    score: float = 0.0


@dataclass
class SubtitleSegment:
    """A subtitle segment with text, timing, and optional word-level detail."""

    text: str
    start: float  # seconds
    end: float  # seconds
    words: list[WordSegment] = field(default_factory=list)
    speaker: str | None = None


@dataclass
class TranscriptionResult:
    """Output from the transcription engine."""

    segments: list[SubtitleSegment]
    language: str
    audio_path: Path
    model_used: str


@dataclass
class TranslationResult:
    """Output from the translation engine."""

    original: list[SubtitleSegment]
    translated: list[SubtitleSegment]
    source_language: str
    target_language: str


@dataclass
class VideoSource:
    """Resolved video source — either local file or downloaded."""

    video_path: Path
    audio_path: Path | None = None
    source_url: str | None = None
    title: str = ""
    duration: float | None = None
