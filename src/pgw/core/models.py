"""Shared data models for PolyglotWhisperer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SubtitleSegment:
    """A subtitle segment with text and timing, used for LLM processing."""

    text: str
    start: float  # seconds
    end: float  # seconds
    speaker: str | None = None


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
