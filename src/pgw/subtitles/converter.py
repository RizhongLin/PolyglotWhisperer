"""Subtitle conversion utilities.

For transcription output, use stable-ts built-in methods (to_srt_vtt, to_txt,
save_as_json) directly. This module handles:
- Converting stable-ts results to SubtitleSegment for LLM processing
- Saving LLM-modified segments back to subtitle files
- Loading existing subtitle files
"""

from __future__ import annotations

from pathlib import Path

import pysubs2

from pgw.core.models import SubtitleSegment


def result_to_segments(result) -> list[SubtitleSegment]:
    """Convert a stable-ts WhisperResult to SubtitleSegments for LLM processing."""
    segments = []
    for seg in result.segments:
        segments.append(
            SubtitleSegment(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
            )
        )
    return segments


def save_subtitles(segments: list[SubtitleSegment], path: Path, fmt: str = "srt") -> Path:
    """Save LLM-modified SubtitleSegments to a subtitle file.

    Args:
        segments: List of subtitle segments.
        path: Output file path.
        fmt: Format — "srt", "vtt", "ass", or "txt".

    Returns:
        The path the file was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "txt":
        text = "\n".join(seg.text for seg in segments if seg.text.strip())
        path.write_text(text, encoding="utf-8")
    else:
        subs = pysubs2.SSAFile()
        for seg in segments:
            subs.events.append(
                pysubs2.SSAEvent(
                    start=pysubs2.make_time(s=seg.start),
                    end=pysubs2.make_time(s=seg.end),
                    text=seg.text,
                )
            )
        subs.save(str(path))

    return path


def load_subtitles(path: Path) -> list[SubtitleSegment]:
    """Load a subtitle file into SubtitleSegments.

    Supports SRT, VTT, ASS, and plain TXT (one line per segment, no timestamps).
    """
    path = Path(path)

    if path.suffix == ".txt":
        text = path.read_text(encoding="utf-8")
        return [
            SubtitleSegment(text=line.strip(), start=0.0, end=0.0)
            for line in text.splitlines()
            if line.strip()
        ]

    subs = pysubs2.load(str(path))
    return [
        SubtitleSegment(
            text=event.plaintext,
            start=event.start / 1000.0,
            end=event.end / 1000.0,
        )
        for event in subs.events
        if not event.is_comment
    ]
