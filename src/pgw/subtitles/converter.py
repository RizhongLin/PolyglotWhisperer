"""Subtitle conversion utilities.

For transcription output, use stable-ts built-in methods (to_srt_vtt, to_txt,
save_as_json) directly. This module handles:
- Converting stable-ts results to SubtitleSegment for LLM processing
- Saving LLM-modified segments back to subtitle files
- Loading existing subtitle files
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pysubs2

from pgw.core.models import SubtitleSegment

logger = logging.getLogger(__name__)

# VTT inline timestamp cues: <00:00:13.120>
_VTT_CUE_RE = re.compile(r"<\d{2}:\d{2}[:\.][\d.]+>")


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


def save_subtitles(segments: list[SubtitleSegment], path: Path, fmt: str = "vtt") -> Path:
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


def save_bilingual_vtt(
    original: list[SubtitleSegment],
    translated: list[SubtitleSegment],
    path: Path,
) -> Path:
    """Save bilingual subtitles as a single VTT with positioning.

    Original text at bottom (line:85%), translated text at top (line:5%).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(original) != len(translated):
        logger.warning(
            "Bilingual VTT: segment count mismatch (original=%d, translated=%d), truncating",
            len(original),
            len(translated),
        )
    count = min(len(original), len(translated))
    lines = ["WEBVTT", ""]
    for i, (orig, trans) in enumerate(zip(original[:count], translated[:count]), 1):
        start = _format_vtt_time(orig.start)
        end = _format_vtt_time(orig.end)

        # Original at bottom
        lines.append(str(i * 2 - 1))
        lines.append(f"{start} --> {end} line:85%")
        lines.append(orig.text)
        lines.append("")

        # Translation at top
        lines.append(str(i * 2))
        lines.append(f"{start} --> {end} line:5%")
        lines.append(trans.text)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _format_vtt_time(seconds: float) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


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
            text=_strip_vtt_cues(event.plaintext),
            start=event.start / 1000.0,
            end=event.end / 1000.0,
        )
        for event in subs.events
        if not event.is_comment
    ]


def _strip_vtt_cues(text: str) -> str:
    """Strip VTT inline timestamp cues from text.

    stable-ts writes word-level timestamps as VTT cues (e.g. <00:00:13.120>).
    pysubs2 doesn't strip these, so they end up in segment text.
    """
    return _VTT_CUE_RE.sub("", text).strip()
