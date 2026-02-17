"""Convert between internal models and subtitle file formats."""

from __future__ import annotations

from pathlib import Path

import pysubs2

from pgw.core.models import SubtitleSegment, WordSegment


def segments_to_subs(segments: list[SubtitleSegment]) -> pysubs2.SSAFile:
    """Convert SubtitleSegments to a pysubs2 SSAFile."""
    subs = pysubs2.SSAFile()
    for seg in segments:
        event = pysubs2.SSAEvent(
            start=pysubs2.make_time(s=seg.start),
            end=pysubs2.make_time(s=seg.end),
            text=seg.text,
        )
        subs.events.append(event)
    return subs


def subs_to_segments(subs: pysubs2.SSAFile) -> list[SubtitleSegment]:
    """Convert a pysubs2 SSAFile to SubtitleSegments."""
    segments = []
    for event in subs.events:
        if event.is_comment:
            continue
        segments.append(
            SubtitleSegment(
                text=event.plaintext,
                start=event.start / 1000.0,
                end=event.end / 1000.0,
            )
        )
    return segments


def segments_to_text(segments: list[SubtitleSegment]) -> str:
    """Convert SubtitleSegments to plain text (no timestamps)."""
    return "\n".join(seg.text for seg in segments if seg.text.strip())


def save_subtitles(segments: list[SubtitleSegment], path: Path, fmt: str = "srt") -> Path:
    """Save segments to a subtitle file.

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
        path.write_text(segments_to_text(segments), encoding="utf-8")
    else:
        subs = segments_to_subs(segments)
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
    return subs_to_segments(subs)


def from_whisperx(raw_result: dict) -> list[SubtitleSegment]:
    """Convert raw WhisperX output dict to SubtitleSegments."""
    segments = []
    for seg in raw_result.get("segments", []):
        words = []
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append(
                    WordSegment(
                        word=w["word"],
                        start=w["start"],
                        end=w["end"],
                        score=w.get("score", 0.0),
                    )
                )
        segments.append(
            SubtitleSegment(
                text=seg.get("text", "").strip(),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                words=words,
            )
        )
    return segments
