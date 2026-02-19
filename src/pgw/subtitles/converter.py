"""Subtitle conversion utilities.

For transcription output, use stable-ts built-in methods (to_srt_vtt, to_txt,
save_as_json) directly. This module handles:
- Converting stable-ts results to SubtitleSegment for LLM processing
- Saving LLM-modified segments back to subtitle files
- Loading existing subtitle files
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pysubs2

from pgw.core.models import SubtitleSegment

# French elision clitics — Whisper tokenizes "l'école" as ['l', "'", 'école'],
# so clitics can be split across segments in multiple ways:
#   Case 1: "...de l'" / "école"       — clitic+apostrophe at end
#   Case 2: "...de l"  / "'école"      — letter at end, apostrophe on next
#   Case 3: "...de l"  / "'" / "école" — three-way split (fixed by two passes)
# Supports both straight (') and curly (\u2019) apostrophes.
_APOSTROPHE = "'\u2019"

# Case 1: trailing clitic with apostrophe (e.g. "de l'" or "qu'")
_TRAILING_CLITIC_RE = re.compile(r"(?:^|\s)(\S*(?:qu|[ljdnscmt])['\u2019])\s*$", re.IGNORECASE)

# Case 2: trailing clitic letter without apostrophe, next starts with apostrophe
_TRAILING_LETTER_RE = re.compile(r"(?:^|\s)(\S*(?:qu|[ljdnscmt]))\s*$", re.IGNORECASE)


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


def fix_trailing_clitics(segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
    """Move trailing French clitics (l', d', qu', etc.) to the next segment.

    Handles multiple split patterns from Whisper's tokenizer:
    - "...de l'" / "école"  (clitic+apostrophe at end)
    - "...de l" / "'école"  (letter at end, apostrophe starts next)
    - "...de l" / "'" / "école"  (three-way split, fixed by two passes)
    """
    if not segments:
        return segments

    fixed = [copy.copy(seg) for seg in segments]

    # Two passes to handle three-way splits (l / ' / école → l' / école → l'école)
    for _ in range(2):
        for i in range(len(fixed) - 1):
            cur_text = fixed[i].text
            next_text = fixed[i + 1].text

            # Case 1: trailing clitic with apostrophe (e.g. "de l'")
            match = _TRAILING_CLITIC_RE.search(cur_text)
            if match:
                clitic = match.group(1).strip()
                fixed[i].text = cur_text[: match.start()].rstrip()
                fixed[i + 1].text = clitic + next_text
                continue

            # Case 2: trailing letter + next starts with apostrophe
            if next_text and next_text[0] in _APOSTROPHE:
                match = _TRAILING_LETTER_RE.search(cur_text)
                if match:
                    letter = match.group(1).strip()
                    fixed[i].text = cur_text[: match.start()].rstrip()
                    fixed[i + 1].text = letter + next_text
                    continue

    # Remove segments that became empty after clitic removal
    return [seg for seg in fixed if seg.text.strip()]


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

    lines = ["WEBVTT", ""]
    for i, (orig, trans) in enumerate(zip(original, translated, strict=True), 1):
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
            text=event.plaintext,
            start=event.start / 1000.0,
            end=event.end / 1000.0,
        )
        for event in subs.events
        if not event.is_comment
    ]
