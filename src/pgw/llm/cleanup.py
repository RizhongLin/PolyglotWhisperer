"""LLM-based subtitle cleanup — fix ASR errors, remove fillers, normalize punctuation."""

from __future__ import annotations

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from pgw.core.config import LLMConfig
from pgw.core.models import SubtitleSegment
from pgw.llm.client import complete
from pgw.llm.prompts import (
    CLEANUP_SYSTEM,
    CLEANUP_USER,
    format_numbered_segments,
    parse_numbered_response,
)

console = Console()

CHUNK_SIZE = 20


def cleanup_subtitles(
    segments: list[SubtitleSegment],
    language: str,
    config: LLMConfig,
    chunk_size: int = CHUNK_SIZE,
) -> list[SubtitleSegment]:
    """Clean up subtitle segments using an LLM.

    Processes segments in chunks to stay within context limits.
    Preserves all timestamps — only text is modified.

    Args:
        segments: Subtitle segments to clean.
        language: Language of the subtitles (e.g. "fr").
        config: LLM configuration.
        chunk_size: Number of segments per LLM call.

    Returns:
        New list of SubtitleSegments with cleaned text.
    """
    cleaned = []
    total_chunks = (len(segments) + chunk_size - 1) // chunk_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TextColumn("{task.completed}/{task.total} chunks"),
        console=console,
    ) as progress:
        task = progress.add_task("Cleaning subtitles", total=total_chunks)

        for i in range(0, len(segments), chunk_size):
            chunk = segments[i : i + chunk_size]
            texts = [seg.text for seg in chunk]

            numbered = format_numbered_segments(texts)
            messages = [
                {"role": "system", "content": CLEANUP_SYSTEM},
                {
                    "role": "user",
                    "content": CLEANUP_USER.format(
                        count=len(texts),
                        language=language,
                        numbered_segments=numbered,
                    ),
                },
            ]

            try:
                response = complete(messages, config)
                cleaned_texts = parse_numbered_response(response, len(texts))
            except Exception as e:
                console.print(f"[yellow]Cleanup failed for chunk, keeping original:[/yellow] {e}")
                cleaned_texts = texts

            for seg, new_text in zip(chunk, cleaned_texts):
                # Fall back to original if LLM returned empty
                final_text = new_text if new_text.strip() else seg.text
                cleaned.append(
                    SubtitleSegment(
                        text=final_text,
                        start=seg.start,
                        end=seg.end,
                        words=seg.words,
                        speaker=seg.speaker,
                    )
                )

            progress.advance(task)

    console.print(f"[green]Cleanup complete:[/green] {len(cleaned)} segments")
    return cleaned
