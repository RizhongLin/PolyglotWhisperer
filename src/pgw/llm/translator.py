"""LLM-based subtitle translation with chunked processing."""

from __future__ import annotations

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from pgw.core.config import LLMConfig
from pgw.core.models import SubtitleSegment, TranslationResult
from pgw.llm.client import complete
from pgw.llm.prompts import (
    TRANSLATION_SYSTEM,
    TRANSLATION_USER,
    format_numbered_segments,
    parse_numbered_response,
)

console = Console()

CHUNK_SIZE = 15
OVERLAP = 2  # Context overlap between chunks for coherence


def translate_subtitles(
    segments: list[SubtitleSegment],
    source_lang: str,
    target_lang: str,
    config: LLMConfig,
    chunk_size: int = CHUNK_SIZE,
) -> TranslationResult:
    """Translate subtitle segments using an LLM.

    Processes segments in chunks with overlap for context coherence.
    Translated segments inherit timestamps from originals.

    Args:
        segments: Subtitle segments to translate.
        source_lang: Source language code (e.g. "fr").
        target_lang: Target language code (e.g. "en").
        config: LLM configuration.
        chunk_size: Number of segments per LLM call.

    Returns:
        TranslationResult with original and translated segments.
    """
    translated = []
    total_chunks = (len(segments) + chunk_size - 1) // chunk_size

    system_prompt = TRANSLATION_SYSTEM.format(
        source_lang=source_lang,
        target_lang=target_lang,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TextColumn("{task.completed}/{task.total} chunks"),
        console=console,
    ) as progress:
        task = progress.add_task("Translating subtitles", total=total_chunks)

        for i in range(0, len(segments), chunk_size):
            chunk = segments[i : i + chunk_size]

            # Include overlap context from previous chunk (not included in output)
            context_start = max(0, i - OVERLAP)
            context_segments = segments[context_start:i]
            context_prefix = ""
            if context_segments:
                context_lines = [f"[context] {seg.text}" for seg in context_segments]
                context_prefix = (
                    "For context, here are the preceding lines (do NOT translate these, "
                    "they are just for context):\n"
                    + "\n".join(context_lines)
                    + "\n\nNow translate these:\n"
                )

            texts = [seg.text for seg in chunk]
            numbered = format_numbered_segments(texts)

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": TRANSLATION_USER.format(
                        count=len(texts),
                        source_lang=source_lang,
                        target_lang=target_lang,
                        numbered_segments=context_prefix + numbered,
                    ),
                },
            ]

            try:
                response = complete(messages, config)
                translated_texts = parse_numbered_response(response, len(texts))
            except Exception as e:
                console.print(
                    f"[yellow]Translation failed for chunk, keeping original:[/yellow] {e}"
                )
                translated_texts = texts

            for seg, new_text in zip(chunk, translated_texts):
                final_text = new_text if new_text.strip() else seg.text
                translated.append(
                    SubtitleSegment(
                        text=final_text,
                        start=seg.start,
                        end=seg.end,
                    )
                )

            progress.advance(task)

    console.print(f"[green]Translation complete:[/green] {len(translated)} segments")

    return TranslationResult(
        original=segments,
        translated=translated,
        source_language=source_lang,
        target_language=target_lang,
    )
