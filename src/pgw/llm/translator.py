"""LLM-based subtitle translation with chunked processing."""

from __future__ import annotations

from typing import Callable

from rich.progress import Progress, SpinnerColumn, TextColumn

from pgw.core.config import LLMConfig
from pgw.core.models import SubtitleSegment, TranslationResult
from pgw.llm.client import complete
from pgw.llm.prompts import (
    TRANSLATION_SYSTEM,
    TRANSLATION_USER,
    format_history_context,
    format_numbered_segments,
    parse_numbered_response,
)
from pgw.utils.console import console

CHUNK_SIZE = 15
OVERLAP = 2  # Context overlap between chunks for coherence
HISTORY_SIZE = 5  # Number of previous translated pairs to include as context


def _process_chunk(
    texts: list[str],
    source_lang: str,
    target_lang: str,
    config: LLMConfig,
    system_prompt: str,
    context_prefix: str,
    _depth: int = 0,
) -> list[str]:
    """Process a single translation chunk, retrying with smaller batches on mismatch."""
    if _depth >= 3:
        return texts  # Best-effort: return originals to avoid infinite recursion

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

    response = complete(messages, config)
    translated_texts, exact_match = parse_numbered_response(response, len(texts))

    if exact_match or len(texts) <= 2:
        return translated_texts

    # Count mismatch — retry with smaller batches
    console.print(
        f"[yellow]Translation count mismatch ({len(texts)} expected), "
        f"retrying with smaller batches...[/yellow]"
    )
    mid = len(texts) // 2
    first_half = _process_chunk(
        texts[:mid],
        source_lang,
        target_lang,
        config,
        system_prompt,
        context_prefix,
        _depth=_depth + 1,
    )
    second_half = _process_chunk(
        texts[mid:],
        source_lang,
        target_lang,
        config,
        system_prompt,
        "",
        _depth=_depth + 1,
    )
    return first_half + second_half


def translate_subtitles(
    segments: list[SubtitleSegment],
    source_lang: str,
    target_lang: str,
    config: LLMConfig,
    chunk_size: int = CHUNK_SIZE,
    on_progress: Callable[[float], None] | None = None,
) -> TranslationResult:
    """Translate subtitle segments using an LLM.

    Processes segments in chunks with overlap for context coherence.
    Includes recent translation pairs as few-shot context for consistency.
    Retries with smaller batches on count mismatch.
    Translated segments inherit timestamps from originals.

    Args:
        segments: Subtitle segments to translate.
        source_lang: Source language code (e.g. "fr").
        target_lang: Target language code (e.g. "en").
        config: LLM configuration.
        chunk_size: Number of segments per LLM call.
        on_progress: Optional callback receiving progress fraction (0.0–1.0).

    Returns:
        TranslationResult with original and translated segments.
    """
    translated = []
    total_chunks = (len(segments) + chunk_size - 1) // chunk_size

    system_prompt = TRANSLATION_SYSTEM.format(
        source_lang=source_lang,
        target_lang=target_lang,
    )

    # Track recent translations for history context
    recent_source: list[str] = []
    recent_translated: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TextColumn("{task.completed}/{task.total} chunks"),
        console=console,
    ) as progress:
        task = progress.add_task("Translating subtitles", total=total_chunks)

        for i in range(0, len(segments), chunk_size):
            chunk = segments[i : i + chunk_size]

            # Include overlap context from surrounding segments
            before = segments[max(0, i - OVERLAP) : i]
            after = segments[i + chunk_size : i + chunk_size + OVERLAP]
            context_parts = []
            if before:
                context_parts.append("\n".join(f"[preceding context] {seg.text}" for seg in before))
            if after:
                context_parts.append("\n".join(f"[following context] {seg.text}" for seg in after))
            context_prefix = ""
            if context_parts:
                context_prefix = (
                    "For context only (do NOT translate these, "
                    "they are just for reference):\n" + "\n".join(context_parts) + "\n\n"
                )

            # Add translation history for style consistency
            history = format_history_context(
                recent_source[-HISTORY_SIZE:],
                recent_translated[-HISTORY_SIZE:],
            )
            if history:
                context_prefix = history + context_prefix

            texts = [seg.text for seg in chunk]

            # Skip empty segments — don't send to LLM
            non_empty_idx = [j for j, t in enumerate(texts) if t.strip()]
            non_empty_texts = [texts[j] for j in non_empty_idx]

            if non_empty_texts:
                try:
                    result_texts = _process_chunk(
                        non_empty_texts,
                        source_lang,
                        target_lang,
                        config,
                        system_prompt,
                        context_prefix,
                    )
                except Exception as e:
                    console.print(
                        f"[yellow]Translation failed for chunk, keeping original:[/yellow] {e}"
                    )
                    result_texts = non_empty_texts
            else:
                result_texts = []

            # Reconstruct full list with empties preserved
            translated_texts = [""] * len(texts)
            for j, idx in enumerate(non_empty_idx):
                translated_texts[idx] = result_texts[j] if j < len(result_texts) else texts[idx]

            for seg, new_text in zip(chunk, translated_texts):
                final_text = new_text if new_text.strip() else seg.text
                translated.append(
                    SubtitleSegment(
                        text=final_text,
                        start=seg.start,
                        end=seg.end,
                        speaker=seg.speaker,
                    )
                )

            # Update history — keep source/translation pairs aligned
            for src, trans in zip(non_empty_texts, result_texts):
                if trans.strip():
                    recent_source.append(src)
                    recent_translated.append(trans)

            progress.advance(task)
            if on_progress:
                chunk_idx = i // chunk_size + 1
                on_progress(chunk_idx / total_chunks)

    console.print(f"[green]Translation complete:[/green] {len(translated)} segments")

    return TranslationResult(
        original=segments,
        translated=translated,
        source_language=source_lang,
        target_language=target_lang,
    )
