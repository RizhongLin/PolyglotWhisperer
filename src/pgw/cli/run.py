"""pgw run command — full pipeline from URL/file to dual-subtitle playback."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from pgw.core.config import load_config


def run(
    input_path: Annotated[
        str,
        typer.Argument(help="URL or path to video/audio file."),
    ],
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Source language code (see 'pgw languages')."),
    ] = "fr",
    translate: Annotated[
        Optional[str],
        typer.Option("--translate", "-t", help="Target language for translation."),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Whisper model size."),
    ] = "large-v3-turbo",
    device: Annotated[
        str,
        typer.Option(help="Compute device: cpu, cuda, mps, or auto."),
    ] = "auto",
    cleanup: Annotated[
        bool,
        typer.Option("--cleanup/--no-cleanup", help="Clean up transcription with LLM."),
    ] = False,
    start: Annotated[
        Optional[str],
        typer.Option("--start", help="Start time (e.g. '00:01:00' or '60')."),
    ] = None,
    duration: Annotated[
        Optional[str],
        typer.Option("--duration", help="Duration to process (e.g. '00:05:00' or '300')."),
    ] = None,
    no_play: Annotated[
        bool,
        typer.Option("--no-play", help="Skip video playback after processing."),
    ] = False,
    provider: Annotated[
        Optional[str],
        typer.Option(help="LLM provider (e.g. ollama_chat/qwen3:8b)."),
    ] = None,
) -> None:
    """Run the full pipeline: download, transcribe, translate, and play."""
    from rich.console import Console

    from pgw.core.languages import validate_language
    from pgw.core.pipeline import run_pipeline

    _console = Console()

    try:
        validate_language(language)
    except ValueError as e:
        _console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if translate is not None:
        try:
            validate_language(translate)
        except ValueError as e:
            _console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    overrides: dict[str, object] = {
        "whisper.model_size": model,
        "whisper.language": language,
        "whisper.device": device,
    }
    if provider is not None:
        overrides["llm.provider"] = provider
    if translate is not None:
        overrides["llm.target_language"] = translate

    config = load_config(**overrides)

    run_pipeline(
        input_path=input_path,
        config=config,
        translate=translate,
        cleanup=cleanup,
        play=not no_play,
        start=start,
        duration=duration,
    )
