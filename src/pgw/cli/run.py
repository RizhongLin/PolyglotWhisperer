"""pgw run command — full pipeline from URL/file to dual-subtitle playback."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from pgw.cli.utils import build_config_overrides, expand_inputs, print_batch_summary
from pgw.core.config import load_config


def run(
    inputs: Annotated[
        list[str],
        typer.Argument(help="URLs, file paths, or glob patterns. Accepts multiple inputs."),
    ],
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Source language code (see 'pgw languages')."),
    ] = "fr",
    translate: Annotated[
        Optional[str],
        typer.Option("--translate", "-t", help="Target language for translation."),
    ] = None,
    whisper_model: Annotated[
        Optional[str],
        typer.Option("--whisper-model", "-w", help="Whisper model (e.g. large-v3-turbo)."),
    ] = None,
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
    llm_model: Annotated[
        Optional[str],
        typer.Option("--llm-model", help="LLM model (e.g. ollama_chat/qwen3:8b)."),
    ] = None,
    llm_backend: Annotated[
        Optional[str],
        typer.Option(help="LLM backend: local or api."),
    ] = None,
    backend: Annotated[
        Optional[str],
        typer.Option(help="Transcription backend: local or api."),
    ] = None,
) -> None:
    """Run the full pipeline: download, transcribe, translate, and play.

    Accepts multiple inputs — files, URLs, glob patterns (*.mp4), or .txt
    files containing one URL/path per line.
    """
    from pgw.core.languages import validate_language
    from pgw.core.pipeline import run_pipeline
    from pgw.utils.console import console

    try:
        validate_language(language)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if translate is not None:
        try:
            validate_language(translate)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    overrides = build_config_overrides(
        language=language,
        device=device,
        whisper_model=whisper_model,
        llm_model=llm_model,
        llm_backend=llm_backend,
        backend=backend,
        translate=translate,
    )
    config = load_config(**overrides)

    expanded = expand_inputs(inputs)
    if not expanded:
        console.print("[red]No inputs resolved. Check your paths or patterns.[/red]")
        raise typer.Exit(1)

    # Single input — original behavior
    if len(expanded) == 1:
        run_pipeline(
            input_path=expanded[0],
            config=config,
            translate=translate,
            cleanup=cleanup,
            play=not no_play,
            start=start,
            duration=duration,
        )
        return

    # Batch mode — process each, collect results
    results: list[tuple[str, str, str]] = []  # (input, status, workspace)
    console.print(f"[bold]Batch processing {len(expanded)} inputs...[/bold]\n")

    for i, input_path in enumerate(expanded, 1):
        console.rule(f"[bold][{i}/{len(expanded)}] {input_path}[/bold]")
        try:
            workspace = run_pipeline(
                input_path=input_path,
                config=config,
                translate=translate,
                cleanup=cleanup,
                play=False,  # Never auto-play in batch mode
                start=start,
                duration=duration,
            )
            results.append((input_path, "success", str(workspace)))
        except Exception as e:
            console.print(f"[red]Failed:[/red] {e}")
            results.append((input_path, "failed", str(e)))

    # Summary table
    console.print()
    print_batch_summary(results, total=len(expanded), show_output=True)
