"""pgw run command — full pipeline from URL/file to dual-subtitle playback."""

from __future__ import annotations

from typing import Annotated, Optional

import typer
from rich.table import Table

from pgw.cli.utils import expand_inputs
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
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model for the active backend."),
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
    from pgw.utils.console import console as _console

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
        "whisper.language": language,
        "whisper.device": device,
    }
    if model is not None:
        model_key = "whisper.api_model" if backend == "api" else "whisper.local_model"
        overrides[model_key] = model
    if llm_model is not None:
        overrides["llm.model"] = llm_model
    if backend is not None:
        overrides["whisper.backend"] = backend
    if translate is not None:
        overrides["llm.target_language"] = translate

    config = load_config(**overrides)

    expanded = expand_inputs(inputs)
    if not expanded:
        _console.print("[red]No inputs resolved. Check your paths or patterns.[/red]")
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
    _console.print(f"[bold]Batch processing {len(expanded)} inputs...[/bold]\n")

    for i, input_path in enumerate(expanded, 1):
        _console.rule(f"[bold][{i}/{len(expanded)}] {input_path}[/bold]")
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
            _console.print(f"[red]Failed:[/red] {e}")
            results.append((input_path, "failed", str(e)))

    # Summary table
    _console.print()
    table = Table(title=f"Batch Results ({len(expanded)} files)")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Input", max_width=50, no_wrap=True)
    table.add_column("Status")
    table.add_column("Output", max_width=50, no_wrap=True)

    succeeded = 0
    for i, (inp, status, output) in enumerate(results, 1):
        style = "green" if status == "success" else "red"
        table.add_row(str(i), inp, f"[{style}]{status}[/{style}]", output)
        if status == "success":
            succeeded += 1

    _console.print(table)
    _console.print(f"\n[bold]{succeeded}/{len(expanded)} succeeded[/bold]")
