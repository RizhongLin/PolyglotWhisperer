"""pgw translate command — translate existing subtitle files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from pgw.core.config import load_config
from pgw.subtitles.converter import load_subtitles, save_subtitles
from pgw.utils.console import console, stage


def translate(
    subtitle_file: Annotated[
        Path,
        typer.Argument(help="Path to subtitle file (SRT, VTT, ASS, TXT)."),
    ],
    to: Annotated[
        str,
        typer.Option("--to", "-t", help="Target language code (run 'pgw languages' to list)."),
    ] = "en",
    source: Annotated[
        str,
        typer.Option("--from", "-s", help="Source language code (run 'pgw languages' to list)."),
    ] = "fr",
    llm_model: Annotated[
        Optional[str],
        typer.Option("--llm-model", help="LLM model (e.g. ollama_chat/qwen3:8b)."),
    ] = None,
    llm_backend: Annotated[
        Optional[str],
        typer.Option(help="LLM backend: local or api."),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path."),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: vtt, srt, ass, txt."),
    ] = "vtt",
    no_txt: Annotated[
        bool,
        typer.Option("--no-txt", help="Skip generating plain text file."),
    ] = False,
) -> None:
    """Translate a subtitle file to another language using an LLM."""
    from pgw.core.languages import validate_language
    from pgw.llm.translator import translate_subtitles

    for code in (source, to):
        try:
            validate_language(code)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    if not subtitle_file.is_file():
        console.print(f"[red]File not found:[/red] {subtitle_file}")
        raise typer.Exit(1)

    overrides = {}
    if llm_model is not None:
        model_key = "llm.api_model" if llm_backend == "api" else "llm.local_model"
        overrides[model_key] = llm_model
    if llm_backend is not None:
        overrides["llm.backend"] = llm_backend
    overrides["llm.target_language"] = to

    config = load_config(**overrides)

    segments = load_subtitles(subtitle_file)
    stage("Loading", f"{subtitle_file.name} ({len(segments)} segments)")
    stage(f"Translating to {to}", config.llm.model)

    result = translate_subtitles(segments, source, to, config.llm)

    # Determine output path
    if output is not None:
        sub_path = output
    else:
        sub_path = subtitle_file.with_suffix(f".{to}.{fmt}")

    save_subtitles(result.translated, sub_path, fmt=fmt)
    saved_names = [sub_path.name]

    if not no_txt:
        txt_path = sub_path.with_suffix(".txt")
        if txt_path != sub_path:
            save_subtitles(result.translated, txt_path, fmt="txt")
            saved_names.append(txt_path.name)

    console.print(f"\n[green]Saved:[/green] {'  '.join(saved_names)}")
