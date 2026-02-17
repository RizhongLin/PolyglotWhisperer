"""pgw translate command — translate existing subtitle files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from pgw.core.config import load_config
from pgw.subtitles.converter import load_subtitles, save_subtitles

console = Console()


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
    provider: Annotated[
        Optional[str],
        typer.Option(help="LLM provider string (e.g. ollama_chat/qwen3:8b)."),
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
    if provider is not None:
        overrides["llm.provider"] = provider
    overrides["llm.target_language"] = to

    config = load_config(**overrides)

    console.print(f"[bold]Loading subtitles:[/bold] {subtitle_file}")
    segments = load_subtitles(subtitle_file)
    console.print(f"[bold]Segments:[/bold] {len(segments)}")

    result = translate_subtitles(segments, source, to, config.llm)

    # Determine output path
    if output is not None:
        sub_path = output
    else:
        sub_path = subtitle_file.with_suffix(f".{to}.{fmt}")

    save_subtitles(result.translated, sub_path, fmt=fmt)
    console.print(f"[green]Saved:[/green] {sub_path}")

    if not no_txt:
        txt_path = sub_path.with_suffix(".txt")
        if txt_path != sub_path:
            save_subtitles(result.translated, txt_path, fmt="txt")
            console.print(f"[green]Saved:[/green] {txt_path}")
