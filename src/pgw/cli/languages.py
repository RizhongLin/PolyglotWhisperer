"""pgw languages command — list supported languages."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from pgw.core.languages import ALIGNMENT_LANGUAGES, WHISPER_LANGUAGES

console = Console()


def languages(
    align_only: Annotated[
        bool,
        typer.Option("--align-only", help="Only show languages with word-level alignment support."),
    ] = False,
) -> None:
    """List all languages supported by Whisper for transcription."""
    table = Table(title=f"Supported Languages ({len(WHISPER_LANGUAGES)})")
    table.add_column("Code", style="bold cyan", width=5)
    table.add_column("Language", width=20)
    table.add_column("Alignment", width=10)

    for code in sorted(WHISPER_LANGUAGES):
        name = WHISPER_LANGUAGES[code].title()
        has_align = code in ALIGNMENT_LANGUAGES

        if align_only and not has_align:
            continue

        align_mark = "yes" if has_align else "-"
        table.add_row(code, name, align_mark)

    console.print(table)
    console.print(
        "\n[dim]Alignment = word-level timestamps via stable-ts. "
        "Languages without alignment can still be transcribed.[/dim]"
    )
