"""Vocabulary CSV export — generates Anki-compatible vocabulary lists."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import typer

from pgw.utils.console import console, error, saved
from pgw.utils.paths import GLOB_VOCABULARY_JSON

_COLUMNS = ["word", "lemma", "pos", "difficulty", "count", "context", "translation"]


def export(
    workspace: Path = typer.Argument(..., help="Workspace directory to export vocabulary from."),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV path. Defaults to <workspace>/vocabulary.csv",
    ),
    delimiter: str = typer.Option(
        ";", "--delimiter", "-d", help="CSV delimiter (default: semicolon for Anki)."
    ),
) -> None:
    """Export vocabulary analysis as CSV for Anki or spreadsheet import.

    Reads vocabulary.<lang>.json from the workspace and writes a CSV
    with word, lemma, POS, difficulty tier, occurrence count, context,
    and translation columns.  Suitable for direct import into Anki
    (File → Import, map columns to deck fields).
    """
    workspace = Path(workspace)
    if not workspace.is_dir():
        error(f"Not a directory: {workspace}")
        raise typer.Exit(1)

    vocab_files = sorted(workspace.glob(GLOB_VOCABULARY_JSON))
    if not vocab_files:
        error("No vocabulary.*.json found in workspace. Run 'pgw vocab' first.")
        raise typer.Exit(1)

    summary = json.loads(vocab_files[0].read_text(encoding="utf-8"))
    words = summary.get("top_rare_words", [])
    if not words:
        error("No vocabulary words found in summary.")
        raise typer.Exit(1)

    if output is None:
        output = workspace / "vocabulary.csv"

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=_COLUMNS,
            delimiter=delimiter,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()
        for w in words:
            writer.writerow(
                {
                    "word": w["word"],
                    "lemma": w["lemma"],
                    "pos": w["pos"],
                    "difficulty": w["difficulty"],
                    "count": w["count"],
                    "context": w.get("context", ""),
                    "translation": w.get("translation", ""),
                }
            )

    saved(output)
    console.print(f"[dim]Exported {len(words)} words to {output}[/dim]")
    console.print("[dim]Import into Anki: File → Import → select this CSV[/dim]")
