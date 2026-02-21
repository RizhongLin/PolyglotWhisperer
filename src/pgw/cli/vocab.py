"""Vocabulary analysis CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.table import Table

from pgw.utils.console import console


def vocab(
    workspace: Path = typer.Argument(..., help="Workspace directory to analyze."),
    language: str | None = typer.Option(None, "-l", "--language", help="Override language code."),
    top: int = typer.Option(30, "--top", help="Number of rare words to show."),
) -> None:
    """Show vocabulary summary for a processed workspace."""
    workspace = Path(workspace)
    if not workspace.is_dir():
        console.print(f"[red]Not a directory:[/red] {workspace}")
        raise typer.Exit(1)

    # Try loading existing summary
    summary = _load_existing_summary(workspace, language)

    if summary is None:
        # Generate from subtitle files
        summary = _generate_from_workspace(workspace, language, top)

    if summary is None:
        console.print("[red]No subtitle files found in workspace.[/red]")
        raise typer.Exit(1)

    _display_summary(summary, top)


def _load_existing_summary(workspace: Path, language: str | None) -> dict | None:
    """Try loading a pre-generated vocabulary JSON from the workspace."""
    if language:
        path = workspace / f"vocabulary.{language}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    else:
        # Find any vocabulary.*.json
        for path in sorted(workspace.glob("vocabulary.*.json")):
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _generate_from_workspace(workspace: Path, language: str | None, top: int) -> dict | None:
    """Generate vocabulary summary from subtitle files in workspace."""
    from pgw.subtitles.converter import load_subtitles

    # Detect language from subtitle filename if not specified
    vtt_files = list(workspace.glob("transcription.*.vtt"))
    if not vtt_files:
        return None

    vtt_path = vtt_files[0]
    if language is None:
        # Extract language code from filename: transcription.fr.vtt → fr
        parts = vtt_path.stem.split(".")
        language = parts[-1] if len(parts) > 1 else "en"

    segments = load_subtitles(vtt_path)

    # Try loading translations
    translated_segments = None
    trans_files = list(workspace.glob("translation.*.vtt"))
    if trans_files:
        translated_segments = load_subtitles(trans_files[0])

    from pgw.vocab.summary import generate_vocab_summary

    console.print(f"[bold]Analyzing vocabulary ({language})...[/bold]")
    summary = generate_vocab_summary(
        segments,
        language,
        translated_segments=translated_segments,
        top_n=top,
    )

    # Save for future use
    out_path = workspace / f"vocabulary.{language}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]Saved:[/green] {out_path}")

    return summary


def _display_summary(summary: dict, top: int) -> None:
    """Display vocabulary summary as Rich tables."""
    console.print()
    console.print(f"[bold]Language:[/bold] {summary['language']}")
    console.print(f"[bold]Total words:[/bold] {summary['total_words']:,}")
    console.print(f"[bold]Unique words:[/bold] {summary['unique_words']:,}")
    console.print(f"[bold]Unique lemmas:[/bold] {summary['unique_lemmas']:,}")
    console.print(f"[bold]Estimated level:[/bold] {summary['estimated_level']}")

    # CEFR distribution
    console.print()
    dist = summary["cefr_distribution"]
    dist_table = Table(title="CEFR Distribution")
    for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        dist_table.add_column(level, justify="right")
    dist_table.add_row(*[str(dist.get(level, 0)) for level in ["A1", "A2", "B1", "B2", "C1", "C2"]])
    console.print(dist_table)

    # Top rare words
    rare = summary.get("top_rare_words", [])[:top]
    if rare:
        console.print()
        word_table = Table(title=f"Top {len(rare)} Rare Words")
        word_table.add_column("Word", style="bold")
        word_table.add_column("Lemma")
        word_table.add_column("POS")
        word_table.add_column("CEFR")
        word_table.add_column("Zipf", justify="right")
        word_table.add_column("#", justify="right")
        word_table.add_column("Context", max_width=40, no_wrap=True)
        word_table.add_column("Translation", max_width=40, no_wrap=True)

        for w in rare:
            word_table.add_row(
                w["word"],
                w["lemma"],
                w["pos"],
                w["cefr"],
                str(w["zipf"]),
                str(w["count"]),
                w.get("context", "")[:40],
                w.get("translation", "")[:40],
            )
        console.print(word_table)
