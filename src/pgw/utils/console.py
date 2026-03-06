"""Shared Rich console instance and progress/output utilities."""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def chunk_progress() -> Progress:
    """Create a Rich Progress bar for chunk-based LLM processing."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TextColumn("{task.completed}/{task.total} chunks"),
        console=console,
    )


def stage(name: str, detail: str = "") -> None:
    """Print a pipeline stage header: bold name, dim detail."""
    if detail:
        console.print(f"[bold]{name}[/bold]  [dim]{detail}[/dim]")
    else:
        console.print(f"[bold]{name}[/bold]")


def cache_hit(message: str = "Using cache") -> None:
    """Print an indented dim cache-hit message."""
    console.print(f"  [dim]{message}[/dim]")


def error(message: str) -> None:
    """Print an error message in red."""
    console.print(f"[red]{message}[/red]")


def warning(message: str) -> None:
    """Print a warning message in yellow."""
    console.print(f"[yellow]{message}[/yellow]")


def saved(*paths: Path | str) -> None:
    """Print green 'Saved' with file path(s)."""
    names = "  ".join(str(p) for p in paths)
    console.print(f"[green]Saved:[/green] {names}")


def debug(message: str) -> None:
    """Print only when PGW_DEBUG=1."""
    if os.environ.get("PGW_DEBUG"):
        console.print(f"[dim]{message}[/dim]")


def workspace_done(workspace: Path, files: list[Path]) -> None:
    """Print the final Done + Workspace + file listing block."""
    names = [f.name for f in files if f.is_file()]
    console.print(f"\n[bold green]Done![/bold green] Workspace: {workspace}")
    if names:
        # Wrap file names at ~80 columns
        lines: list[list[str]] = [[]]
        line_len = 0
        for n in names:
            if line_len > 0 and line_len + len(n) + 2 > 76:
                lines.append([])
                line_len = 0
            lines[-1].append(n)
            line_len += len(n) + 2
        for line in lines:
            console.print(f"  {('  ').join(line)}")
