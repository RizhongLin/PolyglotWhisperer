"""Shared Rich console instance and progress/output utilities.

All output functions route through the pgw logging system so messages
appear in the terminal AND in any configured log file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

_LOGGER = logging.getLogger("pgw")


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
        _LOGGER.info("[bold]%s[/bold]  [dim]%s[/dim]", name, detail)
    else:
        _LOGGER.info("[bold]%s[/bold]", name)


def cache_hit(message: str = "Using cache") -> None:
    """Print an indented dim cache-hit message."""
    _LOGGER.info("  [dim]%s[/dim]", message)


def error(message: str) -> None:
    """Print an error message in red."""
    _LOGGER.error("[red]%s[/red]", message)


def warning(message: str) -> None:
    """Print a warning message in yellow."""
    _LOGGER.warning("[yellow]%s[/yellow]", message)


def saved(*paths: Path | str) -> None:
    """Print green 'Saved' with file path(s)."""
    names = "  ".join(str(p) for p in paths)
    _LOGGER.info("[green]Saved:[/green] %s", names)


def debug(message: str) -> None:
    """Print only when PGW_DEBUG=1."""
    if os.environ.get("PGW_DEBUG"):
        _LOGGER.debug("[dim]%s[/dim]", message)


def workspace_done(workspace: Path, files: list[Path]) -> None:
    """Print the final Done + Workspace + file listing block."""
    names = [f.name for f in files if f.is_file()]
    _LOGGER.info("\n[bold green]Done![/bold green] Workspace: %s", workspace)
    if names:
        lines: list[list[str]] = [[]]
        line_len = 0
        for n in names:
            if line_len > 0 and line_len + len(n) + 2 > 76:
                lines.append([])
                line_len = 0
            lines[-1].append(n)
            line_len += len(n) + 2
        for line in lines:
            _LOGGER.info("  %s", ("  ").join(line))
