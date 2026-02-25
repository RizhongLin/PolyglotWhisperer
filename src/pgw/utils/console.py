"""Shared Rich console instance and progress utilities."""

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
