"""pgw clean command — show cache usage and clear cached files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pgw.core.config import load_config
from pgw.utils.console import console
from pgw.utils.text import BYTES_PER_GB, BYTES_PER_KB, BYTES_PER_MB

_CACHE_CATEGORIES = ["audio", "compressed", "downloads", "transcriptions"]


def _dir_size(path: Path) -> tuple[int, int]:
    """Return (total_bytes, file_count) for a directory."""
    if not path.is_dir():
        return 0, 0
    total = 0
    count = 0
    for f in path.rglob("*"):
        if f.is_file() and not f.is_symlink():
            total += f.stat().st_size
            count += 1
        elif f.is_symlink():
            count += 1
    return total, count


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if size_bytes < BYTES_PER_KB:
        return f"{size_bytes} B"
    if size_bytes < BYTES_PER_MB:
        return f"{size_bytes / BYTES_PER_KB:.0f} KB"
    if size_bytes < BYTES_PER_GB:
        return f"{size_bytes / BYTES_PER_MB:.1f} MB"
    return f"{size_bytes / BYTES_PER_GB:.2f} GB"


def clean(
    category: Annotated[
        list[str] | None,
        typer.Argument(help="Categories to clear: audio, compressed, downloads, transcriptions."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Show what would be deleted without deleting."),
    ] = False,
    workspaces: Annotated[
        bool,
        typer.Option("--workspaces", "-w", help="Also remove workspace output directories."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
) -> None:
    """Show cache usage and clear cached files."""
    import shutil

    config = load_config()
    workspace_dir = Path(config.workspace_dir).resolve()
    cache_dir = workspace_dir / ".cache"

    # Show current usage
    console.print(f"[bold]Workspace:[/bold] {workspace_dir}")
    console.print()

    total_size = 0
    total_files = 0
    rows: list[tuple[str, int, int]] = []

    for cat in _CACHE_CATEGORIES:
        cat_dir = cache_dir / cat
        size, count = _dir_size(cat_dir)
        rows.append((cat, size, count))
        total_size += size
        total_files += count

    # Workspace directories (non-cache)
    ws_size = 0
    ws_count = 0
    if workspace_dir.is_dir():
        for d in workspace_dir.iterdir():
            if d.name.startswith(".") or not d.is_dir():
                continue
            size, count = _dir_size(d)
            ws_size += size
            ws_count += count

    console.print("[bold]Cache usage:[/bold]")
    for cat, size, count in rows:
        if count > 0:
            console.print(f"  {cat:<16} {_format_size(size):>10}  ({count} files)")
        else:
            console.print(f"  {cat:<16}          —")
    console.print(f"  {'':─<16} {'':─>10}")
    console.print(f"  {'total':<16} {_format_size(total_size):>10}  ({total_files} files)")

    if workspaces:
        ws_info = f"{_format_size(ws_size):>10}  ({ws_count} files)"
        console.print(f"\n[bold]Workspaces:[/bold]     {ws_info}")

    # Determine what to clear
    if category is not None:
        targets = []
        for c in category:
            if c not in _CACHE_CATEGORIES:
                console.print(f"[red]Unknown category:[/red] {c}")
                console.print(f"[dim]Valid: {', '.join(_CACHE_CATEGORIES)}[/dim]")
                raise typer.Exit(1)
            targets.append(c)
    else:
        targets = list(_CACHE_CATEGORIES)

    # Collect dirs to remove
    dirs_to_remove: list[Path] = []
    for t in targets:
        d = cache_dir / t
        if d.is_dir():
            dirs_to_remove.append(d)

    if workspaces and workspace_dir.is_dir():
        for d in workspace_dir.iterdir():
            if d.name.startswith(".") or not d.is_dir():
                continue
            dirs_to_remove.append(d)

    if not dirs_to_remove:
        console.print("\n[dim]Nothing to clean.[/dim]")
        return

    # Calculate what will be removed
    remove_size = sum(_dir_size(d)[0] for d in dirs_to_remove)
    label = ", ".join(targets)
    if workspaces:
        label += " + workspaces"

    if dry_run:
        rm_info = f"{_format_size(remove_size)} ({label})"
        console.print(f"\n[yellow]Dry run:[/yellow] would remove {rm_info}")
        for d in dirs_to_remove:
            console.print(f"  [dim]{d}[/dim]")
        return

    if not yes:
        console.print(f"\n[bold]Will remove:[/bold] {_format_size(remove_size)} ({label})")
        confirm = typer.confirm("Continue?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            return

    for d in dirs_to_remove:
        shutil.rmtree(d)

    console.print(f"\n[green]Cleared {_format_size(remove_size)}[/green] ({label})")
