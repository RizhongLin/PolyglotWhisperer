"""pgw play command — play video with dual subtitles."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from pgw.core.config import load_config

console = Console()


def play(
    video: Annotated[
        Path,
        typer.Argument(help="Path to video file."),
    ],
    subs: Annotated[
        Optional[Path],
        typer.Option("--subs", "-s", help="Primary subtitle file (original language)."),
    ] = None,
    translation: Annotated[
        Optional[Path],
        typer.Option("--translation", "-t", help="Translation subtitle file."),
    ] = None,
) -> None:
    """Play a video with optional dual subtitles using mpv."""
    from pgw.player.mpv_player import play as mpv_play

    config = load_config()

    if not video.is_file():
        console.print(f"[red]File not found:[/red] {video}")
        raise typer.Exit(1)

    mpv_play(
        video_path=video,
        primary_subs=subs,
        secondary_subs=translation,
        config=config.player,
    )
