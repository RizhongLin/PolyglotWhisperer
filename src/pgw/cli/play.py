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
        typer.Argument(help="Path to video file or workspace directory."),
    ],
    subs: Annotated[
        Optional[Path],
        typer.Option("--subs", "-s", help="Primary subtitle file (original language)."),
    ] = None,
    bilingual: Annotated[
        Optional[Path],
        typer.Option("--bilingual", "-b", help="Bilingual VTT file (both languages)."),
    ] = None,
) -> None:
    """Play a video with subtitles using mpv.

    Pass a workspace directory to auto-detect video and subtitle files,
    or pass a video file with explicit --subs / --bilingual options.
    """
    from pgw.player.mpv_player import play as mpv_play

    config = load_config()

    # If a directory is passed, auto-detect workspace files
    if video.is_dir():
        workspace = video
        video_path = workspace / "video.mp4"
        if not video_path.is_file():
            console.print(f"[red]No video.mp4 found in workspace:[/red] {workspace}")
            raise typer.Exit(1)

        # Auto-detect bilingual VTT
        if bilingual is None:
            candidates = sorted(workspace.glob("bilingual.*.vtt"))
            if candidates:
                bilingual = candidates[0]

        # Auto-detect primary subs
        if subs is None:
            candidates = sorted(workspace.glob("transcription.*.vtt"))
            if candidates:
                subs = candidates[0]
    else:
        video_path = video
        if not video_path.is_file():
            console.print(f"[red]File not found:[/red] {video}")
            raise typer.Exit(1)

    mpv_play(
        video_path=video_path,
        primary_subs=subs,
        bilingual_subs=bilingual,
        config=config.player,
    )
