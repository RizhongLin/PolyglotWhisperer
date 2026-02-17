"""Video playback with dual subtitle overlay using mpv."""

from __future__ import annotations

import shutil
from pathlib import Path

from rich.console import Console

from pgw.core.config import PlayerConfig

console = Console()


def check_mpv() -> bool:
    """Check if mpv is available on the system."""
    return shutil.which("mpv") is not None


def play(
    video_path: Path,
    primary_subs: Path | None = None,
    secondary_subs: Path | None = None,
    config: PlayerConfig | None = None,
) -> None:
    """Play a video with optional dual subtitles.

    Primary subs (original language) shown at bottom.
    Secondary subs (translation) shown at top.

    Args:
        video_path: Path to the video file.
        primary_subs: Path to primary subtitle file (original language).
        secondary_subs: Path to secondary subtitle file (translation).
        config: Player configuration.
    """
    if not check_mpv():
        raise FileNotFoundError("mpv not found. Install it with: brew install mpv")

    try:
        import mpv
    except ImportError:
        raise ImportError("python-mpv is not installed. Install with: uv sync --extra player")

    if config is None:
        config = PlayerConfig()

    player = mpv.MPV(
        input_default_bindings=True,
        input_vo_keyboard=True,
        osc=True,
    )

    player["sub-font-size"] = config.sub_font_size

    if primary_subs and primary_subs.is_file():
        player["sub-file"] = str(primary_subs)
    if secondary_subs and secondary_subs.is_file():
        player["secondary-sub-file"] = str(secondary_subs)
        player["secondary-sub-visibility"] = True

    console.print(f"[bold]Playing:[/bold] {video_path.name}")
    if primary_subs:
        console.print(f"[bold]Subtitles:[/bold] {primary_subs.name}")
    if secondary_subs:
        console.print(f"[bold]Translation:[/bold] {secondary_subs.name}")

    player.play(str(video_path))

    try:
        player.wait_for_playback()
    except mpv.ShutdownError:
        pass
    finally:
        player.terminate()
