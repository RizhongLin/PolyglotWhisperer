"""Video playback with dual subtitle overlay using mpv."""

from __future__ import annotations

import shutil
import subprocess
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
    bilingual_subs: Path | None = None,
    config: PlayerConfig | None = None,
) -> None:
    """Play a video with subtitles via mpv subprocess.

    If bilingual_subs is provided, uses it as a single track with
    built-in positioning (original at bottom, translation at top).
    Otherwise falls back to primary_subs only.

    Args:
        video_path: Path to the video file.
        primary_subs: Path to primary subtitle file (original language).
        secondary_subs: Unused, kept for API compatibility.
        bilingual_subs: Path to bilingual VTT with positioning cues.
        config: Player configuration.
    """
    if not check_mpv():
        raise FileNotFoundError("mpv not found. Install it with: brew install mpv")

    if config is None:
        config = PlayerConfig()

    cmd = ["mpv", str(video_path)]
    cmd.append(f"--sub-font-size={config.sub_font_size}")

    # Prefer bilingual VTT (has positioning baked in)
    if bilingual_subs and bilingual_subs.is_file():
        cmd.append(f"--sub-file={bilingual_subs}")
        console.print(f"[bold]Playing:[/bold] {video_path.name}")
        console.print(f"[bold]Subtitles:[/bold] {bilingual_subs.name} (bilingual)")
    elif primary_subs and primary_subs.is_file():
        cmd.append(f"--sub-file={primary_subs}")
        console.print(f"[bold]Playing:[/bold] {video_path.name}")
        console.print(f"[bold]Subtitles:[/bold] {primary_subs.name}")
    else:
        console.print(f"[bold]Playing:[/bold] {video_path.name}")

    subprocess.run(cmd)
