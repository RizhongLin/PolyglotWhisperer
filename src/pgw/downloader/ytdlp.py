"""yt-dlp wrapper for downloading videos from URLs."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeRemainingColumn

from pgw.core.models import VideoSource

console = Console()

_DEFAULT_FORMAT = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"


def _make_progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    )


def download(
    url: str,
    output_dir: Path | None = None,
    fmt: str = _DEFAULT_FORMAT,
) -> VideoSource:
    """Download a video from URL using yt-dlp.

    Args:
        url: Video URL.
        output_dir: Directory to save the file. Defaults to ./downloads.
        fmt: yt-dlp format string.

    Returns:
        VideoSource with local video path and metadata.
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError("yt-dlp is not installed. Install with: uv sync --extra download")

    if output_dir is None:
        output_dir = Path("./downloads")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = _make_progress()
    task_id: TaskID | None = None

    def _progress_hook(d: dict) -> None:
        nonlocal task_id
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if task_id is None and total > 0:
                task_id = progress.add_task("Downloading", total=total)
            if task_id is not None:
                progress.update(task_id, completed=downloaded)
        elif d["status"] == "finished":
            if task_id is not None:
                progress.update(task_id, completed=progress.tasks[task_id].total)

    opts = {
        "format": fmt,
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "progress_hooks": [_progress_hook],
        "quiet": True,
        "no_warnings": True,
    }

    console.print(f"[bold]Fetching info:[/bold] {url}")

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get("title", "video")
        duration = info.get("duration")

        console.print(f"[bold]Title:[/bold] {title}")
        with progress:
            ydl.download([url])

    # Find the downloaded file
    glob_pattern = f"{_sanitize_title(title)}.*"
    downloaded_files = sorted(
        output_dir.glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not downloaded_files:
        # Fallback: find most recent file in output_dir
        downloaded_files = sorted(
            output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        )

    if not downloaded_files:
        raise RuntimeError(f"Download completed but no file found in {output_dir}")

    video_path = downloaded_files[0]
    console.print(f"[green]Downloaded:[/green] {video_path}")

    return VideoSource(
        video_path=video_path,
        source_url=url,
        title=title,
        duration=duration,
    )


def extract_info(url: str) -> dict:
    """Get video metadata without downloading."""
    try:
        import yt_dlp
    except ImportError:
        raise ImportError("yt-dlp is not installed. Install with: uv sync --extra download")

    opts = {"quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


def _sanitize_title(title: str) -> str:
    """Rough sanitization to match yt-dlp's filename output."""
    # yt-dlp replaces some chars; this is a best-effort match for glob
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .-_()")
    return "".join(c if c in keep else "_" for c in title)
