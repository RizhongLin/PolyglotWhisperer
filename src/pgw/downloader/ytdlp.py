"""yt-dlp wrapper for downloading videos from URLs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeRemainingColumn

from pgw.core.models import VideoSource
from pgw.utils.console import console

_DEFAULT_FORMAT = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
_MANIFEST_NAME = ".downloads.jsonl"


def _make_progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    )


def _sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(output_dir: Path) -> list[dict]:
    """Load the download manifest (JSONL), skipping corrupt entries."""
    manifest_path = output_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        return []
    entries = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _append_manifest(output_dir: Path, entry: dict) -> None:
    """Append an entry to the download manifest."""
    manifest_path = output_dir / _MANIFEST_NAME
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _find_cached(url: str, output_dir: Path) -> VideoSource | None:
    """Check if a URL has already been downloaded and the file is intact."""
    for entry in _load_manifest(output_dir):
        if entry.get("url") != url:
            continue
        path_str = entry.get("path")
        if not path_str:
            continue
        cached_path = Path(path_str)
        if not cached_path.is_file():
            continue
        # Quick size check before expensive hash verification
        expected_size = entry.get("size_bytes")
        if expected_size is not None and cached_path.stat().st_size != expected_size:
            console.print(f"[yellow]Cache stale (size mismatch):[/yellow] {cached_path}")
            continue
        # Full hash verification only if size matches
        expected_hash = entry.get("sha256")
        if expected_hash and _sha256(cached_path) != expected_hash:
            console.print(f"[yellow]Cache stale (hash mismatch):[/yellow] {cached_path}")
            continue
        console.print(f"[dim]Found cached download:[/dim] {cached_path}")
        return VideoSource(
            video_path=cached_path,
            source_url=url,
            title=entry.get("title", cached_path.stem),
            duration=entry.get("duration"),
        )
    return None


def download(
    url: str,
    output_dir: Path | None = None,
    fmt: str = _DEFAULT_FORMAT,
) -> VideoSource:
    """Download a video from URL using yt-dlp.

    Checks a local manifest first to avoid re-downloading the same URL.
    Uses single-pass extract_info(download=True) and yt-dlp's
    prepare_filename for exact output path resolution.

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
        output_dir = Path("./pgw_workspace/.cache/downloads")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check cache first
    cached = _find_cached(url, output_dir)
    if cached is not None:
        return cached

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
        "outtmpl": str(output_dir / "%(title)s_%(id)s.%(ext)s"),
        "progress_hooks": [_progress_hook],
        "quiet": True,
        "no_warnings": True,
    }

    console.print(f"[bold]Downloading:[/bold] {url}")

    with yt_dlp.YoutubeDL(opts) as ydl:
        with progress:
            info = ydl.extract_info(url, download=True)

        title = info.get("title", "video")
        duration = info.get("duration")

        # Use yt-dlp's own path resolution for exact output filename
        video_path = Path(ydl.prepare_filename(info))

    # Verify the file exists (handle edge cases like format merging)
    if not video_path.is_file():
        # Fallback: most recent non-hidden file in output_dir
        candidates = sorted(
            (p for p in output_dir.iterdir() if not p.name.startswith(".")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(f"Download completed but no file found in {output_dir}")
        video_path = candidates[0]

    console.print(f"[bold]Title:[/bold] {title}")
    console.print(f"[green]Downloaded:[/green] {video_path}")

    # Compute hash and save to manifest
    console.print("[dim]Computing file hash...[/dim]")
    file_hash = _sha256(video_path)
    from datetime import datetime, timezone

    _append_manifest(
        output_dir,
        {
            "url": url,
            "title": title,
            "path": str(video_path),
            "sha256": file_hash,
            "size_bytes": video_path.stat().st_size,
            "duration": duration,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    return VideoSource(
        video_path=video_path,
        source_url=url,
        title=title,
        duration=duration,
    )
