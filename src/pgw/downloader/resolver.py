"""Resolve input to a local video source — URL or local file."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from pgw.core.models import VideoSource


def is_url(input_path: str) -> bool:
    """Check if the input looks like a URL."""
    parsed = urlparse(input_path)
    return parsed.scheme in ("http", "https")


def resolve(input_path: str, output_dir: Path | None = None) -> VideoSource:
    """Resolve input to a VideoSource.

    If input is a local file, return it directly.
    If input is a URL, download via yt-dlp.

    Args:
        input_path: URL or local file path.
        output_dir: Directory for downloaded files.

    Returns:
        VideoSource with local video path.
    """
    if is_url(input_path):
        from pgw.downloader.ytdlp import download

        return download(input_path, output_dir=output_dir)

    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {input_path}")

    return VideoSource(
        video_path=path,
        title=path.stem,
    )
