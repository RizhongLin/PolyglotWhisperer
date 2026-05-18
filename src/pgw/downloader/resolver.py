"""Resolve input to a local video source — URL or local file."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from pgw.core.models import VideoSource
from pgw.utils.cache import file_hash
from pgw.utils.console import debug
from pgw.utils.text import BYTES_PER_MB

# Show "Indexing file..." debug message for files larger than this
_DEBUG_SIZE_THRESHOLD = 100 * BYTES_PER_MB


def is_url(input_path: str) -> bool:
    """Check if the input looks like a URL."""
    parsed = urlparse(input_path)
    return parsed.scheme in ("http", "https")


def resolve(
    input_path: str,
    output_dir: Path | None = None,
    fmt: str | None = None,
    language: str | None = None,
) -> VideoSource:
    """Resolve input to a VideoSource.

    If input is a URL, first attempts stream resolution (no download)
    via yt-dlp ``extract_info(download=False)``.  On success the
    returned ``VideoSource`` carries ``video_url`` / ``audio_url`` and
    the pipeline will skip video download entirely.  Falls back to a
    full download when stream resolution is unavailable.

    If input is a local file, returns it directly.

    Args:
        input_path: URL or local file path.
        output_dir: Directory for downloaded/cached files.
        fmt: yt-dlp format string override.
        language: Source language code for subtitle download.

    Returns:
        VideoSource with local video path and (when resolved) stream URLs.
    """
    if is_url(input_path):
        from pgw.downloader.ytdlp import download, resolve_stream

        # Try stream resolution first — no download
        streamed = resolve_stream(input_path, output_dir=output_dir, language=language)
        if streamed is not None and streamed.video_url:
            return streamed

        # Fall back to full download
        kwargs: dict = {"output_dir": output_dir}
        if fmt:
            kwargs["fmt"] = fmt
        if language:
            kwargs["language"] = language
        return download(input_path, **kwargs)

    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {input_path}")

    if path.stat().st_size > _DEBUG_SIZE_THRESHOLD:
        debug("Indexing file...")
    content_hash = file_hash(path)

    return VideoSource(
        video_path=path,
        title=path.stem,
        content_hash=content_hash,
    )
