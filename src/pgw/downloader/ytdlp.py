"""yt-dlp wrapper for downloading videos from URLs."""

from __future__ import annotations

import json
from pathlib import Path

from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeRemainingColumn

from pgw.core.models import VideoSource
from pgw.utils.cache import file_hash
from pgw.utils.console import console

_DEFAULT_FORMAT = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
_MANIFEST_NAME = ".downloads.jsonl"
_SUBTITLE_EXTS = (".vtt", ".srt", ".ass", ".ssa", ".ttml")

# Language code aliases: our ISO 639-1 codes → alternatives used by YouTube/yt-dlp
_LANG_ALIASES: dict[str, list[str]] = {
    "he": ["iw"],
    "iw": ["he"],
    "jw": ["jv"],
    "jv": ["jw"],
    "no": ["nb"],
    "nb": ["no"],
}


def _make_progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    )


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


def _find_subtitle_file(video_path: Path, language: str) -> tuple[Path | None, bool]:
    """Find a previously downloaded subtitle file alongside a video.

    Checks for files matching {video_stem}.{lang}.{ext} patterns,
    including regional variants (e.g. zh-Hans for zh).

    Returns:
        (subtitle_path, is_auto) — is_auto defaults to False for file-based
        detection since we can't determine origin from the filename.
    """
    stem = video_path.stem
    parent = video_path.parent

    candidates = [language] + _LANG_ALIASES.get(language, [])

    for lang in candidates:
        for ext in _SUBTITLE_EXTS:
            # Exact match: Title_ID.en.vtt
            path = parent / f"{stem}.{lang}{ext}"
            if path.is_file():
                return path, False
            # Regional variants: Title_ID.zh-Hans.vtt, Title_ID.pt-BR.srt
            for p in parent.glob(f"{stem}.{lang}-*{ext}"):
                if p.is_file():
                    return p, False
            for p in parent.glob(f"{stem}.{lang}_*{ext}"):
                if p.is_file():
                    return p, False

    return None, False


def _extract_subtitle_info(info: dict, video_path: Path, language: str) -> tuple[Path | None, bool]:
    """Extract downloaded subtitle path and type from yt-dlp info dict.

    Inspects ``info["requested_subtitles"]`` for downloaded files.
    Prefers exact language match and human-made over auto-generated.

    Returns:
        (subtitle_path, is_auto) or (None, False) if nothing found.
    """
    requested = info.get("requested_subtitles") or {}
    human_subs = info.get("subtitles") or {}

    if not requested:
        return None, False

    best_path: Path | None = None
    best_is_auto = True
    best_is_exact = False

    for key, sub_info in requested.items():
        filepath = sub_info.get("filepath")
        if not filepath:
            continue
        path = Path(filepath)
        if not path.is_file():
            continue

        is_auto = key not in human_subs
        is_exact = key == language

        if (
            best_path is None
            or (is_exact and not best_is_exact)
            or (not best_is_exact and not is_auto and best_is_auto)
        ):
            best_path = path
            best_is_auto = is_auto
            best_is_exact = is_exact

    return best_path, best_is_auto


def _find_cached(url: str, output_dir: Path, language: str | None = None) -> VideoSource | None:
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
        if expected_hash and file_hash(cached_path) != expected_hash:
            console.print(f"[yellow]Cache stale (hash mismatch):[/yellow] {cached_path}")
            continue
        console.print(f"[dim]Found cached download:[/dim] {cached_path}")
        # Check for previously downloaded subtitle files
        subtitle_path = None
        subtitle_is_auto = False
        if language:
            subtitle_path, subtitle_is_auto = _find_subtitle_file(cached_path, language)
            if subtitle_path:
                console.print(f"[dim]Found cached subtitles:[/dim] {subtitle_path.name}")
        return VideoSource(
            video_path=cached_path,
            source_url=url,
            title=entry.get("title", cached_path.stem),
            duration=entry.get("duration"),
            content_hash=expected_hash,
            subtitle_path=subtitle_path,
            subtitle_is_auto=subtitle_is_auto,
        )
    return None


def download(
    url: str,
    output_dir: Path | None = None,
    fmt: str = _DEFAULT_FORMAT,
    language: str | None = None,
) -> VideoSource:
    """Download a video from URL using yt-dlp.

    Checks a local manifest first to avoid re-downloading the same URL.
    Uses single-pass extract_info(download=True) and yt-dlp's
    prepare_filename for exact output path resolution.

    When *language* is provided, also downloads existing subtitles from
    the video page (human-made preferred, auto-generated as fallback).

    Args:
        url: Video URL.
        output_dir: Directory to save the file. Defaults to ./downloads.
        fmt: yt-dlp format string.
        language: Source language code for subtitle download (e.g. "en", "fr").

    Returns:
        VideoSource with local video path, metadata, and optional subtitle path.
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
    cached = _find_cached(url, output_dir, language=language)
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

    # Add subtitle download options (regex patterns catch regional variants)
    if language:
        sub_langs = [language]
        sub_langs.extend(_LANG_ALIASES.get(language, []))
        sub_langs.append(f"{language}-.*")
        sub_langs.append(f"{language}_.*")
        opts.update(
            {
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": sub_langs,
                "subtitlesformat": "vtt/srt/ass",
            }
        )

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
    content_sha = file_hash(video_path)
    from datetime import datetime, timezone

    _append_manifest(
        output_dir,
        {
            "url": url,
            "title": title,
            "path": str(video_path),
            "sha256": content_sha,
            "size_bytes": video_path.stat().st_size,
            "duration": duration,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    # Check for downloaded subtitles
    subtitle_path = None
    subtitle_is_auto = False
    if language:
        subtitle_path, subtitle_is_auto = _extract_subtitle_info(info, video_path, language)
        # Fallback: glob for subtitle files (some extractors don't set filepath)
        if subtitle_path is None:
            subtitle_path, subtitle_is_auto = _find_subtitle_file(video_path, language)
        if subtitle_path:
            kind = "auto-generated" if subtitle_is_auto else "human-made"
            console.print(f"[green]Found {kind} subtitles ({language})[/green]")

    return VideoSource(
        video_path=video_path,
        source_url=url,
        title=title,
        duration=duration,
        content_hash=content_sha,
        subtitle_path=subtitle_path,
        subtitle_is_auto=subtitle_is_auto,
    )
