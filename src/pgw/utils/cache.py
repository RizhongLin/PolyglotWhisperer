"""Shared media cache for PolyglotWhisperer.

Content-addressable cache at <workspace_dir>/.cache/ for extracted or
processed media files. Cache hits are symlinked into workspace directories
to avoid redundant work across runs.

When a content hash (SHA-256) is available, cache keys are derived from
the file content — so re-downloading the same video produces the same
cache key. Falls back to metadata-based keys (path + size + mtime) when
no content hash is available.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


def file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file contents.

    Reads in 1 MB chunks to handle large files efficiently.

    Returns:
        Full 64-char hex SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(
    file_path: Path | None = None,
    *,
    content_hash: str | None = None,
    **params: object,
) -> str:
    """Compute a cache key from content hash or file metadata.

    When *content_hash* is provided the key is content-addressable:
    same content produces the same key regardless of path or mtime.
    Falls back to metadata-based keying (path + size + mtime) when
    no hash is available.

    Args:
        file_path: Path for metadata-based key (legacy fallback).
        content_hash: SHA-256 content hash for content-addressable key.
        **params: Extra parameters (sample_rate, model, backend, …).

    Returns:
        16-char hex string.
    """
    if content_hash:
        base = content_hash
    elif file_path is not None:
        resolved = file_path.resolve()
        stat = resolved.stat()
        base = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
    else:
        raise ValueError("Either file_path or content_hash must be provided")

    for k, v in sorted(params.items()):
        base += f"|{k}={v}"
    return hashlib.sha256(base.encode()).hexdigest()[:16]


def find_cached_file(
    cache_dir: Path,
    suffix: str,
    *,
    content_hash: str | None = None,
    file_path: Path | None = None,
    **params: object,
) -> Path | None:
    """Look up a cached file, trying content-based key first, then metadata fallback.

    Args:
        cache_dir: Directory containing cached files.
        suffix: File extension including dot (e.g. ".wav", ".json").
        content_hash: Content hash for content-addressable lookup.
        file_path: File path for metadata-based fallback lookup.
        **params: Additional cache key parameters.

    Returns:
        Path to cached file if found, None otherwise.
    """
    if content_hash:
        key = cache_key(content_hash=content_hash, **params)
        cached = cache_dir / f"{key}{suffix}"
        if cached.is_file():
            return cached
        if cached.is_symlink():
            cached.unlink(missing_ok=True)  # Clean up broken symlink

    if file_path is not None:
        try:
            meta_key = cache_key(file_path=file_path, **params)
            cached = cache_dir / f"{meta_key}{suffix}"
            if cached.is_file():
                return cached
            if cached.is_symlink():
                cached.unlink(missing_ok=True)  # Clean up broken symlink
        except OSError:
            pass  # File may not exist for metadata stat

    return None


def get_cache_dir(workspace_dir: Path, category: str) -> Path:
    """Get or create a cache subdirectory (e.g. "audio")."""
    cache_dir = Path(workspace_dir) / ".cache" / category
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text atomically: write to temp file, then rename.

    Prevents corrupted cache files if the process crashes mid-write.
    """
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def link_or_copy(source: Path, dest: Path) -> None:
    """Symlink source to dest, with copy fallback.

    Creates parent directories as needed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source.resolve(), dest)
    except OSError:
        shutil.copy2(source, dest)
