"""Shared media cache for PolyglotWhisperer.

Hash-based cache at <workspace_dir>/.cache/ for extracted or processed
media files. Cache hits are symlinked into workspace directories to
avoid redundant work across runs.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


def cache_key(file_path: Path, **params: object) -> str:
    """Compute a cache key from file metadata and parameters.

    Uses resolved path, size, and mtime — no file content reading.
    Extra keyword arguments (sample_rate, start, duration, etc.) are
    included in the hash.

    Returns:
        16-char hex string.
    """
    resolved = file_path.resolve()
    stat = resolved.stat()
    parts = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
    for k, v in sorted(params.items()):
        parts += f"|{k}={v}"
    return hashlib.sha256(parts.encode()).hexdigest()[:16]


def get_cache_dir(workspace_dir: Path, category: str) -> Path:
    """Get or create a cache subdirectory (e.g. "audio")."""
    cache_dir = Path(workspace_dir) / ".cache" / category
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def link_or_copy(source: Path, dest: Path) -> None:
    """Symlink source to dest, with copy fallback.

    Creates parent directories as needed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source.resolve(), dest)
    except OSError:
        shutil.copy2(source, dest)
