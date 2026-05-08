"""Audio clip extraction for flashcards.

Cuts a small ``[start_ms, end_ms)`` slice out of the workspace's
``audio.<ext>`` and caches it under ``<workspace>/.cache/clips/`` keyed
by ``(content_hash, start_ms, end_ms)``. The clip is encoded as MP3
(64 kbps mono) for fast network delivery and consistent ETags.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Cap how much audio a single clip can carry. 30s is plenty for a
# review prompt; anything bigger probably means the caller passed
# milliseconds-vs-seconds wrong and we don't want to spend ffmpeg on it.
_MAX_CLIP_MS = 30_000
_AUDIO_EXTS = (".mp3", ".m4a", ".wav", ".ogg", ".aac", ".opus", ".flac")


def find_workspace_audio(workspace: Path) -> Path | None:
    """Locate the canonical ``audio.<ext>`` file in a workspace."""
    for ext in _AUDIO_EXTS:
        candidate = workspace / f"audio{ext}"
        if candidate.is_file():
            return candidate
    # Fallback: any audio-shaped file at the workspace root.
    for entry in sorted(workspace.iterdir()):
        if entry.is_file() and entry.suffix.lower() in _AUDIO_EXTS:
            return entry
    return None


def cut_clip(
    workspace: Path,
    *,
    start_ms: int,
    end_ms: int,
) -> Path:
    """Return the path to a cached MP3 clip; create it if missing.

    Raises ``ValueError`` for an invalid range, ``FileNotFoundError``
    when the workspace has no audio, ``RuntimeError`` if ffmpeg fails.
    """
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError(f"invalid clip range [{start_ms}, {end_ms})")
    duration_ms = end_ms - start_ms
    if duration_ms > _MAX_CLIP_MS:
        raise ValueError(f"clip duration {duration_ms}ms exceeds cap {_MAX_CLIP_MS}ms")

    source = find_workspace_audio(workspace)
    if source is None:
        raise FileNotFoundError(f"no audio file under {workspace}")

    cache_dir = workspace / ".cache" / "clips"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = _cache_key(source, start_ms, end_ms)
    out_path = cache_dir / f"{cache_key}.mp3"
    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not on PATH")

    # Encode to a process-unique temp path then atomically rename — two
    # concurrent requests for the same clip will race on os.replace, but
    # both end up pointing at a complete, byte-identical file. Without
    # this the second writer's ``-y`` overwrite tears the first reader's
    # stream mid-flight.
    tmp_path = cache_dir / f"{cache_key}.{os.getpid()}.tmp.mp3"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start_ms / 1000:.3f}",
        "-t",
        f"{duration_ms / 1000:.3f}",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "44100",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "64k",
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    except subprocess.CalledProcessError as exc:
        # Surface stderr for easier debugging — ffmpeg is verbose but
        # `loglevel error` keeps it short.
        tmp_path.unlink(missing_ok=True)
        msg = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else "unknown"
        raise RuntimeError(f"ffmpeg failed: {msg}") from exc
    except subprocess.TimeoutExpired as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg timed out cutting clip") from exc
    os.replace(tmp_path, out_path)
    return out_path


def _cache_key(source: Path, start_ms: int, end_ms: int) -> str:
    """Stable per-source-content cache key.

    We use ``size + mtime`` rather than a full sha256 of the source to
    keep first-clip latency low; collisions only happen if the file is
    swapped out without a stat change which is fine for our use case.
    """
    try:
        st = source.stat()
        sig = f"{st.st_size}-{int(st.st_mtime)}"
    except OSError:
        sig = "unknown"
    raw = f"{source.name}|{sig}|{start_ms}|{end_ms}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]
