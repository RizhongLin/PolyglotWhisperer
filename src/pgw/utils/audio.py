"""Audio extraction from video files and stream URLs using ffmpeg."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

from pgw.utils.cache import cache_key, find_cached_file, get_cache_dir, link_or_copy


def _is_url(source: str | Path) -> bool:
    """True when *source* is a network URL ffmpeg should read directly."""
    s = str(source)
    return s.startswith(("http://", "https://", "rtmp://", "rtmps://"))


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def extract_audio(
    video_path: Path | str,
    output_path: Path | None = None,
    sample_rate: int = 16000,
    start: str | None = None,
    duration: str | None = None,
) -> Path:
    """Extract audio to 16 kHz mono PCM WAV.

    Accepts a local file path or a network URL (https / rtmp).
    When the input is a URL, ffmpeg reads the stream directly — no
    file needs to exist on disk.

    Args:
        video_path: Local file path or network URL.
        output_path: Path for the output WAV file.
        sample_rate: Audio sample rate in Hz. Whisper expects 16000.
        start: Start time for clipping (ffmpeg format).
        duration: Duration to extract (ffmpeg format).

    Returns:
        Path to the extracted audio file.

    Raises:
        FileNotFoundError: If ffmpeg is not installed, or if a local
            file path does not exist.
        subprocess.CalledProcessError: If ffmpeg fails.
    """
    if not check_ffmpeg():
        raise FileNotFoundError("ffmpeg not found. Install it with: brew install ffmpeg")

    source_str = str(video_path)
    is_remote = _is_url(source_str)

    if not is_remote:
        local = Path(video_path)
        if not local.is_file():
            raise FileNotFoundError(f"Video file not found: {local}")

    if output_path is None:
        if is_remote:
            output_path = Path("audio.wav")
        else:
            output_path = Path(video_path).with_suffix(".wav")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg"]

    if start is not None:
        cmd.extend(["-ss", str(start)])

    cmd.extend(["-i", source_str])

    if duration is not None:
        cmd.extend(["-t", str(duration)])

    cmd.extend(
        [
            "-vn",  # no video
            "-acodec",
            "pcm_s16le",  # 16-bit PCM
            "-ar",
            str(sample_rate),  # sample rate
            "-ac",
            "1",  # mono
            "-map_metadata",
            "-1",  # strip source metadata
            "-y",  # overwrite
            str(output_path),
        ]
    )

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr_msg = result.stderr.decode(errors="replace").strip()
        raise subprocess.CalledProcessError(result.returncode, cmd, stderr=stderr_msg)
    return output_path


def _hash_url(url: str) -> str:
    """Short URL hash for cache keys."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def extract_audio_cached(
    video_path: Path | str,
    output_path: Path,
    workspace_dir: Path,
    sample_rate: int = 16000,
    start: str | None = None,
    duration: str | None = None,
    content_hash: str | None = None,
    source_url: str | None = None,
) -> tuple[Path, bool]:
    """Extract audio with caching. Returns (path, cache_hit).

    Cache lives at ``<workspace_dir>/.cache/audio/``. On hit, symlinks
    the cached file into the workspace. On miss, extracts to cache then
    symlinks.

    For local files: uses *content_hash* (SHA-256 of video content) for
    content-addressable caching, falling back to metadata-based keys.

    For stream URLs: uses ``source_url`` hash as the cache key so the
    same episode always hits the cache regardless of which workspace
    requested it.

    Args:
        video_path: Local file path or network URL.
        output_path: Desired output path in workspace.
        workspace_dir: Base workspace directory for the cache.
        sample_rate: Audio sample rate in Hz.
        start: Start time for clipping (ffmpeg format).
        duration: Duration to extract (ffmpeg format).
        content_hash: SHA-256 of video content (local files).
        source_url: Source URL for stream-based caching.

    Returns:
        Tuple of (audio_path, cache_hit).
    """
    cache_dir = get_cache_dir(workspace_dir, "audio")
    params = dict(sample_rate=sample_rate, start=start, duration=duration)

    is_remote = _is_url(video_path)

    # Build cache identity
    if is_remote and source_url:
        cache_identity = _hash_url(source_url)
        resolved_file_path = None
    elif content_hash:
        cache_identity = content_hash
        resolved_file_path = video_path if not is_remote else None
    else:
        cache_identity = None
        resolved_file_path = video_path if not is_remote else None

    # Lookup
    if cache_identity:
        cached_path = find_cached_file(
            cache_dir,
            ".wav",
            content_hash=cache_identity,
            **params,
        )
        if cached_path is not None:
            link_or_copy(cached_path, output_path)
            return output_path, True

    # Determine write key
    if cache_identity:
        key = cache_key(content_hash=cache_identity, **params)
    elif resolved_file_path:
        key = cache_key(resolved_file_path, **params)
    else:
        key = None

    new_cached_path = (
        cache_dir / f"{key}.wav" if key else cache_dir / f"{_hash_url(str(video_path))}.wav"
    )

    extract_audio(
        video_path,
        output_path=new_cached_path,
        sample_rate=sample_rate,
        start=start,
        duration=duration,
    )
    link_or_copy(new_cached_path, output_path)
    return output_path, False
