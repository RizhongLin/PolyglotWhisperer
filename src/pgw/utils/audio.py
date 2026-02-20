"""Audio extraction from video files using ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pgw.utils.cache import cache_key, get_cache_dir, link_or_copy


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def extract_audio(
    video_path: Path,
    output_path: Path | None = None,
    sample_rate: int = 16000,
    start: str | None = None,
    duration: str | None = None,
) -> Path:
    """Extract audio from a video file to WAV format.

    Args:
        video_path: Path to the input video file.
        output_path: Path for the output WAV file. Defaults to
            same directory and stem as video with .wav extension.
        sample_rate: Audio sample rate in Hz. Whisper expects 16000.
        start: Start time for clipping (ffmpeg format: "HH:MM:SS" or seconds).
        duration: Duration to extract (ffmpeg format: "HH:MM:SS" or seconds).

    Returns:
        Path to the extracted audio file.

    Raises:
        FileNotFoundError: If ffmpeg is not installed or video doesn't exist.
        subprocess.CalledProcessError: If ffmpeg fails.
    """
    if not check_ffmpeg():
        raise FileNotFoundError("ffmpeg not found. Install it with: brew install ffmpeg")

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        output_path = video_path.with_suffix(".wav")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
    ]

    if start is not None:
        cmd.extend(["-ss", str(start)])

    cmd.extend(
        [
            "-i",
            str(video_path),
        ]
    )

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


def extract_audio_cached(
    video_path: Path,
    output_path: Path,
    workspace_dir: Path,
    sample_rate: int = 16000,
    start: str | None = None,
    duration: str | None = None,
) -> tuple[Path, bool]:
    """Extract audio with caching. Returns (path, cache_hit).

    Cache lives at <workspace_dir>/.cache/audio/. On hit, symlinks the
    cached file into the workspace. On miss, extracts to cache then symlinks.

    Args:
        video_path: Path to the input video file.
        output_path: Desired output path in workspace.
        workspace_dir: Base workspace directory for the cache.
        sample_rate: Audio sample rate in Hz.
        start: Start time for clipping (ffmpeg format).
        duration: Duration to extract (ffmpeg format).

    Returns:
        Tuple of (audio_path, cache_hit).
    """
    cache_dir = get_cache_dir(workspace_dir, "audio")
    key = cache_key(video_path, sample_rate=sample_rate, start=start, duration=duration)
    cached_path = cache_dir / f"{key}.wav"

    if cached_path.is_file():
        link_or_copy(cached_path, output_path)
        return output_path, True

    # Cache miss — extract to cache, then symlink
    extract_audio(
        video_path,
        output_path=cached_path,
        sample_rate=sample_rate,
        start=start,
        duration=duration,
    )
    link_or_copy(cached_path, output_path)
    return output_path, False
