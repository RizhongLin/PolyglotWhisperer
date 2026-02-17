"""Audio extraction from video files using ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


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
            "-y",  # overwrite
            str(output_path),
        ]
    )

    subprocess.run(cmd, check=True, capture_output=True)
    return output_path
