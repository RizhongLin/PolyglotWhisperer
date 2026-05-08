"""Audio clip extraction + cache."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pgw.audio.clips import _MAX_CLIP_MS, _cache_key, cut_clip, find_workspace_audio


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _make_workspace_with_silence(tmp_path: Path, duration_s: float = 5.0) -> Path:
    """Create a workspace dir with a small silent audio file."""
    if not _has_ffmpeg():
        pytest.skip("ffmpeg not available")
    workspace = tmp_path / "test-slug" / "20260508_120000"
    workspace.mkdir(parents=True)
    audio = workspace / "audio.mp3"
    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=44100:cl=mono:d={duration_s}",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "64k",
            str(audio),
        ],
        check=True,
    )
    return workspace


def test_find_workspace_audio_prefers_audio_dot_ext(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "other.wav").write_bytes(b"x")
    assert find_workspace_audio(workspace) == workspace / "other.wav"
    audio = workspace / "audio.mp3"
    audio.write_bytes(b"x")
    assert find_workspace_audio(workspace) == audio


def test_find_workspace_audio_returns_none_when_empty(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert find_workspace_audio(workspace) is None


def test_cache_key_changes_with_range() -> None:
    p = Path(__file__)  # any real file with stable mtime
    a = _cache_key(p, 0, 1000)
    b = _cache_key(p, 0, 2000)
    c = _cache_key(p, 500, 1500)
    assert a != b
    assert a != c
    assert _cache_key(p, 0, 1000) == a  # stable across calls


def test_cut_clip_invalid_range_raises(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with pytest.raises(ValueError, match="invalid clip range"):
        cut_clip(workspace, start_ms=100, end_ms=50)
    with pytest.raises(ValueError, match="invalid clip range"):
        cut_clip(workspace, start_ms=-1, end_ms=10)
    with pytest.raises(ValueError, match="exceeds cap"):
        cut_clip(workspace, start_ms=0, end_ms=_MAX_CLIP_MS + 1)


def test_cut_clip_no_audio_raises(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with pytest.raises(FileNotFoundError):
        cut_clip(workspace, start_ms=0, end_ms=1000)


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not installed")
def test_cut_clip_creates_and_caches(tmp_path: Path) -> None:
    workspace = _make_workspace_with_silence(tmp_path, duration_s=3.0)
    out1 = cut_clip(workspace, start_ms=500, end_ms=2000)
    assert out1.is_file()
    assert out1.stat().st_size > 0
    out2 = cut_clip(workspace, start_ms=500, end_ms=2000)
    assert out1 == out2
    # Different range → different file
    out3 = cut_clip(workspace, start_ms=0, end_ms=1000)
    assert out3 != out1
    assert out3.is_file()
