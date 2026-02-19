"""Tests for input resolver."""

from pathlib import Path

from pgw.downloader.resolver import is_url, resolve


def test_is_url_http():
    assert is_url("https://www.rts.ch/play/tv/video/123") is True
    assert is_url("http://example.com/video.mp4") is True


def test_is_url_local_path():
    assert is_url("/home/user/video.mp4") is False
    assert is_url("./video.mp4") is False
    assert is_url("video.mp4") is False


def test_resolve_local_file(sample_vtt: Path):
    """Resolve a local file returns VideoSource with the path."""
    source = resolve(str(sample_vtt))
    assert source.video_path == sample_vtt
    assert source.title == sample_vtt.stem
    assert source.source_url is None


def test_resolve_nonexistent_file():
    """Resolve a nonexistent local file raises FileNotFoundError."""
    import pytest

    with pytest.raises(FileNotFoundError):
        resolve("/nonexistent/video.mp4")
