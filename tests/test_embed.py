"""Provider detection for the player ``embed`` block."""

from __future__ import annotations

import pytest

from pgw.server.embed import detect_embed


@pytest.mark.parametrize(
    "url,expected_id",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/watch?v=dQw4w9WgXcQ&t=42s", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_detect_youtube_id(url: str, expected_id: str) -> None:
    target = detect_embed(url)
    assert target is not None
    assert target.provider == "youtube"
    assert target.video_id == expected_id
    assert target.embed_url.startswith(f"https://www.youtube.com/embed/{expected_id}")


@pytest.mark.parametrize(
    "url,expected_id",
    [
        ("https://vimeo.com/76979871", "76979871"),
        ("https://www.vimeo.com/76979871", "76979871"),
        ("https://player.vimeo.com/video/76979871", "76979871"),
    ],
)
def test_detect_vimeo_id(url: str, expected_id: str) -> None:
    target = detect_embed(url)
    assert target is not None
    assert target.provider == "vimeo"
    assert target.video_id == expected_id
    assert target.embed_url == f"https://player.vimeo.com/video/{expected_id}"


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "not-a-url",
        "https://example.com/video.mp4",
        "https://www.youtube.com/watch?v=invalid",  # 11-char rule rejects
        "https://vimeo.com/profile-only",
        "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
    ],
)
def test_detect_returns_none_for_unknown(url: str | None) -> None:
    assert detect_embed(url) is None
