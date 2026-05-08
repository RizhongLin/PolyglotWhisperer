"""Provider detection for the player ``embed`` block.

A workspace may carry a ``source_url`` (typically the YouTube / Vimeo
URL the user originally pointed yt-dlp at). When we can recognise the
provider, the SPA renders a provider-native iframe instead of the
``<video>`` element so users get the original captions, watch-history,
and "watch on YouTube" affordances.

Detection is conservative and never raises — unknown URLs return
``None`` and the player falls back to HTML5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse


@dataclass(frozen=True)
class EmbedTarget:
    """Provider-native embed target for the player.

    ``provider`` is a short stable id (``"youtube"`` / ``"vimeo"``).
    ``embed_url`` is the iframe-ready URL the SPA loads directly.
    ``video_id`` is the provider's id for the video — kept separately
    so the SPA can build the IFrame Player API call without parsing
    URLs in JavaScript.
    """

    provider: str
    embed_url: str
    video_id: str


_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
_YT_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_VIMEO_HOSTS = {"vimeo.com", "www.vimeo.com", "player.vimeo.com"}
_VIMEO_VIDEO_ID = re.compile(r"^[0-9]+$")


def detect_embed(source_url: str | None) -> EmbedTarget | None:
    """Return an ``EmbedTarget`` for known providers, else ``None``.

    Tolerates any input (None / empty / malformed) — bad input always
    returns ``None`` so the caller can fall back to HTML5 video.
    """
    if not source_url:
        return None
    try:
        parsed = urlparse(source_url.strip())
    except (ValueError, AttributeError):
        return None
    if parsed.scheme not in {"http", "https"}:
        return None
    host = (parsed.netloc or "").lower()
    if not host:
        return None

    if host in _YT_HOSTS:
        vid = _extract_youtube_id(parsed)
        if vid:
            return EmbedTarget(
                provider="youtube",
                embed_url=f"https://www.youtube.com/embed/{vid}?enablejsapi=1&rel=0",
                video_id=vid,
            )
        return None

    if host in _VIMEO_HOSTS:
        vid = _extract_vimeo_id(parsed)
        if vid:
            return EmbedTarget(
                provider="vimeo",
                embed_url=f"https://player.vimeo.com/video/{vid}",
                video_id=vid,
            )
        return None

    return None


def _extract_youtube_id(parsed) -> str | None:
    # youtu.be/<id>
    if parsed.netloc.endswith("youtu.be"):
        candidate = parsed.path.lstrip("/").split("/", 1)[0]
        return candidate if _YT_VIDEO_ID.match(candidate) else None
    # /watch?v=<id>
    if parsed.path == "/watch":
        candidates = parse_qs(parsed.query).get("v", [])
        if candidates and _YT_VIDEO_ID.match(candidates[0]):
            return candidates[0]
    # /embed/<id> or /shorts/<id> or /live/<id>
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}:
        if _YT_VIDEO_ID.match(parts[1]):
            return parts[1]
    return None


def _extract_vimeo_id(parsed) -> str | None:
    # vimeo.com/<id> — first numeric segment of the path
    for segment in parsed.path.strip("/").split("/"):
        if _VIMEO_VIDEO_ID.match(segment):
            return segment
    return None
