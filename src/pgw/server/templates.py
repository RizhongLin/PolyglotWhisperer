"""Workspace data helpers for the FastAPI server.

The HTML rendering that used to live here is gone — the React SPA
(``frontend/``) now consumes JSON over ``/api/...`` and these helpers
just compute the data shapes. The icon/logo bytes, sibling-prefix
constant, and the yt-dlp-backed re-download stream stay because the
JSON endpoints in ``server/app.py`` consume them.
"""

from __future__ import annotations

import json
import re
import typing
from importlib.resources import files
from pathlib import Path

from pgw.utils.console import console
from pgw.utils.paths import (
    GLOB_BILINGUAL_VTT,
    GLOB_TRANSCRIPTION_VTT,
    GLOB_TRANSLATION_VTT,
    GLOB_VOCABULARY_JSON,
    METADATA_FILE,
    find_video,
)
from pgw.utils.text import BYTES_PER_KB, BYTES_PER_MB

# ── Static asset bytes (icons referenced from the SPA shell + favicon) ──
_ICON_PNG = (files("pgw.templates") / "icon.png").read_bytes()
_LOGO_PNG = (files("pgw.templates") / "logo.png").read_bytes()

# ── Constants ──
_DEFAULT_VIDEO_EXT = ".mp4"
_METADATA_FIELDS = ("upload_date", "uploader", "thumbnail", "description")
_SIBLING_PREFIX = "sibling:"


# ── Subtitle track discovery ────────────────────────────────────────────


def _discover_tracks(
    workspace: Path, sibling_paths: list[Path] | None = None
) -> list[dict[str, str]]:
    """Find subtitle files across a workspace and its siblings."""
    all_dirs: list[tuple[Path, str]] = [(workspace, "")]
    if sibling_paths:
        for sp in sibling_paths:
            all_dirs.append((sp, f"{_SIBLING_PREFIX}{sp.name}/"))

    tracks: list[dict[str, str]] = []
    seen_labels: set[str] = set()

    for ws_dir, prefix in all_dirs:
        for f in sorted(ws_dir.glob(GLOB_BILINGUAL_VTT)):
            parts = f.stem.split(".")
            if len(parts) >= 2:
                lang_pair = parts[1]
                label = f"Bilingual ({lang_pair})"
                if label not in seen_labels:
                    seen_labels.add(label)
                    tracks.append({"file": prefix + f.name, "label": label, "lang": lang_pair})

        for f in sorted(ws_dir.glob(GLOB_TRANSCRIPTION_VTT)):
            parts = f.stem.split(".")
            if len(parts) >= 2:
                lang = parts[1]
                label = f"Original ({lang})"
                if label not in seen_labels:
                    seen_labels.add(label)
                    tracks.append({"file": prefix + f.name, "label": label, "lang": lang})

        for f in sorted(ws_dir.glob(GLOB_TRANSLATION_VTT)):
            parts = f.stem.split(".")
            if len(parts) >= 2:
                lang = parts[1]
                label = f"Translation ({lang})"
                if label not in seen_labels:
                    seen_labels.add(label)
                    tracks.append({"file": prefix + f.name, "label": label, "lang": lang})

    return tracks


# ── Metadata I/O ────────────────────────────────────────────────────────


def _load_metadata(workspace: Path) -> dict:
    """Load metadata.json from workspace, return empty dict on failure."""
    meta_path = workspace / METADATA_FILE
    if meta_path.is_file():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


# ── Workspace discovery ─────────────────────────────────────────────────


def _find_sibling_workspaces(workspace: Path, base_dir: Path) -> list[Path]:
    """Find workspaces sharing the same source URL (same source, multiple runs)."""
    meta = _load_metadata(workspace)
    source_url = meta.get("source_url")
    if not source_url:
        return []
    siblings: list[Path] = []
    for slug_dir in sorted(base_dir.iterdir()):
        if not slug_dir.is_dir() or slug_dir.name.startswith("."):
            continue
        for ts_dir in sorted(slug_dir.iterdir()):
            if not ts_dir.is_dir() or ts_dir == workspace:
                continue
            if not re.match(r"\d{8}_\d{6}$", ts_dir.name):
                continue
            other_meta = _load_metadata(ts_dir)
            if other_meta.get("source_url") == source_url:
                siblings.append(ts_dir)
    return siblings


def _merge_workspaces(workspaces: list[dict]) -> list[dict]:
    """Merge workspaces for the same source video into a single entry."""
    groups: dict[str, list[dict]] = {}
    no_url: list[dict] = []

    for ws in workspaces:
        meta = _load_metadata(ws["path"])
        source_url = meta.get("source_url", "") if meta else ""

        if not source_url:
            no_url.append(ws)
            continue

        groups.setdefault(source_url, []).append(ws)

    merged: list[dict] = []
    for _url, group in groups.items():

        def _score(w: dict) -> tuple:
            has_video = bool(w.get("has_video"))
            file_count = sum(1 for f in w["path"].iterdir() if not f.name.startswith("."))
            return (has_video, file_count, w["timestamp"])

        group.sort(key=_score, reverse=True)
        primary = group[0]

        lang_pairs: list[tuple[str, str]] = []
        sibling_paths: list[Path] = []
        for ws in group:
            lang = ws.get("language", "")
            target = ws.get("target_language", "")
            pair = (lang, target)
            if pair not in lang_pairs:
                lang_pairs.append(pair)
            if ws["path"] != primary["path"]:
                sibling_paths.append(ws["path"])

        translated_sources = {src for src, tgt in lang_pairs if tgt}
        lang_pairs = [p for p in lang_pairs if p[1] or p[0] not in translated_sources]

        primary["lang_pairs"] = lang_pairs
        primary["sibling_paths"] = sibling_paths
        merged.append(primary)

    for ws in no_url:
        lang = ws.get("language", "")
        target = ws.get("target_language", "")
        ws["lang_pairs"] = [(lang, target)]
        ws["sibling_paths"] = []

    return merged + no_url


def _discover_workspaces(base_dir: Path, backfill_metadata: bool = True) -> list[dict]:
    """Find all workspaces under base_dir, return metadata + paths."""
    workspaces: list[dict] = []
    if not base_dir.is_dir():
        return workspaces
    needs_backfill: list[Path] = []
    for slug_dir in sorted(base_dir.iterdir()):
        if not slug_dir.is_dir() or slug_dir.name.startswith("."):
            continue
        for ts_dir in sorted(slug_dir.iterdir(), reverse=True):
            if not ts_dir.is_dir() or not re.match(r"\d{8}_\d{6}$", ts_dir.name):
                continue
            meta = _load_metadata(ts_dir)
            if not meta:
                continue
            if (
                backfill_metadata
                and meta.get("source_url")
                and not all(meta.get(k) for k in _METADATA_FIELDS)
            ):
                needs_backfill.append(ts_dir)

            difficulty = ""
            for vocab_file in sorted(ts_dir.glob(GLOB_VOCABULARY_JSON)):
                try:
                    vdata = json.loads(vocab_file.read_text(encoding="utf-8"))
                    difficulty = vdata.get("estimated_difficulty", "")
                except (json.JSONDecodeError, OSError):
                    pass
                break

            workspaces.append(
                {
                    "path": ts_dir,
                    "slug": slug_dir.name,
                    "timestamp": ts_dir.name,
                    "title": meta.get("title", slug_dir.name),
                    "language": meta.get("language", ""),
                    "target_language": meta.get("target_language", ""),
                    "duration": meta.get("source_duration"),
                    "created_at": meta.get("created_at", ""),
                    "has_video": find_video(ts_dir) is not None or bool(meta.get("video_url")),
                    "upload_date": meta.get("upload_date", ""),
                    "uploader": meta.get("uploader", ""),
                    "thumbnail": meta.get("thumbnail", ""),
                    "difficulty": difficulty,
                }
            )

    workspaces = _merge_workspaces(workspaces)

    if needs_backfill:
        import threading

        def _backfill() -> None:
            for ws_path in needs_backfill:
                try:
                    _refresh_metadata(ws_path)
                except Exception as exc:  # noqa: BLE001
                    console.print(
                        f"[dim red]Metadata refresh failed for {ws_path.name}: {exc}[/dim red]"
                    )

        threading.Thread(target=_backfill, daemon=True).start()
        console.print(
            f"[dim]Refreshing metadata for {len(needs_backfill)} workspace(s) in background…[/dim]"
        )

    return workspaces


# ── Metadata backfill (used by yt-dlp re-download) ──────────────────────


def _update_workspace_meta(workspace: Path, source: object) -> None:
    """Backfill metadata.json with fields from a VideoSource (or info dict)."""
    meta_path = workspace / METADATA_FILE
    if not meta_path.is_file():
        return
    existing = json.loads(meta_path.read_text(encoding="utf-8"))
    changed = False
    for key in _METADATA_FIELDS:
        val = getattr(source, key, None) if hasattr(source, key) else source.get(key)  # type: ignore[union-attr]
        if val and not existing.get(key):
            existing[key] = val
            changed = True
    for src_key, meta_key in (("title", "title"), ("duration", "source_duration")):
        val = getattr(source, src_key, None) if hasattr(source, src_key) else source.get(src_key)  # type: ignore[union-attr]
        if val and not existing.get(meta_key):
            existing[meta_key] = val
            changed = True
    if changed:
        meta_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _refresh_metadata(workspace: Path) -> bool:
    """Fetch metadata from source URL without downloading the video."""
    meta = _load_metadata(workspace)
    source_url = meta.get("source_url")
    if not source_url:
        return False

    if all(meta.get(k) for k in _METADATA_FIELDS):
        return False

    try:
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(source_url, download=False)

        from types import SimpleNamespace

        source = SimpleNamespace(
            upload_date=info.get("upload_date"),
            uploader=info.get("channel") or info.get("uploader"),
            thumbnail=info.get("thumbnail"),
            description=info.get("description"),
            title=info.get("title"),
            duration=info.get("duration"),
        )
        _update_workspace_meta(workspace, source)
        return True
    except Exception as exc:  # noqa: BLE001
        console.print(f"[dim red]Metadata refresh failed: {exc}[/dim red]")
        return False


# ── Re-download video stream ────────────────────────────────────────────


def _redownload_video_streaming(workspace: Path, send_event: typing.Callable[[str], None]) -> bool:
    """Re-download video with progress events streamed via send_event."""
    meta = _load_metadata(workspace)
    source_url = meta.get("source_url")
    if not source_url:
        send_event(json.dumps({"progress": 0, "status": "error", "detail": "No source URL"}))
        return False

    try:
        import yt_dlp

        from pgw.core.config import load_config
        from pgw.utils.cache import link_or_copy

        config = load_config()
        output_dir = Path(config.download_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fmt = config.download.format

        def progress_hook(d: dict) -> None:
            if d["status"] == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded = d.get("downloaded_bytes", 0)
                pct = (downloaded / total * 100) if total > 0 else 0
                speed = d.get("speed")
                eta = d.get("eta")
                detail_parts = []
                if speed:
                    if speed > BYTES_PER_MB:
                        detail_parts.append(f"{speed / BYTES_PER_MB:.1f} MB/s")
                    else:
                        detail_parts.append(f"{speed / BYTES_PER_KB:.0f} KB/s")
                if eta is not None:
                    m, s = divmod(int(eta), 60)
                    detail_parts.append(f"ETA {m}:{s:02d}")
                send_event(
                    json.dumps(
                        {
                            "progress": round(pct, 1),
                            "status": "downloading",
                            "detail": " · ".join(detail_parts),
                        }
                    )
                )
            elif d["status"] == "finished":
                send_event(
                    json.dumps(
                        {"progress": 100, "status": "processing", "detail": "Merging formats…"}
                    )
                )

        opts = {
            "format": fmt,
            "outtmpl": str(output_dir / "%(title)s_%(id)s.%(ext)s"),
            "progress_hooks": [progress_hook],
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }

        send_event(json.dumps({"progress": 0, "status": "starting", "detail": "Connecting…"}))

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(source_url, download=True)
            video_path = Path(ydl.prepare_filename(info))

        if not video_path.is_file():
            candidates = sorted(
                (p for p in output_dir.iterdir() if not p.name.startswith(".")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                video_path = candidates[0]
            else:
                send_event(
                    json.dumps({"progress": 0, "status": "error", "detail": "No file found"})
                )
                return False

        video_ext = video_path.suffix or _DEFAULT_VIDEO_EXT
        video_dest = workspace / f"video{video_ext}"
        if video_dest.is_symlink():
            video_dest.unlink()
        link_or_copy(video_path, video_dest)

        from types import SimpleNamespace

        source = SimpleNamespace(
            upload_date=info.get("upload_date"),
            uploader=info.get("channel") or info.get("uploader"),
            thumbnail=info.get("thumbnail"),
            description=info.get("description"),
            title=info.get("title"),
            duration=info.get("duration"),
        )
        _update_workspace_meta(workspace, source)

        # After successful re-download, remove any stale stream URL so
        # the SPA prefers the newly downloaded local file.
        _remove_meta_keys(workspace, "video_url", "audio_url")

        send_event(json.dumps({"progress": 100, "status": "done", "detail": "Complete"}))
        return True
    except Exception as e:  # noqa: BLE001
        send_event(json.dumps({"progress": 0, "status": "error", "detail": str(e)}))
        return False


def _remove_meta_keys(workspace: Path, *keys: str) -> None:
    """Remove keys from workspace metadata.json (idempotent)."""
    import json

    meta_path = workspace / "metadata.json"
    if not meta_path.is_file():
        return
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    changed = False
    for key in keys:
        if key in data:
            del data[key]
            changed = True
    if changed:
        meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
