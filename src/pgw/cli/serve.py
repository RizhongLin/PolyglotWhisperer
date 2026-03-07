"""pgw serve command — local web player for a workspace."""

from __future__ import annotations

import functools
import html
import http.server
import json
import mimetypes
import re
import typing
import urllib.parse
import webbrowser
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Optional

import typer

from pgw.utils.console import console, error
from pgw.utils.paths import (
    AUDIO_FILE,
    GITHUB_URL,
    GLOB_BILINGUAL_VTT,
    GLOB_TRANSCRIPTION_VTT,
    GLOB_TRANSLATION_VTT,
    METADATA_FILE,
    find_video,
)
from pgw.utils.text import BYTES_PER_KB, BYTES_PER_MB, SECONDS_PER_HOUR

# ── Constants ──
_DESC_MAX_CHARS = 200  # Truncation limit for descriptions in metadata card
_DEFAULT_PORT = 8321
_DEFAULT_HOST = "127.0.0.1"
_FILE_CHUNK_SIZE = 1 << 20  # 1 MB — chunk size for streaming file responses
_CACHE_MAX_AGE = 86400  # 1 day — Cache-Control max-age for static assets
_DEFAULT_VIDEO_EXT = ".mp4"  # Fallback extension when source has none
_METADATA_FIELDS = ("upload_date", "uploader", "thumbnail", "description")
_SIBLING_PREFIX = "sibling:"  # URL prefix for files from sibling workspaces

_PLAYER_TEMPLATE = (files("pgw.templates") / "player.html").read_text(encoding="utf-8")
_PLAYER_CSS = (files("pgw.templates") / "player.css").read_text(encoding="utf-8")
_LIBRARY_TEMPLATE = (files("pgw.templates") / "library.html").read_text(encoding="utf-8")
_LIBRARY_CSS = (files("pgw.templates") / "library.css").read_text(encoding="utf-8")
_PLAYER_JS = (files("pgw.templates") / "player.js").read_text(encoding="utf-8")
_ICON_PNG = (files("pgw.templates") / "icon.png").read_bytes()
_LOGO_PNG = (files("pgw.templates") / "logo.png").read_bytes()


def _discover_tracks(
    workspace: Path, sibling_paths: list[Path] | None = None
) -> list[dict[str, str]]:
    """Find subtitle files across a workspace and its siblings.

    When *sibling_paths* is provided (library mode), tracks from sibling
    workspaces are included.  File paths for sibling tracks use a
    ``sibling:<timestamp>/<filename>`` scheme so the handler can resolve them.
    """
    all_dirs: list[tuple[Path, str]] = [(workspace, "")]
    if sibling_paths:
        for sp in sibling_paths:
            all_dirs.append((sp, f"{_SIBLING_PREFIX}{sp.name}/"))

    tracks: list[dict[str, str]] = []
    seen_labels: set[str] = set()

    for ws_dir, prefix in all_dirs:
        # Bilingual VTT first (preferred)
        for f in sorted(ws_dir.glob(GLOB_BILINGUAL_VTT)):
            parts = f.stem.split(".")
            if len(parts) >= 2:
                lang_pair = parts[1]
                label = f"Bilingual ({lang_pair})"
                if label not in seen_labels:
                    seen_labels.add(label)
                    tracks.append({"file": prefix + f.name, "label": label, "lang": lang_pair})

        # Transcription VTTs
        for f in sorted(ws_dir.glob(GLOB_TRANSCRIPTION_VTT)):
            parts = f.stem.split(".")
            if len(parts) >= 2:
                lang = parts[1]
                label = f"Original ({lang})"
                if label not in seen_labels:
                    seen_labels.add(label)
                    tracks.append({"file": prefix + f.name, "label": label, "lang": lang})

        # Translation VTTs
        for f in sorted(ws_dir.glob(GLOB_TRANSLATION_VTT)):
            parts = f.stem.split(".")
            if len(parts) >= 2:
                lang = parts[1]
                label = f"Translation ({lang})"
                if label not in seen_labels:
                    seen_labels.add(label)
                    tracks.append({"file": prefix + f.name, "label": label, "lang": lang})

    return tracks


_MIME_MAP = {
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".ts": "video/mp2t",
    ".flv": "video/x-flv",
}


def _format_duration(seconds: float | None) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    if not seconds:
        return ""
    total = int(seconds)
    h, remainder = divmod(total, SECONDS_PER_HOUR)
    m, s = divmod(remainder, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _format_file_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if size_bytes < BYTES_PER_KB:
        return f"{size_bytes} B"
    if size_bytes < BYTES_PER_MB:
        return f"{size_bytes / BYTES_PER_KB:.0f} KB"
    return f"{size_bytes / BYTES_PER_MB:.1f} MB"


def _load_metadata(workspace: Path) -> dict:
    """Load metadata.json from workspace, return empty dict on failure."""
    meta_path = workspace / METADATA_FILE
    if meta_path.is_file():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _build_metadata_rows(meta: dict) -> str:
    """Build <dt>/<dd> pairs for the metadata card."""
    rows = []

    def add(icon: str, label: str, value: str) -> None:
        if value:
            rows.append(
                f'<dt><i data-lucide="{icon}"></i> {html.escape(label)}</dt>' f"<dd>{value}</dd>"
            )

    lang = meta.get("language", "")
    target = meta.get("target_language", "")
    if lang and target:
        add("languages", "Languages", f"{lang.upper()} &rarr; {target.upper()}")
    elif lang:
        add("languages", "Language", lang.upper())

    uploader = meta.get("uploader", "")
    if uploader:
        add("user", "Channel", html.escape(uploader))

    upload_date = meta.get("upload_date", "")
    if upload_date and len(upload_date) == 8:
        formatted = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
        add("calendar-plus", "Uploaded", html.escape(formatted))

    dur = _format_duration(meta.get("source_duration"))
    add("clock", "Duration", html.escape(dur))

    whisper = html.escape(meta.get("whisper_model", ""))
    if whisper:
        add("mic", "Whisper", f"<code>{whisper}</code>")
    llm = html.escape(meta.get("llm_model", ""))
    if llm:
        add("brain", "LLM", f"<code>{llm}</code>")

    source_url = meta.get("source_url")
    if source_url:
        safe_url = html.escape(source_url)
        try:
            domain = urllib.parse.urlparse(source_url).netloc
        except Exception:
            domain = source_url
        add(
            "external-link",
            "Source",
            f'<a href="{safe_url}" target="_blank" rel="noopener"'
            f' title="{safe_url}">{html.escape(domain)}</a>',
        )

    desc = meta.get("description", "")
    if desc:
        # Truncate long descriptions
        short = desc[:_DESC_MAX_CHARS].rstrip() + ("…" if len(desc) > _DESC_MAX_CHARS else "")
        add("text", "Description", html.escape(short))

    created = meta.get("created_at", "")
    if created:
        add("calendar", "Created", html.escape(created[:10]))

    return "\n        ".join(rows)


_FILE_ICONS = {
    ".vtt": "captions",
    ".srt": "captions",
    ".txt": "file-text",
    ".json": "file-json-2",
    ".pdf": "file-text",
    ".epub": "book-open",
}


def _file_icon(suffix: str) -> str:
    """Return a Lucide icon name for a file extension."""
    return _FILE_ICONS.get(suffix, "file")


def _friendly_name(filename: str) -> str:
    """Convert a workspace filename to a human-readable label."""
    stem, _, ext = filename.rpartition(".")
    parts = stem.split(".")
    base = parts[0] if parts else stem
    langs = parts[1] if len(parts) > 1 else ""
    lang_display = langs.upper().replace("-", " \u2192 ") if langs else ""

    names = {
        "bilingual": "Bilingual Subtitles",
        "transcription": "Original Transcription",
        "translation": "Translation",
        "parallel": "Parallel Text",
        "vocabulary": "Vocabulary Analysis",
    }
    label = names.get(base, base.replace("_", " ").title())
    if lang_display:
        label += f" ({lang_display})"
    return label


def _build_download_rows(
    workspace: Path,
    meta: dict,
    url_prefix: str = "",
    sibling_paths: list[Path] | None = None,
) -> str:
    """Build <li> entries for downloadable workspace files."""
    skip = {METADATA_FILE, AUDIO_FILE}

    def _files_for_dir(ws_dir: Path, prefix: str) -> list[tuple[str, str]]:
        """Return (friendly_label_with_ext, html_row) pairs for dedup."""
        if ws_dir == workspace:
            ws_meta = meta.get("files", {})
        else:
            ws_meta = _load_metadata(ws_dir).get("files", {})
        items = []
        for f in sorted(ws_dir.iterdir()):
            if f.name.startswith(".") or f.name in skip:
                continue
            if f.suffix.lower() in _MIME_MAP:
                continue
            if not f.is_file():
                continue

            size = ws_meta.get(f.name, {}).get("size_bytes", f.stat().st_size)
            size_str = _format_file_size(size)
            safe_name = html.escape(f.name)
            icon = _file_icon(f.suffix.lower())
            label = _friendly_name(f.name)
            ext = f.suffix.lstrip(".").upper()
            dedup_key = f"{label}.{ext}"
            href = f"{url_prefix}/{prefix}{safe_name}"
            row_html = (
                f'<li><a href="{href}" download>'
                f'<i data-lucide="{icon}"></i>'
                f'<span class="dl-name">{html.escape(label)}'
                f'<span class="dl-ext">{ext}</span></span></a>'
                f'<span class="size">{size_str}</span></li>'
            )
            items.append((dedup_key, row_html))
        return items

    seen_keys: set[str] = set()
    rows: list[str] = []

    for entry in _files_for_dir(workspace, ""):
        seen_keys.add(entry[0])
        rows.append(entry[1])

    # Include files from siblings, skipping duplicates (same friendly name + ext)
    if sibling_paths:
        for sp in sibling_paths:
            for key, row_html in _files_for_dir(sp, f"{_SIBLING_PREFIX}{sp.name}/"):
                if key not in seen_keys:
                    seen_keys.add(key)
                    rows.append(row_html)

    return "\n        ".join(rows) if rows else "<li>No files available</li>"


def _find_sibling_workspaces(workspace: Path, base_dir: Path) -> list[Path]:
    """Find other workspaces for the same source video.

    Looks at all timestamp dirs under *base_dir* that share the same
    ``source_url`` as *workspace* but are different directories.
    """
    meta = _load_metadata(workspace)
    source_url = meta.get("source_url", "")
    if not source_url:
        return []

    siblings: list[Path] = []
    for slug_dir in base_dir.iterdir():
        if not slug_dir.is_dir() or slug_dir.name.startswith("."):
            continue
        for ts_dir in slug_dir.iterdir():
            if not ts_dir.is_dir() or ts_dir == workspace:
                continue
            if not re.match(r"\d{8}_\d{6}$", ts_dir.name):
                continue
            other_meta = _load_metadata(ts_dir)
            if other_meta.get("source_url") == source_url:
                siblings.append(ts_dir)
    return siblings


def _build_html(
    workspace: Path,
    video_path: Path | None,
    url_prefix: str = "",
    sibling_paths: list[Path] | None = None,
    library_url: str = "",
) -> str:
    """Generate the player HTML for a workspace."""
    tracks = _discover_tracks(workspace, sibling_paths)
    meta = _load_metadata(workspace)
    title = meta.get("title") or workspace.parent.name

    track_tags = []
    for i, t in enumerate(tracks):
        default = " default" if i == 0 else ""
        tag = (
            f'<track kind="subtitles" src="{url_prefix}/{t["file"]}" '
            f'srclang="{t["lang"]}" label="{t["label"]}"{default}>'
        )
        track_tags.append(tag)

    if video_path is not None:
        video_filename = f"{url_prefix}/{video_path.name}"
        video_mime = _MIME_MAP.get(video_path.suffix.lower(), "video/mp4")
        video_section = (
            f'<video id="player" controls autoplay>\n'
            f'    <source src="{video_filename}" type="{video_mime}">\n'
            f"    {chr(10).join(track_tags)}\n"
            f"  </video>"
        )
    else:
        source_url = meta.get("source_url", "")
        thumbnail = meta.get("thumbnail", "")
        if source_url:
            redownload_btn = (
                '<button class="redownload-btn outline" onclick="redownload()">'
                '<i data-lucide="download"></i> Re-download video'
                "</button>"
            )
        else:
            redownload_btn = ""
        # Blurred thumbnail backdrop when available
        if thumbnail:
            backdrop = (
                f'<div class="vm-backdrop">'
                f'<img src="{html.escape(thumbnail)}" alt="">'
                f"</div>"
            )
        else:
            backdrop = ""
        video_section = (
            f'<div class="video-missing">'
            f"{backdrop}"
            f'<div class="vm-content">'
            f'<i data-lucide="video-off"></i>'
            f"<p><strong>Video not available</strong></p>"
            f"<p>The source file may have been moved or cleaned from cache.</p>"
            f"{redownload_btn}"
            f"</div>"
            f"</div>"
        )

    icon_src = f"{url_prefix}/icon.png"
    if library_url:
        brand = (
            f'<p><a href="{html.escape(library_url)}" class="brand-link">'
            f'<i data-lucide="arrow-left" class="brand-back"></i>'
            f'<img src="{icon_src}" class="brand-logo" alt="">'
            f"PolyglotWhisperer</a></p>"
        )
    else:
        brand = f'<p><img src="{icon_src}" class="brand-logo" alt="">' f"PolyglotWhisperer</p>"

    return _PLAYER_TEMPLATE.format(
        title=html.escape(title),
        base_url=url_prefix,
        brand=brand,
        video_section=video_section,
        metadata_rows=_build_metadata_rows(meta),
        download_rows=_build_download_rows(workspace, meta, url_prefix, sibling_paths),
        github_url=GITHUB_URL,
    )


def _merge_workspaces(workspaces: list[dict]) -> list[dict]:
    """Merge workspaces for the same source video into a single entry.

    Workspaces sharing a ``source_url`` are grouped together.  The primary
    workspace (the one shown as the library card) is the one with a video file,
    the most files, and the newest timestamp.  All sibling workspace paths and
    language pairs are collected so the player can load tracks from every run.

    Workspaces without a source_url (local files) are always kept as-is.
    """
    groups: dict[str, list[dict]] = {}  # source_url → [ws, ...]
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
        # Pick primary: has video > most files > newest timestamp
        def _score(w: dict) -> tuple:
            has_video = bool(w.get("has_video"))
            file_count = sum(1 for f in w["path"].iterdir() if not f.name.startswith("."))
            return (has_video, file_count, w["timestamp"])

        group.sort(key=_score, reverse=True)
        primary = group[0]

        # Collect all unique language pairs and sibling paths
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

        # Drop bare source-only pairs when a translated pair exists
        # e.g. (fr, "") is redundant if (fr, "zh") is present
        translated_sources = {src for src, tgt in lang_pairs if tgt}
        lang_pairs = [p for p in lang_pairs if p[1] or p[0] not in translated_sources]

        primary["lang_pairs"] = lang_pairs
        primary["sibling_paths"] = sibling_paths
        merged.append(primary)

    # Local-file workspaces get default values
    for ws in no_url:
        lang = ws.get("language", "")
        target = ws.get("target_language", "")
        ws["lang_pairs"] = [(lang, target)]
        ws["sibling_paths"] = []

    return merged + no_url


def _discover_workspaces(base_dir: Path, backfill_metadata: bool = True) -> list[dict]:
    """Find all workspaces under base_dir, return metadata + paths.

    When *backfill_metadata* is True, workspaces with a source URL but missing
    metadata (thumbnail, uploader, etc.) are auto-refreshed via yt-dlp
    extract_info (no video download).
    """
    workspaces = []
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
            # Check if metadata backfill is needed
            if (
                backfill_metadata
                and meta.get("source_url")
                and not all(meta.get(k) for k in _METADATA_FIELDS)
            ):
                needs_backfill.append(ts_dir)
            lang = meta.get("language", "")
            target = meta.get("target_language", "")
            workspaces.append(
                {
                    "path": ts_dir,
                    "slug": slug_dir.name,
                    "timestamp": ts_dir.name,
                    "title": meta.get("title", slug_dir.name),
                    "language": lang,
                    "target_language": target,
                    "duration": meta.get("source_duration"),
                    "created_at": meta.get("created_at", ""),
                    "has_video": find_video(ts_dir) is not None,
                    "upload_date": meta.get("upload_date", ""),
                    "uploader": meta.get("uploader", ""),
                    "thumbnail": meta.get("thumbnail", ""),
                }
            )

    # Merge: same source_url → single entry with combined languages
    workspaces = _merge_workspaces(workspaces)

    # Backfill missing metadata in background (best-effort)
    if needs_backfill:
        import threading

        def _backfill() -> None:
            for ws_path in needs_backfill:
                try:
                    _refresh_metadata(ws_path)
                except Exception as exc:
                    console.print(
                        f"[dim red]Metadata refresh failed for {ws_path.name}: {exc}[/dim red]"
                    )

        threading.Thread(target=_backfill, daemon=True).start()
        console.print(
            f"[dim]Refreshing metadata for {len(needs_backfill)} workspace(s) in background…[/dim]"
        )

    return workspaces


def _build_library_html(workspaces: list[dict]) -> str:
    """Generate the library HTML listing all workspaces."""
    if not workspaces:
        content = (
            '<div class="ws-empty">'
            '<i data-lucide="folder-open"></i>'
            "<p>No workspaces found.</p>"
            "<p>Run <code>pgw run</code> to process a video first.</p>"
            "</div>"
        )
    else:
        cards = []
        for ws in workspaces:
            title = html.escape(ws["title"])
            slug = html.escape(ws["slug"])
            ts = html.escape(ws["timestamp"])
            icon = "video" if ws["has_video"] else "file-audio"
            thumb = ws.get("thumbnail", "")

            meta_chips = []
            lang_pairs = ws.get("lang_pairs", [])
            if lang_pairs:
                lang_labels = []
                for lang, target in lang_pairs:
                    if lang and target:
                        lang_labels.append(f"{lang.upper()} &rarr; {target.upper()}")
                    elif lang:
                        lang_labels.append(lang.upper())
                if lang_labels:
                    joined = " · ".join(lang_labels)
                    meta_chips.append(f'<span><i data-lucide="languages"></i>{joined}</span>')

            uploader = ws.get("uploader", "")
            if uploader:
                meta_chips.append(
                    f'<span><i data-lucide="user"></i>' f"{html.escape(uploader)}</span>"
                )

            dur = _format_duration(ws["duration"])
            if dur:
                meta_chips.append(f'<span><i data-lucide="clock"></i>' f"{html.escape(dur)}</span>")

            upload_date = ws.get("upload_date", "")
            if upload_date and len(upload_date) == 8:
                formatted = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
                meta_chips.append(
                    f'<span><i data-lucide="calendar-plus"></i>' f"{html.escape(formatted)}</span>"
                )
            else:
                created = ws["created_at"]
                if created:
                    meta_chips.append(
                        f'<span><i data-lucide="calendar"></i>'
                        f"{html.escape(created[:10])}</span>"
                    )

            meta_html = "\n".join(meta_chips)
            if thumb:
                thumb_html = (
                    f'<div class="ws-thumb">'
                    f'<img src="{html.escape(thumb)}" alt="" loading="lazy">'
                    f"</div>"
                )
            else:
                thumb_html = (
                    f'<div class="ws-thumb ws-thumb-placeholder">'
                    f'<i data-lucide="{icon}"></i>'
                    f"</div>"
                )
            cards.append(
                f'<a href="/ws/{slug}/{ts}/" class="ws-card">'
                f"<article>"
                f"{thumb_html}"
                f'<div class="ws-card-body">'
                f"<h3>{title}</h3>"
                f'<p class="ws-meta">{meta_html}</p>'
                f"</div>"
                f"</article></a>"
            )
        content = f'<div class="ws-grid">{"".join(cards)}</div>'

    return _LIBRARY_TEMPLATE.format(
        content=content,
        github_url=GITHUB_URL,
    )


def serve(
    workspace: Annotated[
        Optional[Path],
        typer.Argument(
            help="Workspace directory. Omit to show library.",
        ),
    ] = None,
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to serve on."),
    ] = _DEFAULT_PORT,
    no_open: Annotated[
        bool,
        typer.Option("--no-open", help="Don't open browser automatically."),
    ] = False,
    host: Annotated[
        Optional[str],
        typer.Option("--host", help="Host to bind to."),
    ] = _DEFAULT_HOST,
) -> None:
    """Serve a workspace as a web video player with subtitles."""
    if workspace is None:
        _serve_library(host, port, no_open)
    else:
        _serve_workspace(workspace.resolve(), host, port, no_open)


def _serve_workspace(workspace: Path, host: str, port: int, no_open: bool) -> None:
    """Serve a single workspace as a video player."""
    if not workspace.is_dir():
        error(f"Not a directory: {workspace}")
        raise typer.Exit(1)

    video_path = find_video(workspace)
    if video_path is None:
        from pgw.utils.console import warning

        warning(f"No video file found in: {workspace}")

    player_html = _build_html(workspace, video_path)

    handler_class = functools.partial(
        _WorkspaceHandler,
        workspace=workspace,
        player_html=player_html,
        player_css=_PLAYER_CSS,
        icon_png=_ICON_PNG,
    )
    server = http.server.ThreadingHTTPServer((host, port), handler_class)

    url = f"http://{host}:{port}"
    console.print(f"[bold green]Serving:[/bold green] {url}")
    console.print(f"[bold]Workspace:[/bold] {workspace}")

    tracks = _discover_tracks(workspace)
    for t in tracks:
        console.print(f"  [dim]Track:[/dim] {t['label']} ({t['file']})")

    console.print("[dim]Press Ctrl+C to stop[/dim]")

    if not no_open:
        webbrowser.open(url)

    _run_server(server)


def _serve_library(host: str, port: int, no_open: bool) -> None:
    """Serve the library view listing all workspaces."""
    from pgw.core.config import load_config

    config = load_config()
    base_dir = Path(config.workspace_dir).resolve()

    # Discover once at startup for count + background metadata backfill
    workspaces = _discover_workspaces(base_dir)

    handler_class = functools.partial(
        _LibraryHandler,
        base_dir=base_dir,
    )
    server = http.server.ThreadingHTTPServer((host, port), handler_class)

    url = f"http://{host}:{port}"
    console.print(f"[bold green]Library:[/bold green] {url}")
    console.print(f"[bold]Workspace dir:[/bold] {base_dir}")
    console.print(f"[dim]{len(workspaces)} workspace(s) found[/dim]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    if not no_open:
        webbrowser.open(url)

    _run_server(server)


def _run_server(server: http.server.HTTPServer) -> None:
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[bold]Stopped.[/bold]")
    finally:
        server.server_close()


def _update_workspace_meta(workspace: Path, source: object) -> None:
    """Backfill metadata.json with fields from a VideoSource (or info dict)."""
    meta_path = workspace / METADATA_FILE
    if not meta_path.is_file():
        return
    import json as _json

    existing = _json.loads(meta_path.read_text(encoding="utf-8"))
    changed = False
    for key in _METADATA_FIELDS:
        val = getattr(source, key, None) if hasattr(source, key) else source.get(key)  # type: ignore[union-attr]
        if val and not existing.get(key):
            existing[key] = val
            changed = True
    # Also backfill title and duration
    for src_key, meta_key in (("title", "title"), ("duration", "source_duration")):
        val = getattr(source, src_key, None) if hasattr(source, src_key) else source.get(src_key)  # type: ignore[union-attr]
        if val and not existing.get(meta_key):
            existing[meta_key] = val
            changed = True
    if changed:
        meta_path.write_text(
            _json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _refresh_metadata(workspace: Path) -> bool:
    """Fetch metadata from source URL without downloading the video.

    Uses yt-dlp extract_info(download=False) to get thumbnail, uploader, etc.
    Returns True if metadata was updated.
    """
    meta = _load_metadata(workspace)
    source_url = meta.get("source_url")
    if not source_url:
        return False

    # Skip if all metadata fields already present
    if all(meta.get(k) for k in _METADATA_FIELDS):
        return False

    try:
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(source_url, download=False)

        # Build a dict matching the fields we want
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
    except Exception as exc:
        console.print(f"[dim red]Metadata refresh failed: {exc}[/dim red]")
        return False


def _redownload_video_streaming(workspace: Path, send_event: typing.Callable[[str], None]) -> bool:
    """Re-download video with progress events streamed via send_event.

    Each event is a JSON string: {"progress": 0-100, "status": "...", "detail": "..."}.
    Returns True on success.
    """
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
                        {
                            "progress": 100,
                            "status": "processing",
                            "detail": "Merging formats…",
                        }
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
                msg = {"progress": 0, "status": "error", "detail": "No file found"}
                send_event(json.dumps(msg))
                return False

        # Link into workspace
        video_ext = video_path.suffix or _DEFAULT_VIDEO_EXT
        video_dest = workspace / f"video{video_ext}"
        if video_dest.is_symlink():
            video_dest.unlink()
        link_or_copy(video_path, video_dest)

        # Update metadata
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

        send_event(json.dumps({"progress": 100, "status": "done", "detail": "Complete"}))
        return True
    except Exception as e:
        send_event(json.dumps({"progress": 0, "status": "error", "detail": str(e)}))
        return False


class _WorkspaceHandler(http.server.BaseHTTPRequestHandler):
    """Serve workspace files and the player HTML."""

    def __init__(
        self, *args, workspace: Path, player_html: str, player_css: str, icon_png: bytes, **kwargs
    ):
        self.workspace = workspace
        self.player_html = player_html
        self.player_css = player_css
        self.icon_png = icon_png
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path.lstrip("/")

            if path == "" or path == "index.html":
                self._serve_html()
            elif path == "player.css":
                self._serve_css()
            elif path == "player.js":
                self._serve_js()
            elif path == "icon.png":
                self._serve_icon()
            elif path == "logo.png":
                self._serve_logo()
            else:
                self._serve_file(path)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browser aborted connection (normal during seeking)

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path.lstrip("/")
        if path == "redownload":
            self._stream_redownload(self.workspace)
            # Rebuild player HTML after download
            video_path = find_video(self.workspace)
            self.player_html = _build_html(self.workspace, video_path)
        else:
            self.send_error(404)

    def _stream_redownload(self, workspace: Path) -> None:
        """Stream re-download progress as newline-delimited JSON."""
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

        def send_event(data: str) -> None:
            try:
                self.wfile.write((data + "\n").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        _redownload_video_streaming(workspace, send_event)

    def _serve_html(self) -> None:
        content = self.player_html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "style-src 'self' https://cdn.jsdelivr.net; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data:; "
            "media-src 'self'; "
            "connect-src 'self' https://cdn.jsdelivr.net",
        )
        self.end_headers()
        self.wfile.write(content)

    def _serve_css(self) -> None:
        content = self.player_css.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/css; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_js(self) -> None:
        content = _PLAYER_JS.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/javascript; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_icon(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(self.icon_png)))
        self.send_header("Cache-Control", f"public, max-age={_CACHE_MAX_AGE}")
        self.end_headers()
        self.wfile.write(self.icon_png)

    def _serve_logo(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(_LOGO_PNG)))
        self.send_header("Cache-Control", f"public, max-age={_CACHE_MAX_AGE}")
        self.end_headers()
        self.wfile.write(_LOGO_PNG)

    def _serve_file(self, filename: str) -> None:
        # Only serve files that exist in the workspace (no path traversal)
        safe_name = Path(filename).name
        file_path = self.workspace / safe_name

        if not file_path.is_file():
            self.send_error(404)
            return

        content_type, _ = mimetypes.guess_type(file_path.name)
        if content_type is None:
            content_type = "application/octet-stream"
        # Add charset for text-based formats
        if file_path.suffix in (".vtt", ".srt", ".txt", ".json"):
            content_type = f"{content_type}; charset=utf-8"

        file_size = file_path.stat().st_size

        # Support Range requests for video seeking
        range_header = self.headers.get("Range")
        if range_header and file_path.suffix.lower() in _MIME_MAP:
            self._serve_range(file_path, file_size, content_type, range_header)
        else:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(file_path, "rb") as f:
                while chunk := f.read(_FILE_CHUNK_SIZE):  # 1 MB chunks
                    self.wfile.write(chunk)

    def _serve_range(
        self, file_path: Path, file_size: int, content_type: str, range_header: str
    ) -> None:
        """Handle HTTP Range requests for video seeking."""
        try:
            range_spec = range_header.replace("bytes=", "")
            start_str, end_str = range_spec.split("-", 1)
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            end = min(end, file_size - 1)
            if start < 0 or end < 0 or start > end or start >= file_size:
                self.send_error(416)
                return
            length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(remaining, _FILE_CHUNK_SIZE))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (ValueError, IndexError):
            self.send_error(416)

    def log_message(self, format, *args):
        """Suppress default access logs."""
        pass


class _LibraryHandler(http.server.BaseHTTPRequestHandler):
    """Serve the library view and route /ws/<slug>/<ts>/ to workspace players."""

    def __init__(self, *args, base_dir: Path, **kwargs):
        self.base_dir = base_dir
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path.lstrip("/")

            if path == "" or path == "index.html":
                # Rebuild from disk each time — picks up metadata backfill
                workspaces = _discover_workspaces(self.base_dir, backfill_metadata=False)
                self._serve_html(_build_library_html(workspaces))
            elif path == "library.css":
                self._serve_bytes(_LIBRARY_CSS.encode("utf-8"), "text/css; charset=utf-8")
            elif path == "icon.png":
                self._serve_bytes(_ICON_PNG, "image/png", cache=True)
            elif path == "logo.png":
                self._serve_bytes(_LOGO_PNG, "image/png", cache=True)
            elif path.startswith("ws/"):
                self._route_workspace(path)
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path.lstrip("/")
        if path.startswith("ws/"):
            parts = path.split("/", 4)
            if len(parts) >= 4 and parts[3].rstrip("/") == "redownload":
                slug, timestamp = parts[1], parts[2]
                if re.match(r"^[\w-]+$", slug) and re.match(r"^\d{8}_\d{6}$", timestamp):
                    workspace = self.base_dir / slug / timestamp
                    if workspace.is_dir():
                        self._stream_redownload(workspace)
                        return
            self.send_error(500, "Re-download failed")
        else:
            self.send_error(404)

    def _stream_redownload(self, workspace: Path) -> None:
        """Stream re-download progress as newline-delimited JSON."""
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

        def send_event(data: str) -> None:
            try:
                self.wfile.write((data + "\n").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        _redownload_video_streaming(workspace, send_event)

    def _route_workspace(self, path: str) -> None:
        """Route /ws/<slug>/<timestamp>/... to workspace files."""
        # Parse: ws/<slug>/<timestamp>/<optional file>
        parts = path.split("/", 3)  # ['ws', slug, timestamp, ...]
        if len(parts) < 3:
            self.send_error(404)
            return

        slug = parts[1]
        timestamp = parts[2]

        # Validate slug and timestamp to prevent path traversal
        if not re.match(r"^[\w-]+$", slug):
            self.send_error(404)
            return
        if not re.match(r"^\d{8}_\d{6}$", timestamp):
            self.send_error(404)
            return

        workspace = self.base_dir / slug / timestamp
        if not workspace.is_dir():
            self.send_error(404)
            return

        # Determine what to serve
        file_part = parts[3].rstrip("/") if len(parts) > 3 else ""

        if file_part == "" or file_part == "index.html":
            video_path = find_video(workspace)
            siblings = _find_sibling_workspaces(workspace, self.base_dir)
            player_html = _build_html(
                workspace,
                video_path,
                url_prefix=f"/ws/{slug}/{timestamp}",
                sibling_paths=siblings,
                library_url="/",
            )
            self._serve_html(player_html)
        elif file_part == "player.css":
            self._serve_bytes(_PLAYER_CSS.encode("utf-8"), "text/css; charset=utf-8")
        elif file_part == "player.js":
            self._serve_bytes(_PLAYER_JS.encode("utf-8"), "application/javascript; charset=utf-8")
        elif file_part == "icon.png":
            self._serve_bytes(_ICON_PNG, "image/png", cache=True)
        elif file_part == "logo.png":
            self._serve_bytes(_LOGO_PNG, "image/png", cache=True)
        elif file_part.startswith(_SIBLING_PREFIX):
            # Serve a file from a sibling workspace: sibling:<timestamp>/<file>
            self._serve_sibling_file(workspace, file_part)
        else:
            self._serve_workspace_file(workspace, file_part)

    def _serve_sibling_file(self, workspace: Path, file_part: str) -> None:
        """Serve a file from a sibling workspace: sibling:<timestamp>/<file>."""
        # Parse sibling:<timestamp>/<filename>
        rest = file_part[len(_SIBLING_PREFIX) :]
        if "/" not in rest:
            self.send_error(404)
            return
        sibling_ts, filename = rest.split("/", 1)
        if not re.match(r"^\d{8}_\d{6}$", sibling_ts):
            self.send_error(404)
            return
        safe_name = Path(filename).name
        # Sibling must be under the same slug directory
        slug_dir = workspace.parent
        sibling_dir = slug_dir / sibling_ts
        if not sibling_dir.is_dir():
            # Also check other slug dirs (same source_url may be under a different slug)
            siblings = _find_sibling_workspaces(workspace, self.base_dir)
            sibling_dir = None
            for sp in siblings:
                if sp.name == sibling_ts:
                    sibling_dir = sp
                    break
            if sibling_dir is None:
                self.send_error(404)
                return
        self._serve_workspace_file(sibling_dir, safe_name)

    def _serve_workspace_file(self, workspace: Path, filename: str) -> None:
        """Serve a file from a workspace directory."""
        safe_name = Path(filename).name
        file_path = workspace / safe_name

        if not file_path.is_file():
            self.send_error(404)
            return

        content_type, _ = mimetypes.guess_type(file_path.name)
        if content_type is None:
            content_type = "application/octet-stream"
        if file_path.suffix in (".vtt", ".srt", ".txt", ".json"):
            content_type = f"{content_type}; charset=utf-8"

        file_size = file_path.stat().st_size

        range_header = self.headers.get("Range")
        if range_header and file_path.suffix.lower() in _MIME_MAP:
            self._serve_range(file_path, file_size, content_type, range_header)
        else:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(file_path, "rb") as f:
                while chunk := f.read(_FILE_CHUNK_SIZE):
                    self.wfile.write(chunk)

    def _serve_html(self, content_str: str) -> None:
        content = content_str.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "style-src 'self' https://cdn.jsdelivr.net; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https:; "
            "media-src 'self'; "
            "connect-src 'self' https://cdn.jsdelivr.net",
        )
        self.end_headers()
        self.wfile.write(content)

    def _serve_bytes(self, data: bytes, content_type: str, cache: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if cache:
            self.send_header("Cache-Control", f"public, max-age={_CACHE_MAX_AGE}")
        self.end_headers()
        self.wfile.write(data)

    def _serve_range(
        self, file_path: Path, file_size: int, content_type: str, range_header: str
    ) -> None:
        try:
            range_spec = range_header.replace("bytes=", "")
            start_str, end_str = range_spec.split("-", 1)
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            end = min(end, file_size - 1)
            if start < 0 or end < 0 or start > end or start >= file_size:
                self.send_error(416)
                return
            length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(remaining, _FILE_CHUNK_SIZE))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (ValueError, IndexError):
            self.send_error(416)

    def log_message(self, format, *args):
        """Suppress default access logs."""
        pass
