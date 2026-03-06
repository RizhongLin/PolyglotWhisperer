"""pgw serve command — local web player for a workspace."""

from __future__ import annotations

import functools
import html
import http.server
import json
import mimetypes
import re
import urllib.parse
import webbrowser
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Optional

import typer

from pgw.utils.console import console, error
from pgw.utils.paths import find_video

GITHUB_URL = "https://github.com/RizhongLin/PolyglotWhisperer"

_PLAYER_TEMPLATE = (files("pgw.templates") / "player.html").read_text(encoding="utf-8")
_PLAYER_CSS = (files("pgw.templates") / "player.css").read_text(encoding="utf-8")
_LIBRARY_TEMPLATE = (files("pgw.templates") / "library.html").read_text(encoding="utf-8")
_LIBRARY_CSS = (files("pgw.templates") / "library.css").read_text(encoding="utf-8")
_ICON_PNG = (files("pgw.templates") / "icon.png").read_bytes()
_LOGO_PNG = (files("pgw.templates") / "logo.png").read_bytes()


def _discover_tracks(workspace: Path) -> list[dict[str, str]]:
    """Find subtitle files in a workspace and build track metadata."""
    tracks = []

    # Bilingual VTT first (preferred)
    for f in sorted(workspace.glob("bilingual.*.vtt")):
        parts = f.stem.split(".")
        if len(parts) >= 2:
            lang_pair = parts[1]  # e.g. "fr-en"
            tracks.append({"file": f.name, "label": f"Bilingual ({lang_pair})", "lang": lang_pair})

    # Transcription VTTs
    for f in sorted(workspace.glob("transcription.*.vtt")):
        parts = f.stem.split(".")
        if len(parts) >= 2:
            lang = parts[1]
            tracks.append({"file": f.name, "label": f"Original ({lang})", "lang": lang})

    # Translation VTTs
    for f in sorted(workspace.glob("translation.*.vtt")):
        parts = f.stem.split(".")
        if len(parts) >= 2:
            lang = parts[1]
            tracks.append({"file": f.name, "label": f"Translation ({lang})", "lang": lang})

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
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _format_file_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _load_metadata(workspace: Path) -> dict:
    """Load metadata.json from workspace, return empty dict on failure."""
    meta_path = workspace / "metadata.json"
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


def _build_download_rows(workspace: Path, meta: dict, url_prefix: str = "") -> str:
    """Build <li> entries for downloadable workspace files."""
    skip = {"metadata.json", "audio.wav"}
    files_meta = meta.get("files", {})

    rows = []
    for f in sorted(workspace.iterdir()):
        if f.name.startswith(".") or f.name in skip:
            continue
        if f.suffix.lower() in _MIME_MAP:
            continue
        if not f.is_file():
            continue

        size = files_meta.get(f.name, {}).get("size_bytes", f.stat().st_size)
        size_str = _format_file_size(size)
        safe_name = html.escape(f.name)
        icon = _file_icon(f.suffix.lower())
        label = html.escape(_friendly_name(f.name))
        ext = f.suffix.lstrip(".").upper()
        href = f"{url_prefix}/{safe_name}"
        rows.append(
            f'<li><a href="{href}" download>'
            f'<i data-lucide="{icon}"></i>'
            f'<span class="dl-name">{label}'
            f'<span class="dl-ext">{ext}</span></span></a>'
            f'<span class="size">{size_str}</span></li>'
        )

    return "\n        ".join(rows) if rows else "<li>No files available</li>"


def _build_html(workspace: Path, video_path: Path, url_prefix: str = "") -> str:
    """Generate the player HTML for a workspace."""
    tracks = _discover_tracks(workspace)
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

    video_filename = video_path.name
    video_mime = _MIME_MAP.get(video_path.suffix.lower(), "video/mp4")

    return _PLAYER_TEMPLATE.format(
        title=html.escape(title),
        base_url=url_prefix,
        tracks="\n    ".join(track_tags),
        video_filename=f"{url_prefix}/{video_filename}",
        video_mime=video_mime,
        metadata_rows=_build_metadata_rows(meta),
        download_rows=_build_download_rows(workspace, meta, url_prefix),
        github_url=GITHUB_URL,
    )


def _discover_workspaces(base_dir: Path) -> list[dict]:
    """Find all workspaces under base_dir, return metadata + paths."""
    workspaces = []
    if not base_dir.is_dir():
        return workspaces
    for slug_dir in sorted(base_dir.iterdir()):
        if not slug_dir.is_dir() or slug_dir.name.startswith("."):
            continue
        for ts_dir in sorted(slug_dir.iterdir(), reverse=True):
            if not ts_dir.is_dir() or not re.match(r"\d{8}_\d{6}$", ts_dir.name):
                continue
            meta = _load_metadata(ts_dir)
            if not meta:
                continue
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
                }
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

            meta_chips = []
            lang = ws["language"]
            target = ws["target_language"]
            if lang and target:
                meta_chips.append(
                    f'<span><i data-lucide="languages"></i>'
                    f"{lang.upper()} &rarr; {target.upper()}</span>"
                )
            elif lang:
                meta_chips.append(f'<span><i data-lucide="languages"></i>' f"{lang.upper()}</span>")

            dur = _format_duration(ws["duration"])
            if dur:
                meta_chips.append(f'<span><i data-lucide="clock"></i>' f"{html.escape(dur)}</span>")

            created = ws["created_at"]
            if created:
                meta_chips.append(
                    f'<span><i data-lucide="calendar"></i>' f"{html.escape(created[:10])}</span>"
                )

            meta_html = "\n".join(meta_chips)
            cards.append(
                f'<a href="/ws/{slug}/{ts}/" class="ws-card">'
                f"<article>"
                f'<div class="ws-card-header">'
                f'<div class="ws-icon">'
                f'<i data-lucide="{icon}"></i></div>'
                f"<h3>{title}</h3></div>"
                f'<p class="ws-meta">{meta_html}</p>'
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
    ] = 8321,
    no_open: Annotated[
        bool,
        typer.Option("--no-open", help="Don't open browser automatically."),
    ] = False,
    host: Annotated[
        Optional[str],
        typer.Option("--host", help="Host to bind to."),
    ] = "127.0.0.1",
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
        error(f"No video file found in: {workspace}")
        raise typer.Exit(1)

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

    workspaces = _discover_workspaces(base_dir)
    library_html = _build_library_html(workspaces)

    handler_class = functools.partial(
        _LibraryHandler,
        base_dir=base_dir,
        library_html=library_html,
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
            elif path == "icon.png":
                self._serve_icon()
            elif path == "logo.png":
                self._serve_logo()
            else:
                self._serve_file(path)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browser aborted connection (normal during seeking)

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
            "connect-src 'self'",
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

    def _serve_icon(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(self.icon_png)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(self.icon_png)

    def _serve_logo(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(_LOGO_PNG)))
        self.send_header("Cache-Control", "public, max-age=86400")
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
                while chunk := f.read(1 << 20):  # 1 MB chunks
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
                    chunk = f.read(min(remaining, 1 << 20))
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

    def __init__(self, *args, base_dir: Path, library_html: str, **kwargs):
        self.base_dir = base_dir
        self.library_html = library_html
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path.lstrip("/")

            if path == "" or path == "index.html":
                self._serve_html(self.library_html)
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
            if video_path is None:
                self.send_error(404)
                return
            player_html = _build_html(workspace, video_path, url_prefix=f"/ws/{slug}/{timestamp}")
            self._serve_html(player_html)
        elif file_part == "player.css":
            self._serve_bytes(_PLAYER_CSS.encode("utf-8"), "text/css; charset=utf-8")
        elif file_part == "icon.png":
            self._serve_bytes(_ICON_PNG, "image/png", cache=True)
        elif file_part == "logo.png":
            self._serve_bytes(_LOGO_PNG, "image/png", cache=True)
        else:
            self._serve_workspace_file(workspace, file_part)

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
                while chunk := f.read(1 << 20):
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
            "img-src 'self' data:; "
            "media-src 'self'; "
            "connect-src 'self'",
        )
        self.end_headers()
        self.wfile.write(content)

    def _serve_bytes(self, data: bytes, content_type: str, cache: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=86400")
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
                    chunk = f.read(min(remaining, 1 << 20))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (ValueError, IndexError):
            self.send_error(416)

    def log_message(self, format, *args):
        """Suppress default access logs."""
        pass
