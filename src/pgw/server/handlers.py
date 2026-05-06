"""HTTP request handlers for the web player and library."""

from __future__ import annotations

import http.server
import mimetypes
import re
import urllib.parse
from pathlib import Path

from pgw.server.templates import (
    _ICON_PNG,
    _LIBRARY_CSS,
    _LOGO_PNG,
    _MIME_MAP,
    _PLAYER_CSS,
    _PLAYER_JS,
    _SIBLING_PREFIX,
    _build_html,
    _build_library_html,
    _discover_workspaces,
    _find_sibling_workspaces,
    _redownload_video_streaming,
)
from pgw.utils.paths import find_video

# ── Constants ──
_DEFAULT_PORT = 8321
_DEFAULT_HOST = "127.0.0.1"
_FILE_CHUNK_SIZE = 1 << 20  # 1 MB
_CACHE_MAX_AGE = 86400  # 1 day


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
            pass

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path.lstrip("/")
        if path == "redownload":
            self._stream_redownload(self.workspace)
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
        safe_name = Path(filename).name
        file_path = self.workspace / safe_name

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
        parts = path.split("/", 3)
        if len(parts) < 3:
            self.send_error(404)
            return

        slug = parts[1]
        timestamp = parts[2]

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
            self._serve_sibling_file(workspace, file_part)
        else:
            self._serve_workspace_file(workspace, file_part)

    def _serve_sibling_file(self, workspace: Path, file_part: str) -> None:
        """Serve a file from a sibling workspace: sibling:<timestamp>/<file>."""
        rest = file_part[len(_SIBLING_PREFIX) :]
        if "/" not in rest:
            self.send_error(404)
            return
        sibling_ts, filename = rest.split("/", 1)
        if not re.match(r"^\d{8}_\d{6}$", sibling_ts):
            self.send_error(404)
            return
        safe_name = Path(filename).name
        slug_dir = workspace.parent
        sibling_dir = slug_dir / sibling_ts
        if not sibling_dir.is_dir():
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
