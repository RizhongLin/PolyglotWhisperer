"""pgw serve command — local web player for a workspace."""

from __future__ import annotations

import functools
import http.server
import mimetypes
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Annotated, Optional

import typer

from pgw.utils.console import console
from pgw.utils.paths import find_video

PLAYER_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — PolyglotWhisperer</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; color: #e0e0e0; font-family: system-ui, sans-serif; }}
  .container {{ max-width: 960px; margin: 0 auto; padding: 20px; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 16px; color: #a0c4ff; }}
  video {{
    width: 100%; border-radius: 8px;
    background: #000; box-shadow: 0 4px 20px rgba(0,0,0,0.5);
  }}
  video::cue {{ font-size: 1.1rem; background: rgba(0,0,0,0.7); }}
  .controls {{ margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }}
  .controls label {{
    padding: 6px 14px; border-radius: 6px; cursor: pointer;
    background: #16213e; border: 1px solid #334;
    font-size: 0.85rem; transition: all 0.2s;
  }}
  .controls label:hover {{ background: #1a1a4e; }}
  .controls input:checked + span {{ color: #a0c4ff; font-weight: 600; }}
  .controls input {{ display: none; }}
  .info {{ margin-top: 16px; font-size: 0.8rem; color: #888; }}
</style>
</head>
<body>
<div class="container">
  <h1>{title}</h1>
  <video id="player" controls autoplay>
    <source src="/{video_filename}" type="{video_mime}">
    {tracks}
  </video>
  <div class="controls" id="track-controls"></div>
  <div class="info">
    Served by <strong>pgw serve</strong> &mdash; press Ctrl+C to stop
  </div>
</div>
<script>
  const video = document.getElementById('player');
  const controls = document.getElementById('track-controls');
  const tracks = video.textTracks;

  // Build toggle buttons for each track
  for (let i = 0; i < tracks.length; i++) {{
    const t = tracks[i];
    const id = 'track-' + i;
    const label = document.createElement('label');
    label.innerHTML = `<input type="checkbox" id="${{id}}" checked><span>${{t.label}}</span>`;
    label.querySelector('input').addEventListener('change', (e) => {{
      t.mode = e.target.checked ? 'showing' : 'hidden';
    }});
    controls.appendChild(label);
    t.mode = 'showing';
  }}
</script>
</body>
</html>
"""


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


def _build_html(workspace: Path, video_path: Path) -> str:
    """Generate the player HTML for a workspace."""
    tracks = _discover_tracks(workspace)
    title = workspace.parent.name  # slug directory name

    track_tags = []
    for i, t in enumerate(tracks):
        default = " default" if i == 0 else ""
        tag = (
            f'<track kind="subtitles" src="/{t["file"]}" '
            f'srclang="{t["lang"]}" label="{t["label"]}"{default}>'
        )
        track_tags.append(tag)

    video_filename = video_path.name
    video_mime = _MIME_MAP.get(video_path.suffix.lower(), "video/mp4")

    return PLAYER_HTML.format(
        title=title,
        tracks="\n    ".join(track_tags),
        video_filename=video_filename,
        video_mime=video_mime,
    )


def serve(
    workspace: Annotated[
        Path,
        typer.Argument(help="Path to workspace directory."),
    ],
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
    workspace = workspace.resolve()

    if not workspace.is_dir():
        console.print(f"[red]Not a directory:[/red] {workspace}")
        raise typer.Exit(1)

    video_path = find_video(workspace)
    if video_path is None:
        console.print(f"[red]No video file found in:[/red] {workspace}")
        raise typer.Exit(1)

    player_html = _build_html(workspace, video_path)

    handler_class = functools.partial(
        _WorkspaceHandler, workspace=workspace, player_html=player_html
    )
    server = http.server.HTTPServer((host, port), handler_class)

    url = f"http://{host}:{port}"
    console.print(f"[bold green]Serving:[/bold green] {url}")
    console.print(f"[bold]Workspace:[/bold] {workspace}")

    tracks = _discover_tracks(workspace)
    for t in tracks:
        console.print(f"  [dim]Track:[/dim] {t['label']} ({t['file']})")

    console.print("[dim]Press Ctrl+C to stop[/dim]")

    if not no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[bold]Stopped.[/bold]")
    finally:
        server.server_close()


class _WorkspaceHandler(http.server.BaseHTTPRequestHandler):
    """Serve workspace files and the player HTML."""

    def __init__(self, *args, workspace: Path, player_html: str, **kwargs):
        self.workspace = workspace
        self.player_html = player_html
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.lstrip("/")

        if path == "" or path == "index.html":
            self._serve_html()
        else:
            self._serve_file(path)

    def _serve_html(self) -> None:
        content = self.player_html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

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
