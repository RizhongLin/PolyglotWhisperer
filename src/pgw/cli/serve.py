"""pgw serve command — local web player for a workspace."""

from __future__ import annotations

import functools
import http.server
import webbrowser
from pathlib import Path
from typing import Annotated, Optional

import typer

from pgw.server.handlers import (
    _DEFAULT_HOST,
    _DEFAULT_PORT,
    _ICON_PNG,
    _PLAYER_CSS,
    _LibraryHandler,
    _WorkspaceHandler,
)
from pgw.server.templates import (
    _build_html,
    _discover_tracks,
    _discover_workspaces,
)
from pgw.utils.console import console, error, warning
from pgw.utils.paths import find_video


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
