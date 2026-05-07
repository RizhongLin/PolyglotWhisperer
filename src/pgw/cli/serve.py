"""pgw serve command — local web player for a workspace, powered by FastAPI."""

from __future__ import annotations

import os
import threading
import webbrowser
from pathlib import Path
from typing import Annotated, Optional

import typer

from pgw.server.templates import _discover_tracks, _discover_workspaces
from pgw.utils.console import console, error, warning
from pgw.utils.paths import find_video

# Default bind host. Containers can override via PGW_SERVE_HOST=0.0.0.0.
_DEFAULT_HOST = os.environ.get("PGW_SERVE_HOST", "127.0.0.1")
_DEFAULT_PORT = 8321


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
    """Single-workspace player mode."""
    if not workspace.is_dir():
        error(f"Not a directory: {workspace}")
        raise typer.Exit(1)

    if find_video(workspace) is None:
        warning(f"No video file found in: {workspace}")

    from pgw.server.app import create_workspace_app

    app = create_workspace_app(workspace)
    url = f"http://{host}:{port}"
    console.print(f"[bold green]Serving:[/bold green] {url}")
    console.print(f"[bold]Workspace:[/bold] {workspace}")
    for t in _discover_tracks(workspace):
        console.print(f"  [dim]Track:[/dim] {t['label']} ({t['file']})")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    _run_uvicorn(app, host=host, port=port, no_open=no_open, url=url)


def _serve_library(host: str, port: int, no_open: bool) -> None:
    """Multi-workspace library + end-to-end pipeline UI."""
    from pgw.core.config import load_config
    from pgw.server.app import create_library_app
    from pgw.server.jobs import JobManager

    config = load_config()
    base_dir = Path(config.workspace_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    jobs = JobManager(base_dir=base_dir)
    workspaces = _discover_workspaces(base_dir)

    app = create_library_app(base_dir=base_dir, jobs=jobs)
    url = f"http://{host}:{port}"
    console.print(f"[bold green]Library:[/bold green] {url}")
    console.print(f"[bold]Workspace dir:[/bold] {base_dir}")
    console.print(f"[dim]{len(workspaces)} workspace(s) found[/dim]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    try:
        _run_uvicorn(app, host=host, port=port, no_open=no_open, url=url)
    finally:
        jobs.shutdown(wait=False)


def _run_uvicorn(
    app: object,
    *,
    host: str,
    port: int,
    no_open: bool,
    url: str,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        error("uvicorn is required for `pgw serve`. Install with: uv sync --extra serve")
        raise typer.Exit(1) from exc

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="off",
    )
    server = uvicorn.Server(config)

    if not no_open:
        # Wait until the server is ready before opening the browser so the
        # first GET doesn't 404 on a not-yet-bound socket.
        def _open_when_ready() -> None:
            for _ in range(50):
                if server.started:
                    webbrowser.open(url)
                    return
                threading.Event().wait(0.1)

        threading.Thread(target=_open_when_ready, daemon=True).start()

    try:
        server.run()
    except KeyboardInterrupt:
        console.print("\n[bold]Stopped.[/bold]")
