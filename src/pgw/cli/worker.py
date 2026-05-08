"""``pgw worker`` Typer subcommand."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

worker_app = typer.Typer(
    name="worker",
    help="Run pgw as a remote worker connected to a server.",
    no_args_is_help=True,
)

console = Console()


@worker_app.command("connect")
def connect(
    server: str = typer.Option(..., "--server", help="Server URL, e.g. https://learn.example.com"),
    token: str = typer.Option(..., "--token", help="Worker token (issued by /api/workers)."),
    hostname: str | None = typer.Option(
        None, "--hostname", help="Hostname override (default: machine hostname)."
    ),
) -> None:
    """Open a long-lived WebSocket and accept jobs from the server."""
    from pgw.worker.agent import run

    console.print(f"[cyan]connecting[/] to {server} …")
    try:
        asyncio.run(run(server, token, hostname=hostname))
    except KeyboardInterrupt:
        console.print("[yellow]disconnected by user[/]")
