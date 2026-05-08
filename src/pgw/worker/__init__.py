"""Local worker that connects to a remote pgw server over WebSocket.

The worker runs ``run_pipeline()`` on the user's machine: yt-dlp uses
their IP, Whisper uses their GPU, API keys come from the worker's
``.env``. The server stays a thin orchestrator + identity + library +
flashcard surface.

Modules:
- ``protocol``: framed JSON message types shared with the server.
- ``agent``: long-lived connection loop (P3 ships the handshake; later
  phases add job dispatch + artifact upload + reconnect-with-resume).
- ``cli``: the ``pgw worker`` Typer subcommands.
"""

from __future__ import annotations
