"""SQLAlchemy engine factory.

Reads ``PGW_DATABASE_URL`` from the environment. Falls back to a local
SQLite file under the workspace dir so importing this module never
explodes on a developer machine that hasn't started Postgres yet.

Pool size is intentionally small in P1 (default 5). P3 worker WS load
will need a bump — track that as a knob via ``PGW_DB_POOL_SIZE``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from sqlalchemy import Engine, create_engine

_DEFAULT_SQLITE_PATH = Path("./pgw_workspace") / "pgw.db"

_engine_lock = threading.Lock()
_engine: Engine | None = None


def _resolve_url() -> str:
    """Pick a connection URL from env, falling back to local SQLite."""
    explicit = os.environ.get("PGW_DATABASE_URL")
    if explicit:
        return explicit
    _DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{_DEFAULT_SQLITE_PATH.resolve()}"


def _resolve_pool_size() -> int:
    raw = os.environ.get("PGW_DB_POOL_SIZE")
    if not raw:
        return 5
    try:
        return max(1, int(raw))
    except ValueError:
        return 5


def get_engine() -> Engine:
    """Return the process-wide engine, lazily initialised on first call."""
    global _engine
    with _engine_lock:
        if _engine is None:
            url = _resolve_url()
            kwargs: dict[str, object] = {"future": True}
            if url.startswith("sqlite"):
                # check_same_thread=False lets the engine be shared
                # across the FastAPI thread pool. Safe with SQLAlchemy's
                # connection-per-session model.
                kwargs["connect_args"] = {"check_same_thread": False}
            else:
                kwargs["pool_size"] = _resolve_pool_size()
                kwargs["pool_pre_ping"] = True
            _engine = create_engine(url, **kwargs)
    return _engine


def reset_engine() -> None:
    """Drop the cached engine. Tests use this between fixtures."""
    global _engine
    with _engine_lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
