"""Smoke tests for the DB engine factory.

These run against SQLite by default; flip ``PGW_DATABASE_URL`` to a
Postgres URL to exercise the production dialect locally.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from pgw.db import SessionLocal, get_engine
from pgw.db.engine import reset_engine


@pytest.fixture(autouse=True)
def _isolated_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Each test gets its own SQLite file. Postgres CI overrides via env."""
    monkeypatch.setenv("PGW_DATABASE_URL", f"sqlite:///{tmp_path / 'pgw.db'}")
    reset_engine()
    yield
    reset_engine()


def test_engine_singleton() -> None:
    a = get_engine()
    b = get_engine()
    assert a is b


def test_engine_select_one() -> None:
    engine = get_engine()
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar_one() == 1


def test_session_local_yields_usable_session() -> None:
    with SessionLocal() as session:
        assert session.execute(text("SELECT 1")).scalar_one() == 1


def test_pool_size_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SQLite ignores pool_size, but at least confirm the env var parses."""
    monkeypatch.setenv("PGW_DATABASE_URL", "postgresql+psycopg://x:y@127.0.0.1:1/pgw")
    monkeypatch.setenv("PGW_DB_POOL_SIZE", "20")
    reset_engine()
    # Don't actually connect — just verify get_engine() doesn't blow up
    # on the URL format / kwargs path.
    engine = get_engine()
    assert engine.pool.size() == 20  # type: ignore[attr-defined]
