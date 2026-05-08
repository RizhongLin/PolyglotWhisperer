"""Tests for worker token issuance + the WS handshake.

REST + WebSocket both go through ``TestClient``. Real network is not
involved; FastAPI's TestClient mounts the WS in-process.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from pgw.auth.csrf import CSRF_COOKIE
from pgw.server.app import create_library_app
from pgw.server.jobs import JobManager
from pgw.worker.protocol import (
    PROTOCOL_VERSION,
    HelloFrame,
    ReadyFrame,
    parse_frame,
)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    base = tmp_path / "ws"
    base.mkdir()
    jobs = JobManager(base_dir=base)
    app = create_library_app(base_dir=base, jobs=jobs)
    try:
        with TestClient(app) as c:
            yield c
    finally:
        jobs.shutdown(wait=False)


def _login(client: TestClient) -> None:
    r = client.post(
        "/api/auth/setup",
        json={"email": "u@example.com", "password": "supersecret123"},
    )
    assert r.status_code == 201, r.text


def _csrf(client: TestClient) -> dict[str, str]:
    return {"X-CSRF-Token": client.cookies.get(CSRF_COOKIE) or ""}


# ── REST: token CRUD ──────────────────────────────────────────────────────


def test_create_worker_returns_one_time_token(client: TestClient) -> None:
    _login(client)
    r = client.post("/api/workers", json={"name": "macbook"}, headers=_csrf(client))
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "macbook"
    assert isinstance(body["token"], str)
    assert len(body["token"]) >= 32  # secrets.token_urlsafe(32) → ~43 chars


def test_create_worker_requires_csrf(client: TestClient) -> None:
    _login(client)
    r = client.post("/api/workers", json={"name": "macbook"})
    assert r.status_code == 403


def test_list_workers_requires_auth(client: TestClient) -> None:
    # Setup a user, then drop cookies to hit GET anonymously. GET has no
    # CSRF requirement so this isolates the auth check.
    _login(client)
    client.cookies.clear()
    r = client.get("/api/workers")
    assert r.status_code == 401


def test_list_workers_shows_created(client: TestClient) -> None:
    _login(client)
    client.post("/api/workers", json={"name": "macbook"}, headers=_csrf(client))
    client.post("/api/workers", json={"name": "linux-desktop"}, headers=_csrf(client))
    r = client.get("/api/workers")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2
    names = {row["name"] for row in rows}
    assert names == {"macbook", "linux-desktop"}
    for row in rows:
        assert row["connected"] is False
        assert row["revoked_at"] is None


def test_revoke_worker_idempotent(client: TestClient) -> None:
    _login(client)
    created = client.post("/api/workers", json={"name": "macbook"}, headers=_csrf(client)).json()
    r = client.delete(f"/api/workers/{created['id']}", headers=_csrf(client))
    assert r.status_code == 204
    r2 = client.delete(f"/api/workers/{created['id']}", headers=_csrf(client))
    # Second time the row still exists but already revoked → revoke() flips
    # the timestamp again. Either 204 or 404 is acceptable; current impl
    # returns 204 because the row is still found and writable.
    assert r2.status_code in (204, 404)


def test_revoke_unknown_worker_404(client: TestClient) -> None:
    _login(client)
    r = client.delete("/api/workers/9999", headers=_csrf(client))
    assert r.status_code == 404


# ── WebSocket: handshake ──────────────────────────────────────────────────


def test_ws_rejects_unknown_token(client: TestClient) -> None:
    _login(client)
    with pytest.raises(Exception):
        # WS connection with a bogus token closes with code 4401 — Starlette's
        # TestClient surfaces this as a WebSocketDisconnect on first .receive().
        with client.websocket_connect("/ws/worker?token=not-a-real-token") as ws:
            ws.receive_json()


def test_ws_handshake_happy_path(client: TestClient) -> None:
    _login(client)
    created = client.post("/api/workers", json={"name": "test"}, headers=_csrf(client)).json()
    raw = created["token"]

    with client.websocket_connect(f"/ws/worker?token={raw}") as ws:
        hello_payload = ws.receive_json()
        hello = parse_frame(hello_payload)
        assert isinstance(hello, HelloFrame)
        assert hello.protocol_version == PROTOCOL_VERSION

        ready = ReadyFrame(
            hostname="test-host",
            pgw_version="0.0.0",
            capabilities={"whisper_local": False},
        )
        ws.send_text(ready.model_dump_json())

        # Server stays open. Send a pong unprompted to update last_seen.
        ws.send_text('{"type": "pong"}')
        # Cleanly close from client side.

    # After the WS closes, /api/workers should show the most recent
    # session and that it has a disconnected_at set.
    rows = client.get("/api/workers").json()
    assert rows[0]["connected"] is False


def test_ws_protocol_version_mismatch_closes(client: TestClient) -> None:
    _login(client)
    created = client.post("/api/workers", json={"name": "test"}, headers=_csrf(client)).json()
    raw = created["token"]

    with client.websocket_connect(f"/ws/worker?token={raw}") as ws:
        ws.receive_json()  # hello
        bad = {
            "type": "ready",
            "hostname": "x",
            "pgw_version": "0.0.0",
            "capabilities": {},
            "protocol_version": PROTOCOL_VERSION + 99,
        }
        ws.send_text(__import__("json").dumps(bad))
        # Server closes with code 4426. TestClient raises on next read.
        with pytest.raises(Exception):
            ws.receive_text()


# ── Protocol round-trips ──────────────────────────────────────────────────


def test_protocol_hello_round_trip() -> None:
    h = HelloFrame(server_time=time.time())
    parsed = parse_frame(h.model_dump())
    assert isinstance(parsed, HelloFrame)
    assert parsed.protocol_version == PROTOCOL_VERSION


def test_protocol_unknown_frame_raises() -> None:
    with pytest.raises(ValueError):
        parse_frame({"type": "totally-not-a-frame"})


# ── Worker registry: reconnect / displacement ────────────────────────────


def test_healthz_reports_ok(client: TestClient) -> None:
    """``GET /healthz`` returns 200 with component status."""
    r = client.get("/healthz")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["db"] == "ok"
    assert body["worker_registry"] == "ok"
    assert "workers_known" in body


def test_registry_unregister_with_expected_handle_skips_when_displaced() -> None:
    """A stale ``finally`` must not evict a freshly reconnected worker.

    Models the real bug: an old WS handler's ``finally`` block calls
    ``unregister(user_id, expected=old_handle)`` after a new handler
    has already swapped in via ``register(new_handle)``. The expected
    handle no longer matches, so the new connection survives.
    """
    import asyncio

    from pgw.server.worker_registry import WorkerHandle, WorkerRegistry

    registry = WorkerRegistry()
    loop = asyncio.new_event_loop()
    try:
        old = WorkerHandle(user_id=42, ws=None, loop=loop)  # type: ignore[arg-type]
        new = WorkerHandle(user_id=42, ws=None, loop=loop)  # type: ignore[arg-type]
        registry.register(old)
        registry.register(new)  # displaces old
        assert registry.get(42) is new

        # Stale finally block from the old handler tries to clean up.
        registry.unregister(42, expected=old)
        assert registry.get(42) is new, "stale unregister must not evict the new connection"

        # The new handler's finally block does evict.
        registry.unregister(42, expected=new)
        assert registry.get(42) is None
    finally:
        loop.close()
