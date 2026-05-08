"""End-to-end auth tests via FastAPI TestClient.

Exercises the full bootstrap → login → me → logout cycle plus the
critical safety properties:

- ``POST /api/auth/setup`` only works when no users exist.
- ``GET /api/me`` requires a session.
- CSRF header must accompany state-changing endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from pgw.auth.csrf import CSRF_COOKIE
from pgw.auth.deps import SESSION_COOKIE
from pgw.auth.passwords import hash_password, needs_rehash, verify_password
from pgw.server.app import create_library_app
from pgw.server.jobs import JobManager


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


# ── unit: passwords ──────────────────────────────────────────────────────


def test_password_round_trip() -> None:
    h = hash_password("hunter2-very-secret")
    assert verify_password("hunter2-very-secret", h) is True
    assert verify_password("nope", h) is False


def test_password_rejects_empty() -> None:
    with pytest.raises(ValueError):
        hash_password("")


def test_password_invalid_hash_needs_rehash() -> None:
    assert needs_rehash("not-an-argon2-hash") is True


# ── route: state ─────────────────────────────────────────────────────────


def test_state_no_users(client: TestClient) -> None:
    r = client.get("/api/auth/state")
    assert r.status_code == 200
    body = r.json()
    assert body == {"has_admin": False, "authenticated": False}


# ── route: setup ─────────────────────────────────────────────────────────


def test_setup_creates_admin_and_session(client: TestClient) -> None:
    r = client.post(
        "/api/auth/setup",
        json={"email": "admin@example.com", "password": "supersecret123"},
    )
    assert r.status_code == 201
    # Cookies set
    assert SESSION_COOKIE in r.cookies
    assert CSRF_COOKIE in r.cookies
    # State now reports admin and authenticated
    s = client.get("/api/auth/state")
    assert s.json() == {"has_admin": True, "authenticated": True}


def test_setup_refuses_when_admin_exists(client: TestClient) -> None:
    client.post(
        "/api/auth/setup",
        json={"email": "first@example.com", "password": "supersecret123"},
    )
    r = client.post(
        "/api/auth/setup",
        json={"email": "second@example.com", "password": "supersecret123"},
    )
    assert r.status_code == 409


def test_setup_rejects_invalid_email(client: TestClient) -> None:
    r = client.post(
        "/api/auth/setup",
        json={"email": "not-an-email", "password": "supersecret123"},
    )
    assert r.status_code == 400


def test_setup_rejects_short_password(client: TestClient) -> None:
    r = client.post(
        "/api/auth/setup",
        json={"email": "a@b.com", "password": "short"},
    )
    assert r.status_code == 422


# ── route: login + me + logout ───────────────────────────────────────────


def _setup(client: TestClient, email: str = "u@example.com", pw: str = "supersecret123") -> None:
    r = client.post("/api/auth/setup", json={"email": email, "password": pw})
    assert r.status_code == 201, f"setup precondition failed: {r.status_code} {r.text}"
    # Drop the auto-issued session so login is the next step under test.
    client.cookies.clear()


def test_login_succeeds(client: TestClient) -> None:
    _setup(client)
    r = client.post(
        "/api/auth/login",
        json={"email": "u@example.com", "password": "supersecret123"},
    )
    assert r.status_code == 204
    assert SESSION_COOKIE in r.cookies


def test_login_wrong_password(client: TestClient) -> None:
    _setup(client)
    r = client.post(
        "/api/auth/login",
        json={"email": "u@example.com", "password": "wrong-password"},
    )
    assert r.status_code == 401


def test_login_unknown_user(client: TestClient) -> None:
    _setup(client)
    r = client.post(
        "/api/auth/login",
        json={"email": "ghost@example.com", "password": "supersecret123"},
    )
    assert r.status_code == 401


def test_me_requires_session(client: TestClient) -> None:
    r = client.get("/api/me")
    assert r.status_code == 401


def test_me_returns_user(client: TestClient) -> None:
    _setup(client)
    client.post(
        "/api/auth/login",
        json={"email": "u@example.com", "password": "supersecret123"},
    )
    r = client.get("/api/me")
    assert r.status_code == 200
    body = r.json()
    assert body["email"] == "u@example.com"
    assert body["is_admin"] is True
    assert isinstance(body["id"], int)


def test_logout_requires_csrf(client: TestClient) -> None:
    _setup(client)
    client.post(
        "/api/auth/login",
        json={"email": "u@example.com", "password": "supersecret123"},
    )
    r = client.post("/api/auth/logout")
    # No X-CSRF-Token header → 403.
    assert r.status_code == 403


def test_logout_with_csrf_clears_session(client: TestClient) -> None:
    _setup(client)
    client.post(
        "/api/auth/login",
        json={"email": "u@example.com", "password": "supersecret123"},
    )
    csrf = client.cookies.get(CSRF_COOKIE)
    assert csrf
    r = client.post("/api/auth/logout", headers={"X-CSRF-Token": csrf})
    assert r.status_code == 204
    # /api/me now 401.
    me = client.get("/api/me")
    assert me.status_code == 401


def test_setup_concurrent_requests_only_one_wins(client: TestClient) -> None:
    """Concurrent ``/api/auth/setup`` calls must produce exactly one admin.

    Models the TOCTOU race: two threads both observe an empty users
    table before either commits. The ``setup_lock`` context manager
    serialises within-process so exactly one POST returns 201 and the
    others get 409.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    barrier = threading.Barrier(2)

    def attempt(email: str) -> int:
        # Fresh client per thread so the cookie jars don't conflict.
        # We reuse the same underlying app from the fixture.
        with TestClient(client.app) as sub:
            barrier.wait()
            r = sub.post(
                "/api/auth/setup",
                json={"email": email, "password": "supersecret123"},
            )
            return r.status_code

    with ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(attempt, ["a@example.com", "b@example.com"]))

    # Exactly one wins, exactly one is rejected.
    assert sorted(results) == [201, 409], results
