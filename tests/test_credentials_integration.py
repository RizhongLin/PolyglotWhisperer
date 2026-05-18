"""Integration tests for credential, admin, and DB sync endpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from pgw.auth.csrf import CSRF_COOKIE
from pgw.server.app import create_library_app
from pgw.server.jobs import JobManager


def _csrf_headers(client: TestClient) -> dict:
    csrf = client.cookies.get(CSRF_COOKIE)
    return {"X-CSRF-Token": csrf} if csrf else {}


@pytest.fixture(autouse=True)
def _set_secret_key():
    os.environ.setdefault("PGW_SECRET_KEY", "test-integration-secret-key")


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    base = tmp_path / "ws"
    base.mkdir()
    jobs = JobManager(base_dir=base)
    app = create_library_app(base_dir=base, jobs=jobs)
    with TestClient(app) as c:
        yield c


def _setup_admin(client: TestClient, email: str = "admin@example.com") -> None:
    r = client.post(
        "/api/auth/setup",
        json={"email": email, "password": "password1234"},
    )
    assert r.status_code in (200, 201), r.text


# ── Auth gates: unauthenticated requests ──────────────────────────────


def test_admin_endpoints_require_auth(client):
    assert client.get("/api/admin/users").status_code in (401, 403)
    assert client.post(
        "/api/admin/users",
        json={"email": "n@example.com", "password": "password1234", "is_admin": False},
    ).status_code in (401, 403)


def test_credential_endpoints_require_auth(client):
    assert client.get("/api/auth/credentials").status_code in (401, 403)
    assert client.post(
        "/api/auth/credentials",
        json={"service": "llm", "provider": "openai", "api_key": "sk-test"},
    ).status_code in (401, 403)
    assert client.delete("/api/auth/credentials/1").status_code in (401, 403)


def test_preferences_require_auth(client):
    assert client.get("/api/auth/preferences").status_code in (401, 403)
    assert client.put("/api/auth/preferences", json={"language": "fr"}).status_code in (401, 403)


def test_password_change_requires_auth(client):
    r = client.put(
        "/api/auth/password",
        json={"current_password": "old", "new_password": "newpass1234"},
    )
    assert r.status_code in (401, 403)


# ── Authenticated flows ───────────────────────────────────────────────


def test_credential_create_list_delete(client):
    _setup_admin(client)

    r = client.post(
        "/api/auth/credentials",
        json={"service": "llm", "provider": "openai", "api_key": "sk-test-key-abc123"},
        headers=_csrf_headers(client),
    )
    assert r.status_code == 201
    cred_id = r.json()["id"]

    r = client.get("/api/auth/credentials")
    assert r.status_code == 200
    assert len(r.json()) >= 1

    r = client.delete(f"/api/auth/credentials/{cred_id}", headers=_csrf_headers(client))
    assert r.status_code == 204

    assert client.get("/api/auth/credentials").json() == []


def test_preferences_flow(client):
    _setup_admin(client)

    r = client.get("/api/auth/preferences")
    assert r.status_code == 200
    assert r.json() == {}

    r = client.put(
        "/api/auth/preferences",
        json={"language": "fr", "backend": "api"},
        headers=_csrf_headers(client),
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r = client.get("/api/auth/preferences")
    assert r.json()["language"] == "fr"


def test_admin_list_and_create_user(client):
    _setup_admin(client)

    r = client.get("/api/admin/users")
    assert r.status_code == 200
    assert len(r.json()) >= 1

    r = client.post(
        "/api/admin/users",
        json={
            "email": "user2@example.com",
            "password": "password1234",
            "is_admin": False,
        },
        headers=_csrf_headers(client),
    )
    assert r.status_code == 201

    r = client.get("/api/admin/users")
    assert len(r.json()) == 2


def test_admin_cannot_delete_self(client):
    _setup_admin(client)

    users = client.get("/api/admin/users").json()
    admin_id = users[0]["id"]

    r = client.delete(f"/api/admin/users/{admin_id}", headers=_csrf_headers(client))
    assert r.status_code == 400


def test_admin_list_requires_admin_role(client):
    _setup_admin(client, "a@example.com")

    # Create regular user
    client.post(
        "/api/admin/users",
        json={
            "email": "user@example.com",
            "password": "password1234",
            "is_admin": False,
        },
        headers=_csrf_headers(client),
    )
    client.post("/api/auth/logout", headers=_csrf_headers(client))

    # Login as regular user — may get 200/201/204 (depends on session state)
    r = client.post(
        "/api/auth/login",
        json={
            "email": "user@example.com",
            "password": "password1234",
        },
    )
    assert r.status_code in (200, 201, 204), r.text

    r = client.get("/api/admin/users")
    assert r.status_code == 403


# ── DB sync ───────────────────────────────────────────────────────────


def test_sync_workspace_to_db_creates_row(tmp_path):
    from pgw.db import Base, SessionLocal, get_engine
    from pgw.db.models.user import User
    from pgw.db.models.vocab import VocabEntry
    from pgw.db.models.workspace import Workspace
    from pgw.server.sync import sync_workspace_to_db

    Base.metadata.create_all(get_engine())

    ws_dir = tmp_path / "test-slug" / "20260101_120000"
    ws_dir.mkdir(parents=True)
    ws_dir.joinpath("metadata.json").write_text(
        json.dumps(
            {
                "title": "Test Video",
                "language": "fr",
                "source_duration": 123.4,
                "uploader": "TC",
                "difficulty": "B1",
            }
        ),
        encoding="utf-8",
    )
    ws_dir.joinpath("vocabulary.fr.json").write_text(
        json.dumps(
            {
                "top_rare_words": [
                    {
                        "word": "bonjour",
                        "lemma": "bonjour",
                        "pos": "INTJ",
                        "zipf": 4.5,
                        "difficulty": "B1",
                        "translation": "hello",
                        "context": "Bonjour tout le monde",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with SessionLocal() as db:
        user = User(email="sync@example.com", password_hash="hash", is_admin=True)
        db.add(user)
        db.flush()

        ws_id = sync_workspace_to_db(db, ws_dir, user.id)
        ws = db.get(Workspace, ws_id)
        assert ws is not None
        assert ws.title == "Test Video"
        assert ws.metadata_json["difficulty"] == "B1"

        entries = db.scalars(select(VocabEntry).where(VocabEntry.user_id == user.id)).all()
        assert len(entries) == 1


def test_sync_workspace_idempotent(tmp_path):
    from pgw.db import Base, SessionLocal, get_engine
    from pgw.db.models.user import User
    from pgw.db.models.workspace import Workspace
    from pgw.server.sync import sync_workspace_to_db

    Base.metadata.create_all(get_engine())

    ws_dir = tmp_path / "slug" / "20260101_120000"
    ws_dir.mkdir(parents=True)
    ws_dir.joinpath("metadata.json").write_text(
        json.dumps({"title": "Test", "language": "fr"}), encoding="utf-8"
    )

    with SessionLocal() as db:
        user = User(email="idem@example.com", password_hash="hash")
        db.add(user)
        db.flush()

        id1 = sync_workspace_to_db(db, ws_dir, user.id)
        id2 = sync_workspace_to_db(db, ws_dir, user.id)
        assert id1 == id2

        rows = db.scalars(select(Workspace).where(Workspace.owner_id == user.id)).all()
        assert len(rows) == 1
