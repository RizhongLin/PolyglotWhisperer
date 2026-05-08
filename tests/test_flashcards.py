"""End-to-end tests for the flashcards REST surface + FSRS review."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from pgw.auth.csrf import CSRF_COOKIE
from pgw.db.models.workspace import Workspace
from pgw.db.session import SessionLocal
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


def _setup_user(client: TestClient) -> dict[str, str]:
    """Create the admin + return CSRF headers for state-changing requests."""
    r = client.post(
        "/api/auth/setup",
        json={"email": "u@example.com", "password": "supersecret123"},
    )
    assert r.status_code == 201, r.text
    csrf = client.cookies.get(CSRF_COOKIE) or ""
    return {"X-CSRF-Token": csrf}


def _create_workspace_for_user(email: str, slug: str = "demo") -> int:
    """Insert a Workspace row owned by the user. Returns workspace id."""
    with SessionLocal() as db:
        from pgw.db.models.user import User

        owner = db.scalar(select(User).where(User.email == email))
        assert owner is not None
        ws = Workspace(
            owner_id=owner.id,
            slug=slug,
            timestamp="20260508_120000",
            title="Demo",
            source_url=None,
            source_language="fr",
            target_language="en",
            duration_seconds=120.0,
            fs_path=f"/tmp/{slug}",
            metadata_json={},
        )
        db.add(ws)
        db.commit()
        db.refresh(ws)
        return ws.id


def test_create_flashcard_minimal(client: TestClient) -> None:
    csrf = _setup_user(client)
    ws_id = _create_workspace_for_user("u@example.com")

    r = client.post(
        "/api/flashcards",
        json={
            "workspace_id": ws_id,
            "front": "bonjour",
            "back": "hello",
            "language": "fr",
        },
        headers=csrf,
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["front"] == "bonjour"
    assert body["back"] == "hello"
    assert body["language"] == "fr"
    assert body["audio_start_ms"] is None
    assert body["audio_end_ms"] is None
    assert isinstance(body["id"], int)


def test_create_flashcard_requires_csrf(client: TestClient) -> None:
    _setup_user(client)
    ws_id = _create_workspace_for_user("u@example.com")
    r = client.post(
        "/api/flashcards",
        json={"workspace_id": ws_id, "front": "x", "back": "y", "language": "fr"},
    )
    assert r.status_code == 403


def test_create_flashcard_rejects_other_users_workspace(client: TestClient) -> None:
    """The route must scope by owner_id — no cross-user card creation."""
    csrf = _setup_user(client)
    # Insert a workspace owned by a fictional second user.
    with SessionLocal() as db:
        from pgw.db.models.user import User

        other = User(email="other@example.com", password_hash="x", is_admin=False)
        db.add(other)
        db.commit()
        db.refresh(other)
        ws = Workspace(
            owner_id=other.id,
            slug="other",
            timestamp="20260508_120000",
            title="Other",
            fs_path="/tmp/other",
        )
        db.add(ws)
        db.commit()
        db.refresh(ws)
        other_ws_id = ws.id

    r = client.post(
        "/api/flashcards",
        json={
            "workspace_id": other_ws_id,
            "front": "x",
            "back": "y",
            "language": "fr",
        },
        headers=csrf,
    )
    assert r.status_code == 404


def test_create_flashcard_rejects_invalid_audio_range(client: TestClient) -> None:
    csrf = _setup_user(client)
    ws_id = _create_workspace_for_user("u@example.com")
    r = client.post(
        "/api/flashcards",
        json={
            "workspace_id": ws_id,
            "front": "x",
            "back": "y",
            "language": "fr",
            "audio_start_ms": 5000,
            "audio_end_ms": 2000,
        },
        headers=csrf,
    )
    assert r.status_code == 400


def test_review_queue_returns_cards_due_now(client: TestClient) -> None:
    csrf = _setup_user(client)
    ws_id = _create_workspace_for_user("u@example.com")
    for i in range(3):
        client.post(
            "/api/flashcards",
            json={
                "workspace_id": ws_id,
                "front": f"front-{i}",
                "back": f"back-{i}",
                "language": "fr",
            },
            headers=csrf,
        )
    r = client.get("/api/flashcards/queue?limit=10")
    assert r.status_code == 200
    cards = r.json()
    assert len(cards) == 3
    fronts = sorted(c["front"] for c in cards)
    assert fronts == ["front-0", "front-1", "front-2"]


def test_review_advances_due(client: TestClient) -> None:
    """Submitting a Good (3) review pushes the card's due date forward."""
    csrf = _setup_user(client)
    ws_id = _create_workspace_for_user("u@example.com")
    created = client.post(
        "/api/flashcards",
        json={"workspace_id": ws_id, "front": "x", "back": "y", "language": "fr"},
        headers=csrf,
    ).json()

    def _parse(stamp: str) -> datetime:
        # SQLite drops tz info; backfill UTC so the comparisons line up.
        dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    initial_due = _parse(created["fsrs_due"])
    r = client.post(
        f"/api/flashcards/{created['id']}/review",
        json={"rating": 3, "elapsed_ms": 4500},
        headers=csrf,
    )
    assert r.status_code == 200
    updated = r.json()
    new_due = _parse(updated["fsrs_due"])
    assert new_due > initial_due
    assert new_due > datetime.now(timezone.utc) + timedelta(minutes=1)


def test_review_records_log(client: TestClient) -> None:
    """A review insert must persist a row in flashcard_reviews."""
    from pgw.db.models.flashcard import FlashcardReview

    csrf = _setup_user(client)
    ws_id = _create_workspace_for_user("u@example.com")
    card = client.post(
        "/api/flashcards",
        json={"workspace_id": ws_id, "front": "x", "back": "y", "language": "fr"},
        headers=csrf,
    ).json()
    client.post(
        f"/api/flashcards/{card['id']}/review",
        json={"rating": 2},
        headers=csrf,
    )
    with SessionLocal() as db:
        rows = list(
            db.scalars(select(FlashcardReview).where(FlashcardReview.flashcard_id == card["id"]))
        )
    assert len(rows) == 1
    assert rows[0].rating == 2


def test_review_invalid_rating(client: TestClient) -> None:
    csrf = _setup_user(client)
    ws_id = _create_workspace_for_user("u@example.com")
    card = client.post(
        "/api/flashcards",
        json={"workspace_id": ws_id, "front": "x", "back": "y", "language": "fr"},
        headers=csrf,
    ).json()
    r = client.post(
        f"/api/flashcards/{card['id']}/review",
        json={"rating": 99},
        headers=csrf,
    )
    assert r.status_code == 422  # Pydantic Literal[1,2,3,4] check


def test_delete_flashcard(client: TestClient) -> None:
    csrf = _setup_user(client)
    ws_id = _create_workspace_for_user("u@example.com")
    card = client.post(
        "/api/flashcards",
        json={"workspace_id": ws_id, "front": "x", "back": "y", "language": "fr"},
        headers=csrf,
    ).json()
    r = client.delete(f"/api/flashcards/{card['id']}", headers=csrf)
    assert r.status_code == 204
    # Subsequent review fails with 404
    r2 = client.post(
        f"/api/flashcards/{card['id']}/review",
        json={"rating": 3},
        headers=csrf,
    )
    assert r2.status_code == 404


def test_review_rejects_other_users_card(client: TestClient) -> None:
    """A logged-in user must not be able to review another user's card."""
    csrf = _setup_user(client)
    # Insert a card owned by a different user directly via the DB.
    with SessionLocal() as db:
        from pgw.db.models.flashcard import Flashcard
        from pgw.db.models.user import User

        other = User(email="other@example.com", password_hash="x", is_admin=False)
        db.add(other)
        db.commit()
        db.refresh(other)
        ws = Workspace(
            owner_id=other.id,
            slug="other",
            timestamp="20260508_120000",
            title="Other",
            fs_path="/tmp/other",
        )
        db.add(ws)
        db.commit()
        db.refresh(ws)
        card = Flashcard(
            user_id=other.id,
            workspace_id=ws.id,
            front="x",
            back="y",
            language="fr",
            fsrs_stability=0.0,
            fsrs_difficulty=0.0,
            fsrs_due=datetime.now(timezone.utc),
        )
        db.add(card)
        db.commit()
        db.refresh(card)
        other_card_id = card.id

    r = client.post(
        f"/api/flashcards/{other_card_id}/review",
        json={"rating": 3},
        headers=csrf,
    )
    assert r.status_code == 404, r.text


def test_delete_flashcard_requires_csrf(client: TestClient) -> None:
    csrf = _setup_user(client)
    ws_id = _create_workspace_for_user("u@example.com")
    card = client.post(
        "/api/flashcards",
        json={"workspace_id": ws_id, "front": "x", "back": "y", "language": "fr"},
        headers=csrf,
    ).json()
    # No CSRF header → 403
    r = client.delete(f"/api/flashcards/{card['id']}")
    assert r.status_code == 403


def test_audio_clip_requires_auth(client: TestClient) -> None:
    """Anonymous callers must not be able to stream workspace audio."""
    _setup_user(client)
    _create_workspace_for_user("u@example.com")
    # Drop the auth cookies the setup flow handed back.
    client.cookies.clear()
    r = client.get("/api/workspaces/demo/20260508_120000/audio-clip?start=0&end=1000")
    assert r.status_code == 401, r.text


def test_list_flashcards_filters_by_workspace(client: TestClient) -> None:
    csrf = _setup_user(client)
    ws_a = _create_workspace_for_user("u@example.com", slug="a")
    ws_b = _create_workspace_for_user("u@example.com", slug="b")
    for ws_id in (ws_a, ws_a, ws_b):
        client.post(
            "/api/flashcards",
            json={"workspace_id": ws_id, "front": "x", "back": "y", "language": "fr"},
            headers=csrf,
        )
    r = client.get(f"/api/flashcards?workspace_id={ws_a}")
    assert r.status_code == 200
    cards = r.json()
    assert len(cards) == 2
    assert all(c["workspace_id"] == ws_a for c in cards)
