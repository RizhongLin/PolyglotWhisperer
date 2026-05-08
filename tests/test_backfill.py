"""Tests for the FS → DB backfill maintenance task."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pgw.auth.bootstrap import create_user
from pgw.db import Base, SessionLocal, get_engine
from pgw.maintenance.backfill import run as run_backfill


@pytest.fixture
def db_ready():
    """Schema bootstrap; the autouse `_isolated_db` fixture in conftest
    has already pointed PGW_DATABASE_URL at a tmp SQLite file."""
    Base.metadata.create_all(get_engine())
    yield


def _seed_workspace(base_dir: Path, slug: str, ts: str, *, language: str = "fr") -> Path:
    ws = base_dir / slug / ts
    ws.mkdir(parents=True)
    (ws / "metadata.json").write_text(
        json.dumps(
            {
                "title": f"{slug}/{ts}",
                "language": language,
                "target_language": "en",
                "source_url": f"https://example.com/{slug}",
                "source_duration": 123.4,
                "uploader": "tester",
                "thumbnail": "thumb.jpg",
                "upload_date": "2026-05-07",
            }
        ),
        encoding="utf-8",
    )
    return ws


def _seed_vocab(workspace: Path, language: str, words: list[dict]) -> None:
    (workspace / f"vocabulary.{language}.json").write_text(
        json.dumps({"top_rare_words": words}),
        encoding="utf-8",
    )


def test_backfill_imports_workspace_and_vocab(tmp_path: Path, db_ready) -> None:
    base = tmp_path / "ws"
    base.mkdir()
    ws_dir = _seed_workspace(base, "video1", "20260507_100000")
    _seed_vocab(
        ws_dir,
        "fr",
        [
            {
                "word": "manger",
                "lemma": "manger",
                "pos": "VERB",
                "context": "Je vais manger",
                "translation": "to eat",
                "zipf": 5.1,
                "difficulty": "A1",
            },
            {
                "word": "épistémologie",
                "lemma": "épistémologie",
                "pos": "NOUN",
                "context": "L'épistémologie est complexe",
                "translation": "epistemology",
                "zipf": 1.8,
                "difficulty": "C1",
            },
        ],
    )

    with SessionLocal() as db:
        owner = create_user(
            db, email="admin@example.com", password="hunter22hunter22", is_admin=True
        )
        report = run_backfill(db, owner=owner, base_dir=base)

    assert report.workspaces_imported == 1
    assert report.workspaces_skipped == 0
    assert report.vocab_entries_imported == 2
    assert report.vocab_occurrences_imported == 2


def test_backfill_is_idempotent(tmp_path: Path, db_ready) -> None:
    base = tmp_path / "ws"
    base.mkdir()
    ws_dir = _seed_workspace(base, "v", "20260507_111111")
    _seed_vocab(ws_dir, "fr", [{"word": "x", "lemma": "x", "pos": "NOUN", "zipf": 4.0}])

    with SessionLocal() as db:
        owner = create_user(db, email="a@b.com", password="hunter22hunter22", is_admin=True)
        run_backfill(db, owner=owner, base_dir=base)
        report2 = run_backfill(db, owner=owner, base_dir=base)

    # Second run: no new workspace rows, no new vocab rows.
    assert report2.workspaces_imported == 0
    assert report2.workspaces_skipped == 1
    assert report2.vocab_entries_imported == 0
    assert report2.vocab_occurrences_imported == 0


def test_backfill_skips_unowned_workspaces_for_other_users(tmp_path: Path, db_ready) -> None:
    """Two users running backfill against the same FS independently both
    get their own workspaces. Natural key is (owner_id, slug, timestamp)."""
    base = tmp_path / "ws"
    base.mkdir()
    _seed_workspace(base, "shared", "20260507_120000")

    with SessionLocal() as db:
        u1 = create_user(db, email="u1@example.com", password="hunter22hunter22", is_admin=True)
        u2 = create_user(db, email="u2@example.com", password="hunter22hunter22", is_admin=True)
        r1 = run_backfill(db, owner=u1, base_dir=base)
        r2 = run_backfill(db, owner=u2, base_dir=base)

    assert r1.workspaces_imported == 1
    assert r2.workspaces_imported == 1


def test_backfill_handles_missing_vocab_gracefully(tmp_path: Path, db_ready) -> None:
    base = tmp_path / "ws"
    base.mkdir()
    _seed_workspace(base, "no-vocab", "20260507_130000")
    # No vocabulary.*.json — should still import the workspace itself.

    with SessionLocal() as db:
        owner = create_user(
            db, email="admin@example.com", password="hunter22hunter22", is_admin=True
        )
        report = run_backfill(db, owner=owner, base_dir=base)

    assert report.workspaces_imported == 1
    assert report.vocab_entries_imported == 0


def test_backfill_rejects_unsaved_owner(tmp_path: Path, db_ready) -> None:
    from pgw.db.models.user import User

    base = tmp_path / "ws"
    base.mkdir()
    fake = User(email="ghost@example.com", password_hash="x")  # never persisted
    with SessionLocal() as db:
        with pytest.raises(ValueError):
            run_backfill(db, owner=fake, base_dir=base)
