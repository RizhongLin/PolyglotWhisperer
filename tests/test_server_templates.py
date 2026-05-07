"""Tests for the workspace data helpers in server/templates.py.

The HTML rendering that used to live in this module has moved to the
React SPA — only the data-extraction helpers remain (and are tested
here).
"""

from __future__ import annotations

import json
from pathlib import Path

from pgw.server.templates import _discover_tracks, _discover_workspaces, _load_metadata


def _seed(workspace: Path, meta: dict) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def test_load_metadata_round_trip(tmp_path: Path):
    ws = tmp_path / "slug" / "20260101_000000"
    _seed(ws, {"title": "Hello", "language": "fr"})
    assert _load_metadata(ws) == {"title": "Hello", "language": "fr"}


def test_load_metadata_missing(tmp_path: Path):
    assert _load_metadata(tmp_path / "missing") == {}


def test_load_metadata_corrupt(tmp_path: Path):
    ws = tmp_path / "slug" / "20260101_000000"
    ws.mkdir(parents=True)
    (ws / "metadata.json").write_text("not json", encoding="utf-8")
    assert _load_metadata(ws) == {}


def test_discover_tracks_classifies_by_filename(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "transcription.fr.vtt").write_text("WEBVTT\n", encoding="utf-8")
    (ws / "translation.en.vtt").write_text("WEBVTT\n", encoding="utf-8")
    (ws / "bilingual.fr-en.vtt").write_text("WEBVTT\n", encoding="utf-8")
    tracks = _discover_tracks(ws)
    labels = {t["label"] for t in tracks}
    assert "Original (fr)" in labels
    assert "Translation (en)" in labels
    assert "Bilingual (fr-en)" in labels


def test_discover_workspaces_returns_empty_for_missing_dir(tmp_path: Path):
    assert _discover_workspaces(tmp_path / "nope", backfill_metadata=False) == []


def test_discover_workspaces_finds_one(tmp_path: Path):
    ws = tmp_path / "slug" / "20260101_000000"
    _seed(ws, {"title": "Test", "language": "fr"})
    rows = _discover_workspaces(tmp_path, backfill_metadata=False)
    assert len(rows) == 1
    row = rows[0]
    assert row["slug"] == "slug"
    assert row["timestamp"] == "20260101_000000"
    assert row["title"] == "Test"
    assert row["language"] == "fr"


def test_discover_workspaces_skips_dotted_dirs(tmp_path: Path):
    """``.cache``, ``.jobs``, ``.uploads`` must NOT show up as workspaces."""
    cache = tmp_path / ".cache" / "downloads"
    cache.mkdir(parents=True)
    (cache / "metadata.json").write_text("{}", encoding="utf-8")
    rows = _discover_workspaces(tmp_path, backfill_metadata=False)
    assert rows == []


def test_discover_workspaces_merges_same_source_url(tmp_path: Path):
    """Two timestamped runs of the same URL collapse into one entry with
    a sibling_paths list."""
    ws_a = tmp_path / "slug" / "20260101_000000"
    ws_b = tmp_path / "slug" / "20260102_000000"
    common = "https://example.com/v.mp4"
    _seed(ws_a, {"title": "First", "language": "fr", "source_url": common})
    _seed(ws_b, {"title": "Second", "language": "fr", "source_url": common})
    rows = _discover_workspaces(tmp_path, backfill_metadata=False)
    assert len(rows) == 1
    primary = rows[0]
    assert primary["timestamp"] in {"20260101_000000", "20260102_000000"}
    assert len(primary["sibling_paths"]) == 1
