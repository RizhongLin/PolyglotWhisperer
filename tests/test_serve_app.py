"""Integration tests for the FastAPI ``pgw serve`` apps.

The web UI is now a React SPA (``frontend/``) — these tests exercise the
JSON APIs the SPA consumes plus the static-shell fallback. The pipeline
is stubbed via ``run_pipeline`` monkeypatch so we don't boot Whisper.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from pgw.core.events import PipelineEvent
from pgw.server.app import create_library_app, create_workspace_app
from pgw.server.jobs import JobManager

# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    d = tmp_path / "ws"
    d.mkdir()
    return d


@pytest.fixture
def jobs(base_dir: Path) -> Iterator[JobManager]:
    mgr = JobManager(base_dir=base_dir)
    yield mgr
    mgr.shutdown(wait=False)


@pytest.fixture
def stub_pipeline(base_dir: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = base_dir / "test-job" / "20260507_100000"

    def fake_run_pipeline(
        input_path: str,
        config,
        translate=None,
        refine=False,
        play=False,
        start=None,
        duration=None,
        on_event=None,
        chunk_size=None,
        cancel_token=None,
    ) -> Path:
        workspace.mkdir(parents=True, exist_ok=True)
        # Plant a metadata.json so workspace-detail endpoints have data.
        (workspace / "metadata.json").write_text(
            json.dumps(
                {
                    "title": "Test job",
                    "language": "en",
                    "source_url": input_path,
                }
            ),
            encoding="utf-8",
        )
        for stage_name, msg in [
            ("download", "Resolving"),
            ("audio", "Extracting"),
            ("transcribe", "Transcribing"),
            ("save", "Done"),
        ]:
            if cancel_token is not None and cancel_token.is_set():
                from pgw.server.exceptions import JobCancelled

                raise JobCancelled()
            data = {"workspace": str(workspace)} if stage_name == "save" else None
            if on_event:
                on_event(PipelineEvent(stage=stage_name, progress=1.0, message=msg, data=data))
        return workspace

    monkeypatch.setattr("pgw.core.pipeline.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("pgw.core.languages.validate_language", lambda code: code, raising=False)
    return workspace


@pytest.fixture
def client(base_dir: Path, jobs: JobManager) -> Iterator[TestClient]:
    app = create_library_app(base_dir=base_dir, jobs=jobs)
    with TestClient(app) as c:
        yield c


# ── SPA shell + assets ───────────────────────────────────────────────────


def test_spa_index_served_at_root(client: TestClient):
    """Root either serves the built SPA shell or the fallback page (in
    dev when frontend/ has not been built). Either way, must be 200/503
    HTML."""
    resp = client.get("/")
    assert resp.status_code in (200, 503)
    assert resp.headers["content-type"].startswith("text/html")


def test_spa_index_catches_deep_links(client: TestClient):
    """Refreshing on a SPA route must hit index.html so the client router
    can resolve it (instead of 404'ing)."""
    resp = client.get("/library/some-slug/20260101_000000")
    assert resp.status_code in (200, 503)
    assert resp.headers["content-type"].startswith("text/html")


# ── Workspace JSON API ───────────────────────────────────────────────────


def test_api_workspaces_empty(client: TestClient):
    resp = client.get("/api/workspaces")
    assert resp.status_code == 200
    assert resp.json() == {"workspaces": []}


def test_api_workspaces_lists_after_run(
    client: TestClient,
    jobs: JobManager,
    stub_pipeline: Path,
):
    job_id = client.post(
        "/jobs",
        json={"input": "https://example.com/v.mp4", "language": "en"},
    ).json()["job_id"]
    _wait_for_terminal(jobs, job_id)
    resp = client.get("/api/workspaces")
    assert resp.status_code == 200
    workspaces = resp.json()["workspaces"]
    assert any(w["slug"] == "test-job" for w in workspaces)


def test_api_workspace_detail(client: TestClient, base_dir: Path):
    """Hand-craft a workspace and verify the detail endpoint."""
    ws = base_dir / "manual" / "20260101_000000"
    ws.mkdir(parents=True)
    (ws / "metadata.json").write_text(
        json.dumps({"title": "Manual", "language": "fr"}), encoding="utf-8"
    )
    (ws / "transcript.fr.txt").write_text("hi", encoding="utf-8")
    resp = client.get("/api/workspaces/manual/20260101_000000")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slug"] == "manual"
    assert body["timestamp"] == "20260101_000000"
    assert body["metadata"]["title"] == "Manual"
    assert any(f["name"] == "transcript.fr.txt" for f in body["files"])


def test_api_workspace_detail_404(client: TestClient):
    resp = client.get("/api/workspaces/nope/20990101_000000")
    assert resp.status_code == 404


def test_api_form_defaults(client: TestClient):
    resp = client.get("/api/config/defaults")
    assert resp.status_code == 200
    body = resp.json()
    assert "language" in body
    assert "backend" in body


# ── /jobs validation ─────────────────────────────────────────────────────


def test_create_job_rejects_unknown_keys(client: TestClient):
    """``extra=forbid`` keeps secrets (api_key) from sneaking in."""
    resp = client.post(
        "/jobs",
        json={
            "input": "https://example.com/video",
            "language": "en",
            "api_key": "leaked-secret",
        },
    )
    assert resp.status_code == 400


def test_create_job_rejects_missing_local_path(client: TestClient):
    resp = client.post(
        "/jobs",
        json={"input": "/nope/does-not-exist.mp4", "language": "en"},
    )
    assert resp.status_code == 400


def test_create_job_rejects_path_outside_base_dir(client: TestClient, tmp_path: Path):
    outsider = tmp_path / "elsewhere.mp4"
    outsider.write_bytes(b"x")
    resp = client.post(
        "/jobs",
        json={"input": str(outsider), "language": "en"},
    )
    assert resp.status_code == 400


def test_create_job_rejects_unsafe_time_strings(client: TestClient):
    resp = client.post(
        "/jobs",
        json={
            "input": "https://example.com/v.mp4",
            "language": "en",
            "start": "; rm -rf /",
        },
    )
    assert resp.status_code == 400


def test_create_job_accepts_url(client: TestClient, jobs: JobManager):
    resp = client.post(
        "/jobs",
        json={"input": "https://example.com/video.mp4", "language": "en"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert job_id
    snap = jobs.snapshot(job_id)
    assert snap is not None


# ── Lifecycle ────────────────────────────────────────────────────────────


def _wait_for_terminal(jobs: JobManager, job_id: str, timeout: float = 5.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = jobs.snapshot(job_id)
        if snap and snap.state in {"succeeded", "failed", "cancelled", "interrupted"}:
            return snap.state
        time.sleep(0.05)
    pytest.fail(f"job {job_id} did not reach terminal state within {timeout}s")


def test_job_runs_to_completion(
    client: TestClient,
    jobs: JobManager,
    stub_pipeline: Path,
):
    resp = client.post(
        "/jobs",
        json={"input": "https://example.com/video.mp4", "language": "en"},
    )
    job_id = resp.json()["job_id"]
    state = _wait_for_terminal(jobs, job_id)
    assert state == "succeeded"
    snap = jobs.snapshot(job_id)
    assert snap is not None
    assert snap.workspace and Path(snap.workspace).is_dir()
    assert snap.slug == "test-job"
    assert snap.timestamp == "20260507_100000"


def test_jobs_list_shows_completed(
    client: TestClient,
    jobs: JobManager,
    stub_pipeline: Path,
):
    resp = client.post(
        "/jobs",
        json={"input": "https://example.com/video.mp4", "language": "en"},
    )
    job_id = resp.json()["job_id"]
    _wait_for_terminal(jobs, job_id)
    listing = client.get("/jobs")
    assert listing.status_code == 200
    ids = [j["id"] for j in listing.json()["jobs"]]
    assert job_id in ids


def test_get_job_returns_record(
    client: TestClient,
    jobs: JobManager,
    stub_pipeline: Path,
):
    job_id = client.post(
        "/jobs",
        json={"input": "https://example.com/v.mp4", "language": "en"},
    ).json()["job_id"]
    _wait_for_terminal(jobs, job_id)
    record = client.get(f"/jobs/{job_id}")
    assert record.status_code == 200
    assert record.json()["id"] == job_id


def test_job_404_for_unknown_id(client: TestClient):
    resp = client.get("/jobs/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    assert resp.status_code == 404


def test_event_stream_replays_terminal(
    client: TestClient,
    jobs: JobManager,
    stub_pipeline: Path,
):
    job_id = client.post(
        "/jobs",
        json={"input": "https://example.com/v.mp4", "language": "en"},
    ).json()["job_id"]
    _wait_for_terminal(jobs, job_id)
    with client.stream("GET", f"/jobs/{job_id}/events") as resp:
        assert resp.status_code == 200
        events = []
        for line in resp.iter_lines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    types = [e.get("type") for e in events]
    assert "record" in types
    assert "terminal" in types
    terminal = [e for e in events if e.get("type") == "terminal"][0]
    assert terminal["state"] == "succeeded"


def test_cancel_before_start(
    client: TestClient,
    jobs: JobManager,
    monkeypatch: pytest.MonkeyPatch,
    base_dir: Path,
):
    started = threading.Event()

    def slow_pipeline(*args, on_event=None, cancel_token=None, **kwargs):
        started.set()
        for _ in range(20):
            time.sleep(0.05)
            if cancel_token is not None and cancel_token.is_set():
                from pgw.server.exceptions import JobCancelled

                raise JobCancelled()
        if on_event:
            on_event(PipelineEvent(stage="download", progress=0.0, message="r", data=None))
        return base_dir / "x"

    monkeypatch.setattr("pgw.core.pipeline.run_pipeline", slow_pipeline)
    monkeypatch.setattr("pgw.core.languages.validate_language", lambda code: code, raising=False)

    resp = client.post(
        "/jobs",
        json={"input": "https://example.com/v.mp4", "language": "en"},
    )
    job_id = resp.json()["job_id"]
    assert started.wait(2.0), "pipeline never started"
    cancel = client.delete(f"/jobs/{job_id}")
    assert cancel.status_code == 200
    assert cancel.json() == {"cancelled": True}
    state = _wait_for_terminal(jobs, job_id)
    assert state == "cancelled"


# ── Upload ───────────────────────────────────────────────────────────────


def test_upload_writes_file_under_uploads(client: TestClient, jobs: JobManager):
    payload = b"hello video content"
    resp = client.post(
        "/uploads",
        files={"file": ("clip.mp4", payload, "video/mp4")},
    )
    assert resp.status_code == 201
    saved = resp.json()["files"][0]
    assert saved["name"] == "clip.mp4"
    assert saved["size"] == len(payload)
    assert Path(saved["path"]).is_file()
    assert Path(saved["path"]).read_bytes() == payload
    assert str(Path(saved["path"])).startswith(str(jobs.uploads_dir))


def test_upload_strips_unsafe_chars(client: TestClient):
    resp = client.post(
        "/uploads",
        files={"file": ("../../escape me.mp4", b"x", "video/mp4")},
    )
    assert resp.status_code == 201
    name = resp.json()["files"][0]["name"]
    assert "/" not in name and ".." not in name


# ── Workspace app ────────────────────────────────────────────────────────


def test_workspace_app_serves_spa_index(tmp_path: Path):
    workspace = tmp_path / "wsdir" / "ws"
    workspace.mkdir(parents=True)
    (workspace / "metadata.json").write_text(
        json.dumps({"title": "Test", "language": "fr"}), encoding="utf-8"
    )
    app = create_workspace_app(workspace)
    with TestClient(app) as c:
        resp = c.get("/")
        assert resp.status_code in (200, 503)
        assert resp.headers["content-type"].startswith("text/html")


def test_workspace_app_serves_workspace_files(tmp_path: Path):
    base = tmp_path / "base"
    workspace = base / "slug" / "20260101_000000"
    workspace.mkdir(parents=True)
    (workspace / "metadata.json").write_text("{}", encoding="utf-8")
    (workspace / "transcript.fr.txt").write_bytes(b"bonjour")
    app = create_workspace_app(workspace)
    with TestClient(app) as c:
        resp = c.get("/ws/slug/20260101_000000/transcript.fr.txt")
        assert resp.status_code == 200
        assert resp.content == b"bonjour"


# ── Orphans + retention ─────────────────────────────────────────────────


def test_orphan_jobs_are_marked_interrupted_on_startup(base_dir: Path):
    jobs_dir = base_dir / ".jobs"
    jobs_dir.mkdir()
    job_id = "a" * 32
    orphan = jobs_dir / f"{job_id}.jsonl"
    orphan.write_text(
        '{"type":"record","id":"' + job_id + '","state":"running","inputs":{}}\n'
        '{"type":"state","state":"running","ts":1.0}\n',
        encoding="utf-8",
    )
    mgr = JobManager(base_dir=base_dir)
    try:
        snap = mgr.snapshot(job_id)
        assert snap is not None
        assert snap.state == "interrupted"
    finally:
        mgr.shutdown(wait=False)


def test_retention_caps_finished_logs(base_dir: Path):
    jobs_dir = base_dir / ".jobs"
    jobs_dir.mkdir()
    for i in range(5):
        jid = f"{'b' * 31}{i}"
        path = jobs_dir / f"{jid}.jsonl"
        path.write_text(
            '{"type":"record","id":"' + jid + '","state":"succeeded","inputs":{}}\n'
            '{"type":"terminal","state":"succeeded","ts":' + str(float(i)) + "}\n",
            encoding="utf-8",
        )
        os.utime(path, (1000.0 + i, 1000.0 + i))
    mgr = JobManager(base_dir=base_dir, retention=2)
    try:
        remaining = sorted(p.name for p in jobs_dir.glob("*.jsonl"))
    finally:
        mgr.shutdown(wait=False)
    assert len(remaining) == 2
    assert any(name.endswith("3.jsonl") for name in remaining)
    assert any(name.endswith("4.jsonl") for name in remaining)
