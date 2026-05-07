"""Integration tests for the FastAPI ``pgw serve`` apps.

Uses :class:`fastapi.testclient.TestClient` (synchronous) and a stubbed
``run_pipeline`` so we can exercise the wire format end-to-end without
booting Whisper.
"""

from __future__ import annotations

import json
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
    """Replace ``run_pipeline`` with a fast in-test fake.

    Emits the same stage events the real pipeline would, creates a
    workspace directory the UI can later browse, then returns the path.
    """
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

    # JobManager imports run_pipeline lazily (`from pgw.core.pipeline import run_pipeline`),
    # so patching the source module is sufficient.
    monkeypatch.setattr("pgw.core.pipeline.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("pgw.core.languages.validate_language", lambda code: code, raising=False)
    return workspace


@pytest.fixture
def client(base_dir: Path, jobs: JobManager) -> Iterator[TestClient]:
    app = create_library_app(base_dir=base_dir, jobs=jobs)
    with TestClient(app) as c:
        yield c


# ── library endpoints ────────────────────────────────────────────────────


def test_library_index_renders_form(client: TestClient):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    assert 'id="new-job-dialog"' in body
    assert 'id="jobs-strip"' in body
    assert "/jobs.js" in body


def test_jobs_js_served_with_correct_mime(client: TestClient):
    resp = client.get("/jobs.js")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/javascript")
    # Built bundle should contain the namespaced fetch path.
    assert "/jobs" in resp.text


def test_jobs_js_hot_reloads_with_pgw_dev(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Under ``PGW_DEV``, the jobs.js bundle is re-read on every request.

    Frontend devs running ``cd frontend && npm run watch`` rebuild the
    bundle on save; only a browser refresh should be needed to see the
    change, not a ``pgw serve`` restart.
    """
    import pgw.server.templates as templates_mod

    fake_bundle = tmp_path / "jobs.js"
    fake_bundle.write_text("// version A\n", encoding="utf-8")
    monkeypatch.setattr(templates_mod, "_JOBS_JS_PATH", fake_bundle)
    monkeypatch.setattr(templates_mod, "_JOBS_JS", "// stale cached value\n")
    monkeypatch.setenv("PGW_DEV", "1")

    first = client.get("/jobs.js").text
    assert "version A" in first

    fake_bundle.write_text("// version B\n", encoding="utf-8")
    second = client.get("/jobs.js").text
    assert "version B" in second
    assert "version A" not in second


def test_jobs_js_uses_cached_bundle_without_pgw_dev(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Without ``PGW_DEV``, the cached value is served — disk edits ignored."""
    import pgw.server.templates as templates_mod

    fake_bundle = tmp_path / "jobs.js"
    fake_bundle.write_text("// disk only\n", encoding="utf-8")
    monkeypatch.setattr(templates_mod, "_JOBS_JS_PATH", fake_bundle)
    monkeypatch.setattr(templates_mod, "_JOBS_JS", "// cached\n")
    monkeypatch.delenv("PGW_DEV", raising=False)

    body = client.get("/jobs.js").text
    assert "cached" in body
    assert "disk only" not in body


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
    """Defence-in-depth: even if a server-side path exists, reject it
    unless it lives under the workspace dir."""
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
    # Don't exercise the pipeline; just verify the request is accepted.
    resp = client.post(
        "/jobs",
        json={"input": "https://example.com/video.mp4", "language": "en"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert job_id
    snap = jobs.snapshot(job_id)
    assert snap is not None


# ── job lifecycle through the UI ─────────────────────────────────────────


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
    """After a job finishes, GET /events should replay the full log."""
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
    workspaces = [e for e in events if e.get("type") == "workspace"]
    assert workspaces and workspaces[0]["slug"] == "test-job"


def test_cancel_before_start(
    client: TestClient,
    jobs: JobManager,
    monkeypatch: pytest.MonkeyPatch,
    base_dir: Path,
):
    """Cancel set before run — pipeline raises JobCancelled at first emit."""

    started = threading.Event()

    def slow_pipeline(*args, on_event=None, cancel_token=None, **kwargs):
        started.set()
        # Poll the cancel token like the real pipeline's emit seam does.
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


# ── upload ───────────────────────────────────────────────────────────────


def test_upload_writes_file_under_uploads(client: TestClient, jobs: JobManager):
    payload = b"hello video content"
    resp = client.post(
        "/uploads",
        files={"file": ("clip.mp4", payload, "video/mp4")},
    )
    assert resp.status_code == 201
    data = resp.json()
    saved = data["files"][0]
    assert saved["name"] == "clip.mp4"
    assert saved["size"] == len(payload)
    assert Path(saved["path"]).is_file()
    assert Path(saved["path"]).read_bytes() == payload
    # Path must be under the uploads dir — no traversal escape.
    assert str(Path(saved["path"])).startswith(str(jobs.uploads_dir))


def test_upload_strips_unsafe_chars(client: TestClient):
    resp = client.post(
        "/uploads",
        files={"file": ("../../escape me.mp4", b"x", "video/mp4")},
    )
    assert resp.status_code == 201
    name = resp.json()["files"][0]["name"]
    assert "/" not in name and ".." not in name


# ── workspace app ────────────────────────────────────────────────────────


def test_workspace_app_serves_index(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "metadata.json").write_text(
        json.dumps({"title": "Test", "language": "fr"}), encoding="utf-8"
    )
    app = create_workspace_app(workspace)
    with TestClient(app) as c:
        resp = c.get("/")
        assert resp.status_code == 200
        assert "Test" in resp.text


def test_workspace_app_serves_static_files(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "metadata.json").write_text("{}", encoding="utf-8")
    (workspace / "transcript.fr.txt").write_bytes(b"bonjour")
    app = create_workspace_app(workspace)
    with TestClient(app) as c:
        resp = c.get("/transcript.fr.txt")
        assert resp.status_code == 200
        assert resp.content == b"bonjour"


# ── orphan reaper + retention ────────────────────────────────────────────


def test_orphan_jobs_are_marked_interrupted_on_startup(base_dir: Path):
    jobs_dir = base_dir / ".jobs"
    jobs_dir.mkdir()
    job_id = "a" * 32
    orphan = jobs_dir / f"{job_id}.jsonl"
    # Simulate a partial log left behind by a crashed server.
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
    import os

    jobs_dir = base_dir / ".jobs"
    jobs_dir.mkdir()
    # Drop 5 finished job logs; retention=2 should evict the older 3.
    # Force distinct mtimes so the GC's mtime-sorted ordering is
    # deterministic on filesystems with coarse-grained timestamps.
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
    # Newest two (i=3, i=4) should have survived.
    assert any(name.endswith("3.jsonl") for name in remaining)
    assert any(name.endswith("4.jsonl") for name in remaining)
