"""P3 contract tests: dispatch routing, job lifecycle, cancel, disconnect.

Tests the full round-trip: SPA → server → worker → server → SPA.

Does NOT boot a real worker subprocess. Instead it tests the server-side
worker-dispatch code paths directly.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterator

import pytest

from pgw.server.exceptions import WorkerNotConnectedError
from pgw.server.jobs import JobManager, JobRequest
from pgw.server.worker_registry import GLOBAL_WORKERS

# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def jobs(tmp_path: Path) -> Iterator[JobManager]:
    """JobManager pointing at a temp directory."""
    d = tmp_path / "ws"
    d.mkdir()
    mgr = JobManager(base_dir=d)
    yield mgr
    mgr.shutdown(wait=False)


# ── helpers ───────────────────────────────────────────────────────────────


def _stub_run_job(jobs: JobManager, monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """Replace ``_run_job`` so server-side submits don't run a pipeline."""
    started = threading.Event()

    def fake_run(job_id: str) -> None:
        started.set()

    monkeypatch.setattr(jobs, "_run_job", fake_run)
    return started


def _mock_worker_connected(monkeypatch: pytest.MonkeyPatch, for_uid: int = 1) -> None:
    """Make ``GLOBAL_WORKERS.is_connected`` return True for ``for_uid``."""
    monkeypatch.setattr(GLOBAL_WORKERS, "is_connected", lambda uid: uid == for_uid)


def _mock_send_to_worker(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Capture frames sent via ``send_threadsafe``."""
    sent: list[dict] = []
    monkeypatch.setattr(
        GLOBAL_WORKERS, "send_threadsafe", lambda uid, frame: sent.append(frame) or True
    )
    return sent


# ── Dispatch routing ─────────────────────────────────────────────────────


def test_submit_worker_explicit_no_worker_raises(jobs: JobManager):
    """``executor='worker'`` with no connected worker raises immediately."""
    req = JobRequest(input="https://example.com/v.mp4", language="en", executor="worker")
    with pytest.raises(WorkerNotConnectedError, match="no worker is connected"):
        jobs.submit(req, user_id=1)


def test_submit_auto_no_worker_falls_to_local(jobs: JobManager, monkeypatch: pytest.MonkeyPatch):
    """``executor='auto'`` with no worker runs in-process."""
    started = _stub_run_job(jobs, monkeypatch)
    req = JobRequest(input="https://example.com/v.mp4", language="en", executor="auto")
    job_id = jobs.submit(req)
    assert job_id
    assert started.wait(2.0), "job never started"


def test_submit_server_runs_locally(jobs: JobManager, monkeypatch: pytest.MonkeyPatch):
    """``executor='server'`` always runs in-process regardless of workers."""
    started = _stub_run_job(jobs, monkeypatch)
    req = JobRequest(input="https://example.com/v.mp4", language="en", executor="server")
    job_id = jobs.submit(req)
    assert job_id
    assert started.wait(2.0), "job never started"


def test_submit_auto_dispatches_to_worker(jobs: JobManager, monkeypatch: pytest.MonkeyPatch):
    """``executor='auto'`` with a connected worker sends ``job.assign``."""
    _mock_worker_connected(monkeypatch, for_uid=42)
    sent = _mock_send_to_worker(monkeypatch)

    req = JobRequest(input="https://example.com/v.mp4", language="en", executor="auto")
    job_id = jobs.submit(req, user_id=42)
    assert job_id

    assign = next((f for f in sent if f.get("type") == "job.assign"), None)
    assert assign is not None, f"no job.assign found in {sent}"
    assert assign["job_id"] == job_id


# ── Remote events ────────────────────────────────────────────────────────


def test_remote_event_updates_snapshot(jobs: JobManager, monkeypatch: pytest.MonkeyPatch):
    """``handle_remote_event`` updates progress/stage/message."""
    _stub_run_job(jobs, monkeypatch)
    req = JobRequest(input="https://example.com/v.mp4", language="en", executor="server")
    job_id = jobs.submit(req)

    jobs.handle_remote_event(
        job_id,
        {"stage": "download", "progress": 0.3, "message": "Resolving…", "data": {}},
    )
    jobs.handle_remote_event(
        job_id,
        {"stage": "transcribe", "progress": 0.9, "message": "Whispering…", "data": {}},
    )

    snap = jobs.snapshot(job_id)
    assert snap is not None
    assert snap.progress == 0.9
    assert snap.stage == "transcribe"
    assert snap.message == "Whispering…"


def test_remote_workspace_sets_metadata(
    jobs: JobManager, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """``handle_remote_workspace`` propagates slug/timestamp/path."""
    _stub_run_job(jobs, monkeypatch)
    ws = tmp_path / "test-slug" / "20260507_120000"
    ws.mkdir(parents=True)
    (ws / "metadata.json").write_text(json.dumps({"title": "Remote job"}), encoding="utf-8")

    req = JobRequest(input="https://example.com/v.mp4", language="en", executor="server")
    job_id = jobs.submit(req)

    jobs.handle_remote_workspace(
        job_id,
        slug="test-slug",
        timestamp="20260507_120000",
        fs_path=str(ws),
    )

    snap = jobs.snapshot(job_id)
    assert snap is not None
    assert snap.slug == "test-slug"
    assert snap.timestamp == "20260507_120000"
    assert snap.workspace is not None and Path(snap.workspace).is_dir()


def test_remote_terminal_maps_to_finished(jobs: JobManager, monkeypatch: pytest.MonkeyPatch):
    """``handle_remote_terminal`` marks job as succeeded."""
    _stub_run_job(jobs, monkeypatch)
    req = JobRequest(input="https://example.com/v.mp4", language="en", executor="server")
    job_id = jobs.submit(req)

    jobs.handle_remote_terminal(job_id, terminal_state="succeeded", error=None)

    snap = jobs.snapshot(job_id)
    assert snap is not None
    assert snap.state == "succeeded"
    assert snap.finished_at is not None
    assert snap.error is None


def test_remote_terminal_carries_error(jobs: JobManager, monkeypatch: pytest.MonkeyPatch):
    """Terminal with an error stores it on the record."""
    _stub_run_job(jobs, monkeypatch)
    req = JobRequest(input="https://example.com/v.mp4", language="en", executor="server")
    job_id = jobs.submit(req)

    jobs.handle_remote_terminal(job_id, terminal_state="failed", error="Whisper OOM")

    snap = jobs.snapshot(job_id)
    assert snap is not None
    assert snap.state == "failed"
    assert snap.error == "Whisper OOM"


# ── Cancel forwarding ────────────────────────────────────────────────────


def test_cancel_forwards_job_cancel_frame(jobs: JobManager, monkeypatch: pytest.MonkeyPatch):
    """``cancel()`` sends ``job.cancel`` via the worker registry."""
    _mock_worker_connected(monkeypatch, for_uid=7)
    _stub_run_job(jobs, monkeypatch)
    sent = _mock_send_to_worker(monkeypatch)

    req = JobRequest(input="https://example.com/v.mp4", language="en", executor="auto")
    job_id = jobs.submit(req, user_id=7)

    ok = jobs.cancel(job_id)
    assert ok

    cancel_frame = next((f for f in sent if f.get("type") == "job.cancel"), None)
    assert cancel_frame is not None
    assert cancel_frame["job_id"] == job_id


# ── Worker disconnect → interrupt ────────────────────────────────────────


def test_mark_jobs_interrupted_flags_running_jobs(
    jobs: JobManager, monkeypatch: pytest.MonkeyPatch
):
    """``mark_jobs_interrupted`` sets state to 'interrupted'."""
    _mock_worker_connected(monkeypatch, for_uid=9)
    _mock_send_to_worker(monkeypatch)
    _stub_run_job(jobs, monkeypatch)

    req1 = JobRequest(input="https://example.com/v1.mp4", language="en", executor="auto")
    jid1 = jobs.submit(req1, user_id=9)
    req2 = JobRequest(input="https://example.com/v2.mp4", language="en", executor="auto")
    jid2 = jobs.submit(req2, user_id=9)

    # One succeeds before disconnect
    jobs.handle_remote_terminal(jid1, terminal_state="succeeded", error=None)

    # Disconnect interrupts the other
    jobs.mark_jobs_interrupted({jid2})

    assert jobs.snapshot(jid1).state == "succeeded"
    assert jobs.snapshot(jid2).state == "interrupted"


# ── NDJSON replay after remote events ────────────────────────────────────


def test_event_stream_replays_remote_events(jobs: JobManager, monkeypatch: pytest.MonkeyPatch):
    """NDJSON stream replays events recorded via handle_remote_*."""
    _stub_run_job(jobs, monkeypatch)
    req = JobRequest(input="https://example.com/v.mp4", language="en", executor="server")
    job_id = jobs.submit(req)

    jobs.handle_remote_event(
        job_id, {"stage": "download", "progress": 0.3, "message": "Resolving…", "data": {}}
    )
    jobs.handle_remote_event(
        job_id, {"stage": "transcribe", "progress": 0.9, "message": "Whispering…", "data": {}}
    )
    jobs.handle_remote_terminal(job_id, terminal_state="succeeded", error=None)

    events: list[dict] = []
    log_path = jobs.jobs_dir / f"{job_id}.jsonl"
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))

    types = [e.get("type") for e in events]
    assert "record" in types
    assert "terminal" in types
    assert any(e.get("type") == "event" for e in events)
    terminal = [e for e in events if e.get("type") == "terminal"][0]
    assert terminal["state"] == "succeeded"


def test_remote_event_frame_shape_matches_local_pipeline(
    jobs: JobManager, monkeypatch: pytest.MonkeyPatch
):
    """Worker → server ``job.event`` frames must fan out with the SAME
    NDJSON shape that ``_make_event_callback`` produces locally.

    Subscribers (the SPA's NDJSON stream) cannot tell whether the
    producer was an in-process pipeline or a remote worker — that is
    the contract.
    """
    from pgw.core.events import PipelineEvent

    _stub_run_job(jobs, monkeypatch)

    # ── Local path: drive the in-process callback for one job ──
    local_req = JobRequest(input="https://example.com/a.mp4", language="en", executor="server")
    local_id = jobs.submit(local_req)
    local_state = jobs._states[local_id]  # noqa: SLF001
    local_cb = jobs._make_event_callback(local_state)  # noqa: SLF001
    local_cb(
        PipelineEvent(
            stage="transcribe",
            progress=0.42,
            message="Whispering…",
            data={"foo": "bar"},
        )
    )

    # ── Remote path: simulate a worker frame for a separate job ──
    remote_req = JobRequest(input="https://example.com/b.mp4", language="en", executor="server")
    remote_id = jobs.submit(remote_req)
    jobs.handle_remote_event(
        remote_id,
        {"stage": "transcribe", "progress": 0.42, "message": "Whispering…", "data": {"foo": "bar"}},
    )

    def _last_event_frame(job_id: str) -> dict:
        log_path = jobs.jobs_dir / f"{job_id}.jsonl"
        for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("type") == "event":
                return payload
        raise AssertionError(f"no event frame in {log_path}")

    local_frame = _last_event_frame(local_id)
    remote_frame = _last_event_frame(remote_id)

    # Identical structure: same keys, same value types, same payload —
    # the only field allowed to differ is ``ts`` (wall-clock at emit).
    assert set(local_frame) == set(
        remote_frame
    ), f"key mismatch: local={set(local_frame)} remote={set(remote_frame)}"
    for k in ("type", "stage", "progress", "message", "data"):
        assert (
            local_frame[k] == remote_frame[k]
        ), f"field {k!r} diverges: local={local_frame[k]!r} remote={remote_frame[k]!r}"
    # ts is per-emit; both must exist and be float-coercible.
    assert isinstance(local_frame["ts"], (int, float))
    assert isinstance(remote_frame["ts"], (int, float))
