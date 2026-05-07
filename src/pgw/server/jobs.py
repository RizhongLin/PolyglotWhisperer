"""In-process job manager for the web UI.

Owns a pool of background pipeline workers, a per-job append-only JSONL
event log under ``<workspace_dir>/.jobs/<job_id>.jsonl``, and a fan-out
broadcaster so multiple HTTP clients can tail the same job (e.g. user
opens two tabs). Designed to be resilient to browser refresh: the log is
the source of truth, and clients subscribe via "replay-then-tail" so a
disconnected client can reconnect at any time.

Public surface used by ``server/app.py``:

- ``JobManager.submit(req: JobRequest) -> str`` — enqueue a job, return id.
- ``JobManager.cancel(job_id: str) -> bool`` — request cancellation.
- ``JobManager.snapshot(job_id: str) -> JobRecord | None``
- ``JobManager.list() -> list[JobRecord]``
- ``JobManager.stream(job_id: str) -> Iterator[bytes]`` — NDJSON bytes.

Cancellation is between-stages only — the pipeline checks the
``threading.Event`` cancel token at every ``emit(...)`` call site.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pgw.core.events import PipelineEvent
from pgw.server.exceptions import JobCancelled

logger = logging.getLogger(__name__)

# Sentinel object pushed onto subscriber queues when the job is terminal.
_END: Any = object()
# Sentinel signalling that a slow subscriber should reconnect (we drop them
# rather than hold up writers).
_DISCONNECT: Any = object()
# Heartbeat cadence on quiet streams (server → client).
_HEARTBEAT_INTERVAL_SEC = 15.0
# Per-subscriber queue depth — client too slow → drop the subscriber so it
# reconnects via replay-from-log, never block writers.
_QUEUE_MAXSIZE = 256
# Default cap on retained terminal job logs.
_DEFAULT_RETENTION = 200
# Default worker pool size — keep Whisper local model warm.
_DEFAULT_MAX_WORKERS = 1
# Validate ffmpeg-style time strings to prevent unexpected flag injection
# into ffmpeg argv (subprocess uses list form, so this is defence-in-depth).
_TIME_RE = re.compile(r"^[0-9:.,\s]+$")

JobState = Literal[
    "pending",
    "running",
    "cancelling",
    "cancelled",
    "succeeded",
    "failed",
    "interrupted",
]
_TERMINAL_STATES: frozenset[JobState] = frozenset(
    {"cancelled", "succeeded", "failed", "interrupted"}
)


class JobRequest(BaseModel):
    """Validated form payload for ``POST /jobs``.

    ``extra="forbid"`` so unknown keys are rejected — keeps secrets
    (api_key, tokens) from sneaking in over the wire. API keys remain
    env-only per project policy.
    """

    model_config = ConfigDict(extra="forbid")

    input: str = Field(min_length=1)
    language: str = Field(min_length=2, max_length=8)
    translate: str | None = None
    backend: Literal["local", "api"] | None = None
    llm_backend: Literal["local", "api"] | None = None
    whisper_model: str | None = None
    llm_model: str | None = None
    refine: bool = False
    subs: bool = False
    chunk_size: int | None = Field(default=None, ge=1, le=400)
    start: str | None = None
    duration: str | None = None

    @field_validator("start", "duration")
    @classmethod
    def _check_time(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if not _TIME_RE.match(value):
            raise ValueError("must contain only digits, colons, dots, commas, or whitespace")
        return value


@dataclass
class JobRecord:
    id: str
    state: JobState
    inputs: dict
    workspace: str | None = None
    slug: str | None = None
    timestamp: str | None = None
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    progress: float = 0.0
    stage: str | None = None
    message: str | None = None

    def public(self) -> dict:
        return asdict(self)


# Internal per-job state (not persisted).
@dataclass
class _JobState:
    record: JobRecord
    cancel_token: threading.Event = field(default_factory=threading.Event)
    subscribers: list[Queue] = field(default_factory=list)
    log_path: Path = field(default_factory=Path)


class JobManager:
    """In-process job queue with append-only JSONL event logs.

    Thread-safe. One ``threading.RLock`` guards the records dict, the
    subscriber lists, and per-job log writes. Heavy work (the pipeline
    itself) runs without the lock held in ``ThreadPoolExecutor`` workers.
    """

    def __init__(
        self,
        base_dir: Path,
        max_workers: int | None = None,
        retention: int | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.jobs_dir = self.base_dir / ".jobs"
        self.uploads_dir = self.base_dir / ".uploads"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        env_workers = os.environ.get("PGW_SERVE_MAX_JOBS")
        if max_workers is None and env_workers:
            try:
                max_workers = max(1, int(env_workers))
            except ValueError:
                max_workers = None
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers or _DEFAULT_MAX_WORKERS,
            thread_name_prefix="pgw-job",
        )

        env_retention = os.environ.get("PGW_JOBS_RETENTION")
        if retention is None and env_retention:
            try:
                retention = max(1, int(env_retention))
            except ValueError:
                retention = None
        self._retention = retention or _DEFAULT_RETENTION

        self._lock = threading.RLock()
        self._states: dict[str, _JobState] = {}
        # Newest first; finished records stay in memory until the next
        # rebuild from disk so list() is consistent across restart.
        self._reap_orphans()
        self._load_finished()
        self._gc_logs()

    # ── Public API ───────────────────────────────────────────────────

    def submit(self, req: JobRequest) -> str:
        job_id = uuid.uuid4().hex
        record = JobRecord(
            id=job_id,
            state="pending",
            inputs=req.model_dump(),
            created_at=time.time(),
        )
        log_path = self.jobs_dir / f"{job_id}.jsonl"
        state = _JobState(record=record, log_path=log_path)
        with self._lock:
            self._states[job_id] = state
        # Persist the initial record + pending state outside the lock so a
        # slow filesystem doesn't stall request threads.
        self._broadcast(state, {"type": "record", **record.public()})
        self._broadcast(state, {"type": "state", "state": "pending", "ts": time.time()})
        self._executor.submit(self._run_job, job_id)
        return job_id

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            state = self._states.get(job_id)
            if state is None or state.record.state in _TERMINAL_STATES:
                return False
            state.cancel_token.set()
            should_emit = state.record.state == "running"
            if should_emit:
                state.record.state = "cancelling"
        if should_emit:
            self._broadcast(
                state,
                {"type": "state", "state": "cancelling", "ts": time.time()},
            )
        return True

    def snapshot(self, job_id: str) -> JobRecord | None:
        with self._lock:
            state = self._states.get(job_id)
            return None if state is None else JobRecord(**asdict(state.record))

    def list(self) -> list[JobRecord]:
        """Return all jobs, newest first."""
        with self._lock:
            records = [JobRecord(**asdict(s.record)) for s in self._states.values()]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records

    async def stream(self, job_id: str) -> AsyncIterator[bytes]:
        """Yield NDJSON event lines for a job, replay-then-tail.

        Async generator so each open SSE client only ties up an asyncio
        task, not a Starlette threadpool slot. Snapshots the on-disk log
        under the lock, subscribes a fresh queue, then yields:

        1. all log lines that existed at subscribe time
        2. live events broadcast after subscribe
        3. heartbeats every ``_HEARTBEAT_INTERVAL_SEC`` during quiet stretches
        4. ``_END`` from the broadcaster — terminates the stream

        If the broadcaster signals ``_DISCONNECT`` (slow client whose queue
        filled up), the stream closes; the client is expected to reconnect
        and replay-from-log.
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            state = self._states.get(job_id)
            if state is None:
                return
            log_lines = self._read_log_lines(state)
            already_terminal = state.record.state in _TERMINAL_STATES
            q: Queue | None = None
            if not already_terminal:
                q = Queue(maxsize=_QUEUE_MAXSIZE)
                state.subscribers.append(q)

        try:
            for line in log_lines:
                yield line.encode("utf-8") + b"\n"
            if q is None:
                return
            while True:
                try:
                    item = await loop.run_in_executor(None, q.get, True, _HEARTBEAT_INTERVAL_SEC)
                except Empty:
                    # Heartbeat tick — defence-in-depth re-check that we
                    # are still subscribed. If the broadcaster evicted us
                    # but the ``_DISCONNECT`` sentinel never made it into
                    # the queue (extreme edge: queue at capacity at the
                    # moment of put_nowait), we exit here within one
                    # heartbeat interval rather than looping forever.
                    with self._lock:
                        still_subscribed = q in state.subscribers
                    if not still_subscribed:
                        break
                    yield (json.dumps({"type": "heartbeat", "ts": time.time()}) + "\n").encode(
                        "utf-8"
                    )
                    continue
                if item is _END or item is _DISCONNECT:
                    break
                yield item.encode("utf-8") + b"\n"
        finally:
            if q is not None:
                with self._lock:
                    try:
                        state.subscribers.remove(q)
                    except ValueError:
                        pass

    def shutdown(self, wait: bool = False) -> None:
        """Signal cancel for all running jobs and stop the worker pool."""
        with self._lock:
            for state in self._states.values():
                if state.record.state not in _TERMINAL_STATES:
                    state.cancel_token.set()
        self._executor.shutdown(wait=wait, cancel_futures=True)

    # ── Worker ───────────────────────────────────────────────────────

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            state = self._states.get(job_id)
            if state is None:
                return
            if state.cancel_token.is_set():
                state.record.state = "cancelled"
                state.record.finished_at = time.time()
                early_cancel = True
            else:
                early_cancel = False
                state.record.state = "running"
                state.record.started_at = time.time()
        if early_cancel:
            self._terminal_locked(state)
            return
        self._broadcast(
            state,
            {
                "type": "state",
                "state": "running",
                "started_at": state.record.started_at,
                "ts": time.time(),
            },
        )

        try:
            self._invoke_pipeline(state)
            with self._lock:
                state.record.state = "succeeded"
        except JobCancelled:
            with self._lock:
                state.record.state = "cancelled"
        except Exception as exc:  # noqa: BLE001 — surface anything to the client
            tb = traceback.format_exc()
            err_path = self.jobs_dir / f"{state.record.id}.err.txt"
            try:
                err_path.write_text(tb, encoding="utf-8")
            except OSError:
                logger.exception("Failed to persist traceback")
            with self._lock:
                state.record.state = "failed"
                state.record.error = str(exc) or repr(exc)
        finally:
            with self._lock:
                state.record.finished_at = time.time()
                self._terminal_locked(state)
            self._gc_logs()

    def _invoke_pipeline(self, state: _JobState) -> None:
        # Defer heavy imports until a job actually runs.
        from pgw.cli.utils import build_config_overrides
        from pgw.core.config import load_config
        from pgw.core.languages import validate_language
        from pgw.core.pipeline import run_pipeline

        req = state.record.inputs
        validate_language(req["language"])
        if req.get("translate"):
            validate_language(req["translate"])

        overrides = build_config_overrides(
            language=req["language"],
            device="auto",
            whisper_model=req.get("whisper_model"),
            llm_model=req.get("llm_model"),
            llm_backend=req.get("llm_backend"),
            backend=req.get("backend"),
            translate=req.get("translate"),
            subs=req.get("subs", False),
        )
        config = load_config(**overrides)

        on_event = self._make_event_callback(state)

        workspace = run_pipeline(
            input_path=req["input"],
            config=config,
            translate=req.get("translate"),
            refine=bool(req.get("refine", False)),
            play=False,
            start=req.get("start"),
            duration=req.get("duration"),
            on_event=on_event,
            chunk_size=req.get("chunk_size"),
            cancel_token=state.cancel_token,
        )

        # The save stage already emits a workspace event via on_event when
        # the pipeline calls emit("save", 1.0, ..., data={"workspace": ...}).
        # As a safety net, capture the path here too.
        with self._lock:
            if not state.record.workspace:
                state.record.workspace = str(workspace)
                state.record.slug = workspace.parent.name
                state.record.timestamp = workspace.name
                self._fanout_log_locked(
                    state,
                    {
                        "type": "workspace",
                        "workspace": str(workspace),
                        "slug": workspace.parent.name,
                        "timestamp": workspace.name,
                        "ts": time.time(),
                    },
                )

    def _make_event_callback(self, state: _JobState):
        def on_event(event: PipelineEvent) -> None:
            ts = time.time()
            payload: dict[str, Any] = {
                "type": "event",
                "stage": event.stage,
                "progress": event.progress,
                "message": event.message,
                "data": event.data,
                "ts": ts,
            }
            with self._lock:
                state.record.stage = event.stage
                state.record.progress = event.progress
                state.record.message = event.message
                self._fanout_log_locked(state, payload)
                # Detect workspace path being announced.
                ws = (event.data or {}).get("workspace") if event.data else None
                if ws and not state.record.workspace:
                    ws_path = Path(ws)
                    state.record.workspace = ws
                    state.record.slug = ws_path.parent.name
                    state.record.timestamp = ws_path.name
                    self._fanout_log_locked(
                        state,
                        {
                            "type": "workspace",
                            "workspace": ws,
                            "slug": ws_path.parent.name,
                            "timestamp": ws_path.name,
                            "ts": ts,
                        },
                    )

        return on_event

    # ── Log + fan-out ────────────────────────────────────────────────

    def _append_log(self, log_path: Path, line: str) -> None:
        """Append a serialized JSON line to disk. Caller must hold no lock."""
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            logger.exception("Failed to append job log")

    def _broadcast(self, state: _JobState, payload: dict, *, terminal: bool = False) -> None:
        """Serialize, persist, and fan out a single event.

        Splits the work to keep the lock window small: serialize + read
        subscriber list under the lock, write to disk and push to queues
        outside it. Subscribers whose queues are full are *removed* so a
        terminal/EOF sentinel can never be dropped.
        """
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            log_path = state.log_path
            subscribers = list(state.subscribers)
        self._append_log(log_path, line)
        slow_qs: list[Queue] = []
        for q in subscribers:
            try:
                q.put_nowait(line)
            except Full:
                slow_qs.append(q)
        if terminal:
            for q in subscribers:
                if q in slow_qs:
                    continue
                try:
                    q.put_nowait(_END)
                except Full:
                    slow_qs.append(q)
        # Boot any slow clients so they reconnect and replay from the log
        # rather than miss events (or — worse — miss the terminal sentinel).
        # Best-effort wake via _DISCONNECT; if the queue is at capacity at
        # this very moment the streaming generator's heartbeat-tick path
        # will re-check subscriber membership and exit on its own (see
        # ``stream()``). We deliberately do NOT drain the queue here — that
        # would race the streaming generator's pending ``q.get``.
        if slow_qs:
            with self._lock:
                for q in slow_qs:
                    try:
                        state.subscribers.remove(q)
                    except ValueError:
                        pass
            for q in slow_qs:
                try:
                    q.put_nowait(_DISCONNECT)
                except Full:
                    pass

    # Backwards-compatible aliases — call sites use these names.
    def _fanout_log_locked(self, state: _JobState, payload: dict) -> None:
        self._broadcast(state, payload, terminal=False)

    def _fanout_locked(self, state: _JobState, payload: dict) -> None:
        # Persist alongside fan-out so reattach replays the same sequence.
        self._broadcast(state, payload, terminal=False)

    def _terminal_locked(self, state: _JobState) -> None:
        payload = {
            "type": "terminal",
            "state": state.record.state,
            "error": state.record.error,
            "finished_at": state.record.finished_at,
            "ts": time.time(),
        }
        self._broadcast(state, payload, terminal=True)

    def _read_log_lines(self, state: _JobState) -> list[str]:
        if not state.log_path.is_file():
            return []
        try:
            text = state.log_path.read_text(encoding="utf-8")
        except OSError:
            return []
        return [ln for ln in text.splitlines() if ln.strip()]

    # ── Startup housekeeping ─────────────────────────────────────────

    def _reap_orphans(self) -> None:
        """On startup, mark any job whose log lacks a terminal event as interrupted."""
        for log_path in sorted(self.jobs_dir.glob("*.jsonl")):
            try:
                text = log_path.read_text(encoding="utf-8")
            except OSError:
                continue
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not lines:
                continue
            try:
                last = json.loads(lines[-1])
            except json.JSONDecodeError:
                last = {}
            if last.get("type") == "terminal":
                continue
            terminal = {
                "type": "terminal",
                "state": "interrupted",
                "error": "server restarted before job completed",
                "finished_at": time.time(),
                "ts": time.time(),
            }
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(terminal) + "\n")
            except OSError:
                logger.exception("Failed to mark orphan job interrupted")

    def _load_finished(self) -> None:
        """Hydrate finished JobRecord entries from on-disk logs.

        Lets the library page show recent jobs (succeeded/failed/cancelled)
        across server restarts without having to re-read logs on every list().
        """
        for log_path in sorted(self.jobs_dir.glob("*.jsonl")):
            job_id = log_path.stem
            if job_id in self._states:
                continue
            record = self._record_from_log(log_path, job_id)
            if record is None:
                continue
            self._states[job_id] = _JobState(
                record=record,
                log_path=log_path,
            )
            # Pre-set the cancel token so cancel() is a no-op for finished jobs.
            self._states[job_id].cancel_token.set()

    @staticmethod
    def _record_from_log(log_path: Path, job_id: str) -> JobRecord | None:
        try:
            text = log_path.read_text(encoding="utf-8")
        except OSError:
            return None
        record_dict: dict | None = None
        terminal: dict | None = None
        workspace: dict | None = None
        last_event: dict | None = None
        first_state_ts: float | None = None
        running_ts: float | None = None
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            t = obj.get("type")
            if t == "record":
                record_dict = obj
            elif t == "state" and obj.get("state") == "pending" and first_state_ts is None:
                first_state_ts = obj.get("ts")
            elif t == "state" and obj.get("state") == "running":
                running_ts = obj.get("started_at") or obj.get("ts")
            elif t == "workspace":
                workspace = obj
            elif t == "event":
                last_event = obj
            elif t == "terminal":
                terminal = obj
        if record_dict is None:
            return None
        record = JobRecord(
            id=job_id,
            state=record_dict.get("state", "pending"),
            inputs=record_dict.get("inputs", {}),
            created_at=record_dict.get("created_at", 0.0) or first_state_ts or 0.0,
        )
        if running_ts:
            record.started_at = running_ts
        if workspace:
            record.workspace = workspace.get("workspace")
            record.slug = workspace.get("slug")
            record.timestamp = workspace.get("timestamp")
        if last_event:
            record.stage = last_event.get("stage")
            record.progress = float(last_event.get("progress") or 0.0)
            record.message = last_event.get("message")
        if terminal:
            record.state = terminal.get("state", record.state)
            record.error = terminal.get("error")
            record.finished_at = terminal.get("finished_at") or terminal.get("ts")
        return record

    def _gc_logs(self) -> None:
        """Trim the .jobs/ directory to ``self._retention`` newest terminals."""
        files = sorted(
            self.jobs_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        )
        kept = 0
        for f in files:
            job_id = f.stem
            with self._lock:
                state = self._states.get(job_id)
            is_terminal = state is None or state.record.state in _TERMINAL_STATES
            if not is_terminal:
                continue
            kept += 1
            if kept <= self._retention:
                continue
            with self._lock:
                self._states.pop(job_id, None)
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass
            err = self.jobs_dir / f"{job_id}.err.txt"
            try:
                err.unlink(missing_ok=True)
            except OSError:
                pass
