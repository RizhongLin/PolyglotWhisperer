"""Process-local registry of live worker WebSocket connections.

Bridges the async WebSocket loop (FastAPI/Starlette) with the sync
``JobManager`` running on a ``ThreadPoolExecutor``.

Lifecycle:
- WS handler calls ``register(user_id, handle)`` after the ``ready``
  frame is parsed.
- ``JobManager.submit()`` consults ``is_connected(user_id)`` to decide
  whether to dispatch to a worker or run in-process.
- ``JobManager`` calls ``send_threadsafe(user_id, frame)`` to fire a
  ``JobAssignFrame`` or ``JobCancelFrame``; the registry hands the
  coroutine to the captured event loop.
- WS handler calls ``unregister(user_id)`` on disconnect, which also
  notifies ``JobManager`` so any in-flight jobs land in ``interrupted``.

Only one worker per user for v1 — a second connection from the same
user displaces the first. Multi-worker routing (with tags) is P7+.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class WorkerHandle:
    """Reference to one connected worker.

    ``send`` is a coroutine — the registry schedules it on the captured
    event loop via ``run_coroutine_threadsafe`` so sync code (the
    ``JobManager`` thread) can fire frames safely.
    """

    user_id: int
    ws: "WebSocket"
    loop: asyncio.AbstractEventLoop
    on_disconnect: Callable[[int], None] = field(default=lambda _u: None)
    in_flight_jobs: set[str] = field(default_factory=set)


class WorkerRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._workers: dict[int, WorkerHandle] = {}
        self._on_disconnect_cb: Callable[[int, set[str]], None] | None = None

    def set_disconnect_callback(self, cb: Callable[[int, set[str]], None]) -> None:
        """Hook used by ``JobManager`` to mark in-flight jobs interrupted."""
        self._on_disconnect_cb = cb

    def register(self, handle: WorkerHandle) -> None:
        with self._lock:
            previous = self._workers.get(handle.user_id)
            if previous is not None:
                # Single-worker-per-user rule for v1: tear the old one
                # down so its in-flight jobs become interrupted before
                # the new connection starts dispatching.
                self._notify_disconnect_locked(previous)
            self._workers[handle.user_id] = handle
            logger.info("worker registered: user=%s", handle.user_id)

    def unregister(self, user_id: int, expected: WorkerHandle | None = None) -> None:
        """Remove ``user_id``'s handle.

        When ``expected`` is provided, only remove if the currently
        registered handle is the same object — this prevents a stale
        ``finally`` block from evicting a freshly reconnected worker
        that displaced the old one via ``register()``.
        """
        with self._lock:
            current = self._workers.get(user_id)
            if current is None:
                return
            if expected is not None and current is not expected:
                # A newer connection has taken over; nothing to clean up.
                return
            self._workers.pop(user_id, None)
            self._notify_disconnect_locked(current)
            logger.info("worker unregistered: user=%s", user_id)

    def _notify_disconnect_locked(self, handle: WorkerHandle) -> None:
        if self._on_disconnect_cb is not None:
            try:
                self._on_disconnect_cb(handle.user_id, set(handle.in_flight_jobs))
            except Exception:  # noqa: BLE001
                logger.exception("disconnect callback failed")

    def is_connected(self, user_id: int) -> bool:
        with self._lock:
            return user_id in self._workers

    def get(self, user_id: int) -> WorkerHandle | None:
        with self._lock:
            return self._workers.get(user_id)

    def track_job(self, user_id: int, job_id: str) -> None:
        with self._lock:
            handle = self._workers.get(user_id)
            if handle is not None:
                handle.in_flight_jobs.add(job_id)

    def untrack_job(self, user_id: int, job_id: str) -> None:
        with self._lock:
            handle = self._workers.get(user_id)
            if handle is not None:
                handle.in_flight_jobs.discard(job_id)

    # Short timeout so a stalled WS does not block a JobManager worker
    # for long. With ``PGW_SERVE_MAX_JOBS=1`` (default) the pool has a
    # single thread, so any longer wait stalls all dispatch.
    SEND_TIMEOUT_SECONDS = 1.0

    def send_threadsafe(self, user_id: int, payload: dict[str, Any]) -> bool:
        """Fire a JSON frame to ``user_id``'s worker. Sync-safe."""
        handle = self.get(user_id)
        if handle is None:
            return False

        async def _send() -> None:
            try:
                await handle.ws.send_json(payload)
            except Exception:  # noqa: BLE001
                logger.exception("send_threadsafe failed for user %s", user_id)

        future = asyncio.run_coroutine_threadsafe(_send(), handle.loop)
        try:
            future.result(timeout=self.SEND_TIMEOUT_SECONDS)
            return True
        except Exception:  # noqa: BLE001
            logger.warning(
                "send_threadsafe timed out for user %s after %.1fs — dropping frame",
                user_id,
                self.SEND_TIMEOUT_SECONDS,
            )
            future.cancel()
            return False


# Process-global registry, owned by the FastAPI app.
GLOBAL_WORKERS = WorkerRegistry()
