"""Per-job execution context.

Carries credentials and identity through the pipeline via a
``contextvars.ContextVar`` so concurrent jobs in the same process
cannot read each other's state.

Today (P1): the context is set by ``JobManager._run_job`` for every
worker-thread pipeline invocation. The CLI path leaves it ``None`` —
``load_config()`` keeps reading from ``os.environ`` exactly as before.

Later phases populate ``env_overrides`` with per-user credentials
fetched from the DB (P2+) or from the WebSocket worker handshake (P3+).
"""

from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

#: Thread name prefix used by ``JobManager`` for pipeline workers.
#: ``load_config()`` checks against this prefix to enforce that any
#: thread running a pipeline has an explicit ``JobContext``.
WORKER_THREAD_PREFIX = "pgw-job"


@dataclass(frozen=True)
class JobContext:
    """Per-job, per-thread state passed into ``load_config(context=...)``.

    Frozen so it is safe to share across threads / capture by reference.
    Add fields additively only — never remove fields, since older
    pipelines may still reference them.
    """

    #: User who owns the job. ``None`` for CLI / unauthenticated.
    user_id: int | None = None
    #: Job ID, mostly for logging correlation.
    job_id: str | None = None
    #: Per-user environment-variable overrides applied to ``load_config``.
    #: Wins over real ``os.environ`` (this is how per-user API keys reach
    #: the pipeline without poisoning the global process env).
    env_overrides: dict[str, str] = field(default_factory=dict)


_current: contextvars.ContextVar[JobContext | None] = contextvars.ContextVar(
    "pgw_job_context",
    default=None,
)


def get_context() -> JobContext | None:
    """Return the active ``JobContext`` for this task, or ``None``."""
    return _current.get()


def is_worker_thread() -> bool:
    """True if the current thread was spawned by ``JobManager``."""
    return threading.current_thread().name.startswith(WORKER_THREAD_PREFIX)


@contextmanager
def use_context(ctx: JobContext) -> Iterator[JobContext]:
    """Bind ``ctx`` for the duration of the ``with`` block.

    ``ContextVar`` is thread-safe and ``ThreadPoolExecutor.submit``
    captures the contextvars on submit, so a worker thread running this
    block sees the right context even when the main thread mutates its
    own.
    """
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)
