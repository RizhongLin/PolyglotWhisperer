"""Tests for ``pgw.core.context`` and the worker-thread guard.

The guard is the load-bearing safety check for multi-user pgw: any
pipeline running in a worker thread must carry a ``JobContext`` so that
``load_config`` cannot accidentally read another user's credentials
from the global process env.
"""

from __future__ import annotations

import os
import threading

from pgw.core.config import load_config
from pgw.core.context import (
    WORKER_THREAD_PREFIX,
    JobContext,
    get_context,
    is_worker_thread,
    use_context,
)


def test_default_context_is_none() -> None:
    assert get_context() is None


def test_use_context_sets_and_resets() -> None:
    ctx = JobContext(user_id=42, job_id="abc")
    with use_context(ctx):
        assert get_context() is ctx
    assert get_context() is None


def test_is_worker_thread_main_thread_is_false() -> None:
    assert is_worker_thread() is False


def test_is_worker_thread_detects_pgw_job_prefix() -> None:
    seen: dict[str, bool] = {}

    def _check() -> None:
        seen["worker"] = is_worker_thread()

    t = threading.Thread(target=_check, name=f"{WORKER_THREAD_PREFIX}-test-1")
    t.start()
    t.join()
    assert seen["worker"] is True


def test_load_config_in_worker_thread_without_context_raises() -> None:
    """The guard is the whole point — verify it actually fires."""
    err: dict[str, BaseException] = {}

    def _run() -> None:
        try:
            load_config()
        except RuntimeError as exc:
            err["e"] = exc

    t = threading.Thread(target=_run, name=f"{WORKER_THREAD_PREFIX}-guarded-1")
    t.start()
    t.join()
    assert "e" in err
    assert "JobContext" in str(err["e"])


def test_load_config_in_worker_thread_with_context_succeeds() -> None:
    ok: dict[str, bool] = {}

    def _run() -> None:
        ctx = JobContext(user_id=1, job_id="x")
        load_config(context=ctx)
        ok["done"] = True

    t = threading.Thread(target=_run, name=f"{WORKER_THREAD_PREFIX}-ok-1")
    t.start()
    t.join()
    assert ok.get("done") is True


def test_load_config_main_thread_no_context_still_works() -> None:
    """CLI path must remain byte-identical: no context required."""
    cfg = load_config()
    assert cfg is not None


def test_env_overrides_reach_config_then_env_restored() -> None:
    """``env_overrides`` is the per-user-key injection point for P2+.

    Contract:
      - During ``load_config``, the override wins over ``os.environ``.
      - After ``load_config`` returns, ``os.environ`` is the same as
        before the call — we never leave the process env in a per-user
        state.
    """
    key = "PGW_LLM__API_KEY"
    original = os.environ.get(key)
    os.environ[key] = "outer-secret"
    try:
        ctx = JobContext(env_overrides={key: "inner-secret"})
        captured: dict[str, str | None] = {}

        def _run() -> None:
            with use_context(ctx):
                cfg = load_config(context=ctx)
                captured["api_key_in_config"] = cfg.llm.api_key
                captured["env_during"] = os.environ.get(key)

        t = threading.Thread(target=_run, name=f"{WORKER_THREAD_PREFIX}-env-1")
        t.start()
        t.join()
        # The config object was built from the override — that's the
        # whole point.
        assert captured["api_key_in_config"] == "inner-secret"
        # The override must NOT linger in os.environ after load_config.
        assert os.environ.get(key) == "outer-secret"
    finally:
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original
