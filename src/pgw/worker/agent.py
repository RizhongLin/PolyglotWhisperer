"""Worker agent: long-lived WebSocket loop against a remote pgw server.

Connect → handshake → on ``job.assign`` run ``run_pipeline()`` in a
background thread, streaming ``job.event`` frames back and uploading
artifacts at the end with a ``job.terminal`` to close the job.
"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import threading
import time
from pathlib import Path
from typing import Any

from pgw.worker.protocol import (
    PROTOCOL_VERSION,
    HelloFrame,
    JobAcceptedFrame,
    JobAssignFrame,
    JobCancelFrame,
    JobEventFrame,
    JobTerminalFrame,
    PongFrame,
    ReadyFrame,
    parse_frame,
)

logger = logging.getLogger(__name__)


def _ws_url(server: str) -> str:
    """Translate ``https://server`` → ``wss://server/ws/worker``.

    The token travels in the ``Authorization`` header — keeping it out
    of the URL means it never lands in proxy / uvicorn access logs.
    """
    if server.startswith("https://"):
        base = "wss://" + server[len("https://") :]
    elif server.startswith("http://"):
        base = "ws://" + server[len("http://") :]
    else:
        # Assume bare host means insecure local dev.
        base = "ws://" + server
    base = base.rstrip("/")
    return f"{base}/ws/worker"


def _capabilities() -> dict[str, Any]:
    """Best-effort summary of what this worker can run."""
    caps: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import importlib.util

        caps["whisper_local"] = importlib.util.find_spec("stable_ts") is not None
        caps["yt_dlp"] = importlib.util.find_spec("yt_dlp") is not None
    except Exception:  # noqa: BLE001
        pass
    return caps


def _pgw_version() -> str:
    try:
        from importlib.metadata import version

        return version("pgw")
    except Exception:  # noqa: BLE001
        return "0.0.0"


async def run(server: str, token: str, *, hostname: str | None = None) -> None:
    """Connect once, handshake, idle until the server closes the socket."""
    import websockets

    url = _ws_url(server)
    try:
        ws = await websockets.connect(
            url,
            ping_interval=None,
            max_size=None,
            extra_headers={"Authorization": f"Bearer {token}"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("worker connect failed: %s", exc)
        return

    async with ws:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.error("server did not send hello within 10s")
            return

        try:
            hello = parse_frame(json.loads(raw))
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("invalid hello: %s", exc)
            return
        if not isinstance(hello, HelloFrame):
            logger.error("expected hello frame, got %s", type(hello).__name__)
            return
        if hello.protocol_version != PROTOCOL_VERSION:
            logger.error(
                "protocol version mismatch: server=%s worker=%s",
                hello.protocol_version,
                PROTOCOL_VERSION,
            )
            return

        ready = ReadyFrame(
            hostname=hostname or platform.node(),
            pgw_version=_pgw_version(),
            capabilities=_capabilities(),
        )
        await ws.send(ready.model_dump_json())
        logger.info("worker connected, server_time=%s", hello.server_time)

        # Track in-flight jobs so we can route ``job.cancel`` frames.
        cancel_tokens: dict[str, threading.Event] = {}
        loop = asyncio.get_running_loop()

        last_pong = time.time()
        async for message in ws:
            if not isinstance(message, str):
                continue
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue
            try:
                frame = parse_frame(payload)
            except ValueError:
                # Forward-compat: tolerate frame types we don't know yet.
                continue
            if frame.type == "ping":
                await ws.send(PongFrame().model_dump_json())
                last_pong = time.time()
            elif isinstance(frame, JobAssignFrame):
                await ws.send(JobAcceptedFrame(job_id=frame.job_id).model_dump_json())
                cancel_token = threading.Event()
                cancel_tokens[frame.job_id] = cancel_token
                # Hand the spec to a worker thread; pipeline events
                # flow back into the WS via ``run_coroutine_threadsafe``.
                loop.create_task(
                    _run_assigned_job(
                        ws=ws,
                        loop=loop,
                        server=server,
                        token=token,
                        spec=frame.spec,
                        job_id=frame.job_id,
                        cancel_token=cancel_token,
                    )
                )
            elif isinstance(frame, JobCancelFrame):
                token = cancel_tokens.get(frame.job_id)
                if token is not None:
                    token.set()
        # Disconnect — cancel any in-flight jobs
        for ct in cancel_tokens.values():
            ct.set()
        logger.info("worker disconnected (last activity %.1fs ago)", time.time() - last_pong)


async def _run_assigned_job(
    *,
    ws: Any,
    loop: asyncio.AbstractEventLoop,
    server: str,
    token: str,
    spec: dict,
    job_id: str,
    cancel_token: threading.Event,
) -> None:
    """Drive a single job: pipeline → events → artifacts → terminal."""
    from pgw.cli.utils import build_config_overrides
    from pgw.core.config import load_config
    from pgw.core.events import PipelineEvent
    from pgw.core.languages import validate_language
    from pgw.core.pipeline import run_pipeline
    from pgw.server.exceptions import JobCancelled

    def _send(frame_dict: dict[str, Any]) -> None:
        """Thread-safe send from the pipeline worker thread."""
        asyncio.run_coroutine_threadsafe(ws.send(json.dumps(frame_dict)), loop)

    def _on_event(event: PipelineEvent) -> None:
        _send(
            JobEventFrame(
                job_id=job_id,
                stage=event.stage,
                progress=event.progress,
                message=event.message,
                data=event.data,
            ).model_dump()
        )

    def _do_run() -> None:
        try:
            validate_language(spec["language"])
            if spec.get("translate"):
                validate_language(spec["translate"])
            overrides = build_config_overrides(
                language=spec["language"],
                device="auto",
                whisper_model=spec.get("whisper_model"),
                llm_model=spec.get("llm_model"),
                llm_backend=spec.get("llm_backend"),
                backend=spec.get("backend"),
                translate=spec.get("translate"),
                subs=spec.get("subs", False),
            )
            # Apply per-user credential env overrides from the server
            from pgw.core.context import JobContext, use_context

            ctx = JobContext(
                user_id=None,
                job_id=job_id,
                env_overrides=spec.get("env_overrides") or {},
            )
            with use_context(ctx):
                config = load_config(context=ctx, **overrides)

            workspace = run_pipeline(
                input_path=spec["input"],
                config=config,
                translate=spec.get("translate"),
                refine=bool(spec.get("refine", False)),
                play=False,
                start=spec.get("start"),
                duration=spec.get("duration"),
                on_event=_on_event,
                chunk_size=spec.get("chunk_size"),
                cancel_token=cancel_token,
            )
            # Tell the server where the workspace landed locally so it
            # can show "video on worker" affordances.
            _send(
                {
                    "type": "job.workspace",
                    "job_id": job_id,
                    "slug": workspace.parent.name,
                    "timestamp": workspace.name,
                    "fs_path": str(workspace),
                }
            )
            _upload_artifacts(server=server, token=token, workspace=workspace, job_id=job_id)
            _send(JobTerminalFrame(job_id=job_id, state="succeeded").model_dump())
        except JobCancelled:
            _send(JobTerminalFrame(job_id=job_id, state="cancelled").model_dump())
        except Exception as exc:  # noqa: BLE001
            logger.exception("worker job %s crashed", job_id)
            _send(
                JobTerminalFrame(
                    job_id=job_id, state="failed", error=str(exc) or repr(exc)
                ).model_dump()
            )

    threading.Thread(target=_do_run, name=f"pgw-worker-{job_id[:8]}", daemon=True).start()


def _upload_artifacts(*, server: str, token: str, workspace: Path, job_id: str) -> None:
    """Push small artifacts (VTTs, JSON, audio.ext) to the server.

    Video files stay on the worker by default — the player loads them
    from the worker's loopback host server (P7) when available.
    """
    import urllib.parse
    import urllib.request

    base = server.rstrip("/")
    slug = workspace.parent.name
    ts = workspace.name
    upload_patterns = (
        "*.vtt",
        "*.txt",
        "vocabulary.*.json",
        "metadata.json",
        "word_timestamps.json",
        "audio.*",
    )
    for pattern in upload_patterns:
        for path in sorted(workspace.glob(pattern)):
            if not path.is_file() or path.suffix.lower() in {".mp4", ".mkv", ".webm"}:
                continue
            try:
                payload = path.read_bytes()
            except OSError:
                continue
            qs = urllib.parse.urlencode({"slug": slug, "timestamp": ts, "name": path.name})
            url = f"{base}/api/jobs/{job_id}/artifacts?{qs}"
            req = urllib.request.Request(
                url,
                data=payload,
                method="POST",
                headers={"Authorization": f"Bearer {token}"},
            )
            try:
                urllib.request.urlopen(req, timeout=60).close()  # noqa: S310
            except Exception:  # noqa: BLE001
                logger.exception("artifact upload failed: %s", path.name)
