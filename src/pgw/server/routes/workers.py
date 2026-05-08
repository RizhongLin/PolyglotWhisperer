"""Worker token CRUD + the WebSocket upgrade handler.

REST surface (cookie-authed):
- ``POST /api/workers`` — mint a token. Raw value returned once.
- ``GET /api/workers`` — list user's tokens with live-connect status.
- ``DELETE /api/workers/{id}`` — revoke.

WebSocket upgrade:
- ``WS /ws/worker?token=<raw>`` — token in query string (the ws spec
  doesn't allow custom headers from browsers; the worker is a CLI but
  we keep the shape uniform). Server resolves the token, opens a
  ``WorkerSession`` row, runs the hello/ready handshake, then idles
  with periodic pings until the client disconnects.

P3 ships the spine. Job dispatch over this channel lands in a follow-up
slice that also forks ``JobManager.submit`` to route worker-targeted
jobs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session as SqlaSession

from pgw.auth.csrf import verify_csrf
from pgw.auth.deps import current_user
from pgw.db.models.user import User
from pgw.db.models.worker import WorkerSession, WorkerToken
from pgw.db.session import SessionLocal, get_session
from pgw.errors import Err, WSClose, envelope
from pgw.server.worker_registry import GLOBAL_WORKERS, WorkerHandle
from pgw.worker import tokens as worker_tokens
from pgw.worker.protocol import (
    PROTOCOL_VERSION,
    HelloFrame,
    JobEventFrame,
    JobTerminalFrame,
    JobWorkspaceFrame,
    ReadyFrame,
    parse_frame,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workers", tags=["workers"])
ws_router = APIRouter()


# ── REST ────────────────────────────────────────────────────────────────


class CreateWorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=128)


class WorkerSummary(BaseModel):
    id: int
    name: str
    created_at: datetime
    revoked_at: datetime | None
    last_seen_at: datetime | None
    connected: bool


class CreateWorkerResponse(BaseModel):
    id: int
    name: str
    token: str  # one-time raw value; never persisted in this form


@router.post(
    "",
    response_model=CreateWorkerResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_csrf)],
)
def create_worker(
    payload: CreateWorkerRequest,
    user: User = Depends(current_user),
    db: SqlaSession = Depends(get_session),
) -> CreateWorkerResponse:
    row, raw = worker_tokens.issue(db, user, name=payload.name)
    return CreateWorkerResponse(id=row.id, name=row.name, token=raw)


@router.get("", response_model=list[WorkerSummary])
def list_workers(
    user: User = Depends(current_user),
    db: SqlaSession = Depends(get_session),
) -> list[WorkerSummary]:
    """List the user's tokens with live-connect status.

    Two queries total (not N+1): one for tokens, one for the latest
    session row of each token resolved server-side via a single grouped
    query. We pick the latest by ``connected_at`` and join in Python —
    the row count is bounded by token count, which is small.
    """
    tokens = list(
        db.scalars(
            select(WorkerToken)
            .where(WorkerToken.user_id == user.id)
            .order_by(WorkerToken.created_at.desc())
        )
    )
    if not tokens:
        return []

    token_ids = [t.id for t in tokens]
    sessions = list(
        db.scalars(
            select(WorkerSession)
            .where(WorkerSession.token_id.in_(token_ids))
            .order_by(WorkerSession.token_id, WorkerSession.connected_at.desc())
        )
    )
    latest_by_token: dict[int, WorkerSession] = {}
    for s in sessions:
        # First row per token wins (we ordered DESC by connected_at).
        latest_by_token.setdefault(s.token_id, s)

    summaries: list[WorkerSummary] = []
    for r in tokens:
        latest = latest_by_token.get(r.id)
        connected = bool(
            latest is not None and latest.disconnected_at is None and r.revoked_at is None
        )
        summaries.append(
            WorkerSummary(
                id=r.id,
                name=r.name,
                created_at=r.created_at,
                revoked_at=r.revoked_at,
                last_seen_at=latest.last_seen_at if latest else None,
                connected=connected,
            )
        )
    return summaries


@router.delete(
    "/{token_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_csrf)],
)
def revoke_worker(
    token_id: int,
    user: User = Depends(current_user),
    db: SqlaSession = Depends(get_session),
) -> None:
    if not worker_tokens.revoke(db, user=user, token_id=token_id):
        raise HTTPException(
            status_code=404,
            detail=envelope(Err.WORKER_NOT_FOUND, "worker token not found"),
        )


# ── WebSocket ───────────────────────────────────────────────────────────


_HEARTBEAT_INTERVAL = 20.0  # seconds between server-sent pings on idle


def _resolve_ws_token(ws: WebSocket, query_token: str | None) -> tuple[str | None, bool]:
    """Pick the worker token from ``Authorization: Bearer <t>`` first.

    Returns ``(token, came_from_query_string)``. The query-string
    fallback exists for browser-based workers (custom headers aren't
    available on the WebSocket upgrade in browsers). The CLI worker
    should always prefer the header so the token does not land in
    proxy / uvicorn access logs.
    """
    auth = ws.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        candidate = auth[len("bearer ") :].strip()
        if candidate:
            return candidate, False
    if query_token:
        return query_token, True
    return None, False


@ws_router.websocket("/ws/worker")
async def worker_ws(
    ws: WebSocket,
    token: str | None = Query(default=None, max_length=128),
) -> None:
    """Long-lived WS for a connected worker.

    Handshake → registers in ``GLOBAL_WORKERS`` → relays inbound
    ``job.event`` / ``job.terminal`` / ``job.workspace`` frames into
    the ``JobManager`` so the SPA's NDJSON stream sees worker-originated
    events with the same shape as in-process events.
    """
    raw, from_query = _resolve_ws_token(ws, token)
    if not raw:
        await ws.close(code=WSClose.UNAUTHORIZED, reason="missing token")
        return
    if from_query:
        # Tokens in the query string land in proxy / uvicorn access logs.
        # Browser workers may have to use this path; CLI workers should
        # always use Authorization. Loud warning so operators notice.
        logger.warning(
            "worker token supplied via query string (logged by proxies); "
            "prefer Authorization: Bearer for CLI workers"
        )

    # Resolve token before accepting so unknown clients get a clean reject.
    with SessionLocal() as db:
        row = worker_tokens.lookup(db, raw)
        if row is None:
            await ws.close(code=WSClose.UNAUTHORIZED, reason="invalid token")
            return
        token_id = row.id
        user_id = row.user_id

    await ws.accept()
    handle: WorkerHandle | None = None

    # Open a WorkerSession row to mark this connection live.
    with SessionLocal() as db:
        sess = WorkerSession(token_id=token_id, capabilities={})
        db.add(sess)
        db.commit()
        db.refresh(sess)
        session_id = sess.id

    try:
        await ws.send_json(
            HelloFrame(server_time=time.time(), protocol_version=PROTOCOL_VERSION).model_dump()
        )

        # Wait for the worker's ready frame (with sane timeout).
        try:
            ready_payload: dict[str, Any] = await asyncio.wait_for(ws.receive_json(), timeout=10.0)
        except asyncio.TimeoutError:
            await ws.close(code=WSClose.READY_TIMEOUT, reason="ready timeout")
            return

        try:
            ready = parse_frame(ready_payload)
        except ValueError as exc:
            await ws.close(code=WSClose.INVALID_FRAME, reason=str(exc))
            return
        if not isinstance(ready, ReadyFrame):
            await ws.close(code=WSClose.INVALID_FRAME, reason="expected ready frame")
            return

        if ready.protocol_version != PROTOCOL_VERSION:
            # Strict equality for now; loosen with min/max ranges later.
            await ws.close(
                code=WSClose.PROTOCOL_VERSION_MISMATCH,
                reason="protocol version mismatch",
            )
            return

        with SessionLocal() as db:
            db_sess = db.get(WorkerSession, session_id)
            if db_sess is not None:
                db_sess.hostname = ready.hostname
                db_sess.pgw_version = ready.pgw_version
                db_sess.capabilities = ready.capabilities
                db.commit()

        # Register the live worker so JobManager can dispatch to it.
        loop = asyncio.get_running_loop()
        handle = WorkerHandle(user_id=user_id, ws=ws, loop=loop)
        GLOBAL_WORKERS.register(handle)

        # Lazy import — avoid top-level circular with server.app.
        from pgw.server.app import get_job_manager

        manager = get_job_manager()

        # Idle loop: relay frames + heartbeat.
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=_HEARTBEAT_INTERVAL)
            except asyncio.TimeoutError:
                # Idle — emit a server ping to keep proxies happy.
                await ws.send_json({"type": "ping"})
                continue
            try:
                frame = parse_frame(msg)
            except ValueError:
                # Tolerate unknowns until full protocol lands.
                continue
            if frame.type == "pong":
                with SessionLocal() as db:
                    db_sess = db.get(WorkerSession, session_id)
                    if db_sess is not None:
                        db_sess.last_seen_at = datetime.now(timezone.utc)
                        db.commit()
            elif isinstance(frame, JobEventFrame) and manager is not None:
                manager.handle_remote_event(
                    frame.job_id,
                    {
                        "stage": frame.stage,
                        "progress": frame.progress,
                        "message": frame.message,
                        "data": frame.data,
                    },
                )
            elif isinstance(frame, JobTerminalFrame) and manager is not None:
                manager.handle_remote_terminal(
                    frame.job_id,
                    terminal_state=frame.state,
                    error=frame.error,
                )
            elif isinstance(frame, JobWorkspaceFrame) and manager is not None:
                manager.handle_remote_workspace(
                    frame.job_id,
                    slug=frame.slug,
                    timestamp=frame.timestamp,
                    fs_path=frame.fs_path,
                )
            # Other frame types reserved for future phases.

    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        logger.exception("worker WS crashed (session %s)", session_id)
    finally:
        if handle is not None:
            GLOBAL_WORKERS.unregister(user_id, expected=handle)
        with SessionLocal() as db:
            db_sess = db.get(WorkerSession, session_id)
            if db_sess is not None and db_sess.disconnected_at is None:
                db_sess.disconnected_at = datetime.now(timezone.utc)
                db.commit()
