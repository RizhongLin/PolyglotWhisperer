"""Wire protocol shared by ``pgw worker`` and the server's ``/ws/worker``.

Frame format: one JSON object per WS text frame, discriminated by a
``type`` field. Both directions speak the same set so client and server
share parsing code via Pydantic.

Frames in this file are P3 / handshake. Job-dispatch and
artifact-upload frames land alongside the JobManager fork in a later
slice.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

#: Bumped whenever the wire format changes incompatibly. Server rejects
#: workers with a different major; clients should refuse to talk to a
#: server with a higher version than they understand.
PROTOCOL_VERSION = 1


class _Frame(BaseModel):
    """Base for all framed messages. Refuse unknown keys."""

    model_config = ConfigDict(extra="forbid")


# ── Server → Worker ──────────────────────────────────────────────────────


class HelloFrame(_Frame):
    """Sent by the server immediately after accepting the WS upgrade."""

    type: Literal["hello"] = "hello"
    server_time: float
    protocol_version: int = PROTOCOL_VERSION


class JobAssignFrame(_Frame):
    """Server hands a job spec to the worker."""

    type: Literal["job.assign"] = "job.assign"
    job_id: str
    spec: dict[str, Any]


class JobCancelFrame(_Frame):
    """Server asks the worker to cancel a running job."""

    type: Literal["job.cancel"] = "job.cancel"
    job_id: str


class PingFrame(_Frame):
    type: Literal["ping"] = "ping"


# ── Worker → Server ──────────────────────────────────────────────────────


class ReadyFrame(_Frame):
    """Worker advertises its capabilities once the handshake completes."""

    type: Literal["ready"] = "ready"
    hostname: str
    pgw_version: str
    capabilities: dict[str, Any] = Field(default_factory=dict)
    protocol_version: int = PROTOCOL_VERSION


class JobAcceptedFrame(_Frame):
    type: Literal["job.accepted"] = "job.accepted"
    job_id: str


class JobEventFrame(_Frame):
    """Mirrors ``PipelineEvent``; server fans out into NDJSON log."""

    type: Literal["job.event"] = "job.event"
    job_id: str
    stage: str
    progress: float
    message: str
    data: dict[str, Any] | None = None


class JobTerminalFrame(_Frame):
    type: Literal["job.terminal"] = "job.terminal"
    job_id: str
    state: Literal["succeeded", "failed", "cancelled", "interrupted"]
    error: str | None = None


class PongFrame(_Frame):
    type: Literal["pong"] = "pong"


class JobWorkspaceFrame(_Frame):
    type: Literal["job.workspace"] = "job.workspace"
    job_id: str  # type: ignore[assignment]  # noqa: F811
    slug: str
    timestamp: str
    fs_path: str


# ── Convenience parser ──────────────────────────────────────────────────

_FRAMES: dict[str, type[_Frame]] = {
    "hello": HelloFrame,
    "job.assign": JobAssignFrame,
    "job.cancel": JobCancelFrame,
    "job.workspace": JobWorkspaceFrame,
    "ping": PingFrame,
    "ready": ReadyFrame,
    "job.accepted": JobAcceptedFrame,
    "job.event": JobEventFrame,
    "job.terminal": JobTerminalFrame,
    "pong": PongFrame,
}


def parse_frame(payload: dict[str, Any]) -> _Frame:
    """Discriminate on ``type``; raise ValueError for unknown frames."""
    kind = payload.get("type")
    cls = _FRAMES.get(kind)  # type: ignore[arg-type]
    if cls is None:
        raise ValueError(f"unknown frame type: {kind!r}")
    return cls.model_validate(payload)
