"""Stable error codes consumed by the SPA + the worker agent.

Two registries:

- ``WSClose``: numeric WebSocket close codes (RFC 6455 §7.4 — codes
  4000-4999 are application-defined). Used by ``server/routes/workers.py``
  and the worker agent for handshake errors.
- ``Err``: stable string codes attached to HTTP error envelopes. The
  SPA branches on these instead of free-form ``detail`` text. Add new
  codes freely; never rename or remove an existing one without bumping
  a version note in ``docs/plans/v2-architecture.md``.

Why both: HTTP status codes (401/403/404/...) cover *category* but not
*reason*. ``Err.AUTH_INVALID_CREDENTIALS`` vs ``Err.AUTH_NOT_AUTHENTICATED``
are both 401 but mean different things to the SPA.

The full JSON error-envelope middleware that uses ``Err`` lands in P9
hardening; until then routes raise ``HTTPException`` with the code in
the ``detail`` JSON ``{"code": Err.X, "message": "..."}``.
"""

from __future__ import annotations

from typing import Final


class WSClose:
    """RFC 6455 §7.4 application close codes used by ``/ws/worker``."""

    #: Frame failed Pydantic validation or had wrong shape.
    INVALID_FRAME: Final = 4400
    #: Token missing/invalid/revoked.
    UNAUTHORIZED: Final = 4401
    #: Worker didn't send a ``ready`` frame in time.
    READY_TIMEOUT: Final = 4408
    #: Client/server protocol_version mismatch.
    PROTOCOL_VERSION_MISMATCH: Final = 4426


class Err:
    """Stable string codes for HTTP error responses.

    Naming: ``<area>.<reason>`` lower_snake. The SPA imports nothing
    from the backend so these strings are duplicated client-side; treat
    them like a wire format and keep them stable.
    """

    # ── Auth / sessions ──
    AUTH_INVALID_CREDENTIALS: Final = "auth.invalid_credentials"
    AUTH_NOT_AUTHENTICATED: Final = "auth.not_authenticated"
    AUTH_ADMIN_REQUIRED: Final = "auth.admin_required"
    AUTH_INVALID_EMAIL: Final = "auth.invalid_email"

    # ── CSRF ──
    CSRF_MISSING: Final = "csrf.missing"
    CSRF_MISMATCH: Final = "csrf.mismatch"
    CSRF_INVALID_SIGNATURE: Final = "csrf.invalid_signature"

    # ── Setup / bootstrap ──
    SETUP_ALREADY_COMPLETE: Final = "setup.already_complete"

    # ── Workers ──
    WORKER_NOT_FOUND: Final = "worker.not_found"
    WORKER_NO_CONNECTED: Final = "worker.no_connected_worker"

    # ── Jobs ──
    JOB_NOT_FOUND: Final = "job.not_found"
    JOB_INPUT_REJECTED: Final = "job.input_rejected"


def envelope(code: str, message: str) -> dict[str, str]:
    """Standard ``HTTPException(detail=...)`` payload.

    Yields ``{"code": "auth.invalid_credentials", "message": "..."}``
    so the SPA can switch on ``code`` and surface ``message`` verbatim.
    """
    return {"code": code, "message": message}
