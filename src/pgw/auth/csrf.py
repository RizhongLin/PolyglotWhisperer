"""Signed double-submit-cookie CSRF.

The SPA reads the ``pgw_csrf`` cookie on bootstrap and echoes it as
``X-CSRF-Token`` on every state-changing request. Server verifies the
signature (via ``itsdangerous``) and asserts the cookie/header match.

Why double-submit instead of synchronizer-token: it's stateless, plays
well with horizontal scaling, and the SameSite=Lax session cookie
already blocks the easiest CSRF vectors.
"""

from __future__ import annotations

import os
import secrets

from fastapi import Request
from itsdangerous import BadSignature, URLSafeSerializer

CSRF_COOKIE = "pgw_csrf"
CSRF_HEADER = "X-CSRF-Token"


def _secret() -> str:
    """Resolve the CSRF signing secret.

    Production sets ``PGW_SECRET_KEY``. For dev we fall back to a
    process-local random value — invalidates tokens on restart, which
    is fine for local dev.
    """
    return os.environ.get("PGW_SECRET_KEY") or _DEV_SECRET


_DEV_SECRET = secrets.token_urlsafe(32)


def _serializer() -> URLSafeSerializer:
    return URLSafeSerializer(_secret(), salt="pgw-csrf")


def issue_csrf_token() -> str:
    """Mint a fresh token to drop in the response cookie."""
    return _serializer().dumps(secrets.token_urlsafe(16))


def verify_csrf(request: Request) -> None:
    """Raise ``HTTPException(403)`` if the request lacks valid CSRF.

    GET/HEAD/OPTIONS are exempt — they must remain side-effect-free.
    """
    from fastapi import HTTPException

    from pgw.errors import Err, envelope

    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    cookie = request.cookies.get(CSRF_COOKIE)
    header = request.headers.get(CSRF_HEADER)
    if not cookie or not header:
        raise HTTPException(
            status_code=403, detail=envelope(Err.CSRF_MISSING, "missing CSRF token")
        )
    if cookie != header:
        raise HTTPException(
            status_code=403, detail=envelope(Err.CSRF_MISMATCH, "CSRF token mismatch")
        )
    try:
        _serializer().loads(cookie)
    except BadSignature as exc:
        raise HTTPException(
            status_code=403,
            detail=envelope(Err.CSRF_INVALID_SIGNATURE, "invalid CSRF token"),
        ) from exc
