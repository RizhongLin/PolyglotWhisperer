"""Worker token issuance + lookup.

Server-side helper used by ``/api/workers`` REST handlers and the
``/ws/worker`` upgrade handshake.

Tokens are 32-byte URL-safe random strings shown to the user once. The
DB stores ``sha256(raw)`` so a leaked DB doesn't yield usable tokens.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session as SqlaSession

from pgw.db.models.user import User
from pgw.db.models.worker import WorkerToken

_TOKEN_BYTES = 32


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def issue(db: SqlaSession, user: User, *, name: str) -> tuple[WorkerToken, str]:
    """Mint a fresh worker token for ``user``.

    Returns ``(row, raw)``. ``raw`` is the value to surface to the user
    one time; it is not recoverable from the DB.
    """
    raw = secrets.token_urlsafe(_TOKEN_BYTES)
    row = WorkerToken(user_id=user.id, name=name.strip(), token_hash=_hash(raw))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row, raw


def lookup(db: SqlaSession, raw: str) -> WorkerToken | None:
    """Resolve a raw token to a non-revoked DB row, or ``None``."""
    row = db.scalar(select(WorkerToken).where(WorkerToken.token_hash == _hash(raw)))
    if row is None:
        return None
    if row.revoked_at is not None:
        return None
    return row


def revoke(db: SqlaSession, *, user: User, token_id: int) -> bool:
    """Mark a token as revoked. Returns True if found + revoked."""
    row = db.get(WorkerToken, token_id)
    if row is None or row.user_id != user.id:
        return False
    row.revoked_at = datetime.now(timezone.utc)
    db.commit()
    return True
