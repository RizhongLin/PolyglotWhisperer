"""DB-backed session tokens.

The cookie carries a 256-bit URL-safe random raw token. The DB stores
``sha256(raw)`` so a leaked DB does not yield usable cookies.

Sessions expire 30 days after creation by default. ``last_seen_at`` is
bumped on every authenticated request so we can implement sliding
expiry later without a migration.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session as SqlaSession

from pgw.db.models.user import Session, User

_SESSION_TTL = timedelta(days=30)
_TOKEN_BYTES = 32  # 256 bits


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _aware(dt: datetime) -> datetime:
    """Ensure ``dt`` is tz-aware; SQLite roundtrips drop the tz."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def create_session(
    db: SqlaSession,
    user: User,
    *,
    user_agent: str | None = None,
    ip: str | None = None,
) -> str:
    """Mint a new session row and return the raw cookie value."""
    raw = secrets.token_urlsafe(_TOKEN_BYTES)
    now = datetime.now(timezone.utc)
    db.add(
        Session(
            token=_hash(raw),
            user_id=user.id,
            created_at=now,
            last_seen_at=now,
            expires_at=now + _SESSION_TTL,
            user_agent=user_agent,
            ip=ip,
        )
    )
    db.commit()
    return raw


def lookup_session(db: SqlaSession, raw: str) -> tuple[Session, User] | None:
    """Return the session+user pair for a raw cookie, or None."""
    row = db.scalar(select(Session).where(Session.token == _hash(raw)))
    if row is None:
        return None
    if _aware(row.expires_at) <= datetime.now(timezone.utc):
        # Expired; clean up lazily.
        db.delete(row)
        db.commit()
        return None
    user = db.get(User, row.user_id)
    if user is None:
        return None
    row.last_seen_at = datetime.now(timezone.utc)
    db.commit()
    return row, user


def revoke_session(db: SqlaSession, raw: str) -> None:
    """Delete the session row for a raw cookie. Idempotent."""
    row = db.scalar(select(Session).where(Session.token == _hash(raw)))
    if row is not None:
        db.delete(row)
        db.commit()


def purge_expired(db: SqlaSession) -> int:
    """Drop expired sessions; returns count removed.

    The comparison happens in SQL — both Postgres and SQLite handle
    ``DateTime(timezone=True) <= :param`` correctly when ``param`` is a
    timezone-aware ``datetime`` (SQLAlchemy formats it as an ISO string
    that compares lexicographically on SQLite).
    """
    now = datetime.now(timezone.utc)
    result = db.execute(delete(Session).where(Session.expires_at <= now))
    db.commit()
    return int(result.rowcount or 0)
