"""First-boot admin provisioning.

Three paths, in priority order:

1. ``PGW_ADMIN_EMAIL`` + ``PGW_ADMIN_PASSWORD`` set in env → create
   admin synchronously at server start. For Docker.
2. Web ``/setup`` flow → SPA detects no admin (via
   ``GET /api/auth/state``) and asks the visitor to create one.
3. CLI ``pgw admin create-user --admin`` → for terminal-friendly
   bootstrap.

Idempotent: re-running with an admin already in place is a no-op.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import select, text
from sqlalchemy.orm import Session as SqlaSession

from pgw.auth.passwords import hash_password
from pgw.db.models.user import User

logger = logging.getLogger(__name__)

# Process-level mutex serialises the bootstrap path within one process
# (gunicorn worker / uvicorn reloader / pytest). Cross-process races on
# Postgres are handled by ``setup_lock`` via a transactional advisory
# lock; SQLite is single-writer in practice so the threading lock is
# sufficient there.
_SETUP_THREAD_LOCK = threading.Lock()
# Arbitrary 64-bit int used as the advisory-lock key. Chosen randomly
# from a high band to keep collision risk with app code negligible.
_SETUP_ADVISORY_KEY = 0x70_67_77_5F_5F_5F_5F_31  # "pgw___1"


def has_any_user(db: SqlaSession) -> bool:
    return db.scalar(select(User.id).limit(1)) is not None


@contextmanager
def setup_lock(db: SqlaSession) -> Iterator[None]:
    """Serialise bootstrap setup across processes and threads.

    On Postgres, takes a transactional advisory lock that is auto-
    released on commit/rollback. On other dialects (SQLite for tests),
    the in-process ``_SETUP_THREAD_LOCK`` is the only barrier — but
    SQLite serves bootstrap from a single process anyway.
    """
    bind = db.get_bind()
    is_postgres = bind is not None and bind.dialect.name == "postgresql"
    with _SETUP_THREAD_LOCK:
        if is_postgres:
            db.execute(
                text("SELECT pg_advisory_xact_lock(:k)"),
                {"k": _SETUP_ADVISORY_KEY},
            )
        yield


def ensure_admin_from_env(db: SqlaSession) -> User | None:
    """If env-vars are set and no user exists yet, create the admin.

    Returns the new user, or None if no env vars were set / a user
    already exists.
    """
    email = os.environ.get("PGW_ADMIN_EMAIL")
    password = os.environ.get("PGW_ADMIN_PASSWORD")
    if not email or not password:
        return None
    if has_any_user(db):
        return None
    user = User(
        email=email.lower(),
        password_hash=hash_password(password),
        is_admin=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.warning("Created bootstrap admin %s from PGW_ADMIN_* env vars", email)
    return user


def create_user(
    db: SqlaSession,
    *,
    email: str,
    password: str,
    is_admin: bool = False,
) -> User:
    """Direct user creation. Used by setup endpoint + CLI."""
    user = User(
        email=email.lower().strip(),
        password_hash=hash_password(password),
        is_admin=is_admin,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
