"""Session factory + FastAPI dependency.

Use ``with SessionLocal() as s:`` for one-off scripts. Use
``Depends(get_session)`` in FastAPI routes — it ensures the session is
closed on response.

The sessionmaker is rebuilt every call to honour ``reset_engine()``
(tests, hot-reload). The cost is negligible — sessionmaker is just a
factory wrapper, not a connection.
"""

from __future__ import annotations

from typing import Iterator

from sqlalchemy.orm import Session, sessionmaker

from pgw.db.engine import get_engine


class _SessionLocal:
    """Per-call sessionmaker so engine resets propagate.

    Caching the sessionmaker process-wide (the obvious approach) means
    ``reset_engine`` doesn't actually swap the underlying engine — old
    sessions keep using the disposed engine, leaking state across
    tests. We rebuild on every ``__call__`` to avoid that.
    """

    def __call__(self) -> Session:
        return sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
            future=True,
        )()


SessionLocal = _SessionLocal()


def get_session() -> Iterator[Session]:
    """FastAPI dependency: yield a session, close on completion."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
