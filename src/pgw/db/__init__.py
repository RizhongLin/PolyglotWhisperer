"""Database access layer.

Postgres is the production target; SQLite is supported as an escape
hatch (tests, single-user demos) by virtue of SQLAlchemy's dialect
abstraction.

Connection string comes from ``PGW_DATABASE_URL``; default is a local
SQLite file so importing this module from the CLI never fails.

Phase ownership:
- P1: ``Base``, ``engine``, ``SessionLocal``. No tables yet.
- P2: ORM models in ``pgw.db.models.*``.
- P3+: more models added phase by phase.
"""

from __future__ import annotations

from pgw.db.base import Base
from pgw.db.engine import get_engine
from pgw.db.session import SessionLocal, get_session

__all__ = ["Base", "get_engine", "SessionLocal", "get_session"]
