"""Shared SQLAlchemy column-type helpers.

Centralises the cross-dialect compromises so each model file doesn't
re-derive them. The big one today is the BigInteger PK quirk: Postgres
wants ``BIGINT``, SQLite needs ``INTEGER PRIMARY KEY`` for the rowid
auto-increment to fire. ``BigIntPK`` returns one or the other depending
on the dialect at DDL time.
"""

from __future__ import annotations

import json

from sqlalchemy import BigInteger, Integer, Text
from sqlalchemy.dialects.postgresql import JSON as PGJSON
from sqlalchemy.types import TypeDecorator

#: Use as the column type for primary keys and FKs to BIGINT-PKs.
#: Renders as ``BIGINT`` on Postgres and ``INTEGER`` on SQLite (so the
#: ORM-driven ``Base.metadata.create_all`` test path works without a
#: separate migration column-type mapping).
BigIntPK = BigInteger().with_variant(Integer(), "sqlite")


class JSONText(TypeDecorator):
    """JSON stored as TEXT for SQLite, native JSON for Postgres.

    Auto-serialises Python dicts to JSON strings on write and
    deserialises back to dicts on read for SQLite backends.
    """

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PGJSON())
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.dumps(value, ensure_ascii=False)

    def process_result_value(self, value, dialect):
        if value is None:
            return {}
        if dialect.name == "postgresql":
            return value if isinstance(value, dict) else {}
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return {}
        return value if isinstance(value, dict) else {}
