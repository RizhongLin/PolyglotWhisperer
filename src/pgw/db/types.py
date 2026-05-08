"""Shared SQLAlchemy column-type helpers.

Centralises the cross-dialect compromises so each model file doesn't
re-derive them. The big one today is the BigInteger PK quirk: Postgres
wants ``BIGINT``, SQLite needs ``INTEGER PRIMARY KEY`` for the rowid
auto-increment to fire. ``BigIntPK`` returns one or the other depending
on the dialect at DDL time.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Integer

#: Use as the column type for primary keys and FKs to BIGINT-PKs.
#: Renders as ``BIGINT`` on Postgres and ``INTEGER`` on SQLite (so the
#: ORM-driven ``Base.metadata.create_all`` test path works without a
#: separate migration column-type mapping).
BigIntPK = BigInteger().with_variant(Integer(), "sqlite")
