"""Declarative base for all ORM models.

All ``pgw.db.models.*`` classes inherit from ``Base``. Alembic's
``target_metadata`` reads ``Base.metadata`` to autogenerate migrations.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Project-wide SQLAlchemy declarative base."""
