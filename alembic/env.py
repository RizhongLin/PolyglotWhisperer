"""Alembic environment.

Reads the connection URL from ``PGW_DATABASE_URL`` (same env var the app
uses), so the migration runner is consistent with the application.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make sure all model modules are imported so their tables are
# registered against ``Base.metadata``. The package's ``__init__``
# pulls in every model so ``target_metadata`` reflects the live schema.
import pgw.db.models  # noqa: F401
from pgw.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _resolve_url() -> str:
    explicit = os.environ.get("PGW_DATABASE_URL")
    if explicit:
        return explicit
    return "sqlite:///./pgw_workspace/pgw.db"


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live DB connection."""
    context.configure(
        url=_resolve_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _resolve_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
