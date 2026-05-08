"""worker tokens + sessions

Revision ID: 0003_workers
Revises: 0002_auth_workspaces_vocab
Create Date: 2026-05-07
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003_workers"
down_revision: Union[str, None] = "0002_auth_workspaces_vocab"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "worker_tokens",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.BigInteger(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False, unique=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_worker_tokens_user_id", "worker_tokens", ["user_id"])

    op.create_table(
        "worker_sessions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "token_id",
            sa.BigInteger(),
            sa.ForeignKey("worker_tokens.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "connected_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("disconnected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("hostname", sa.Text(), nullable=True),
        sa.Column("pgw_version", sa.Text(), nullable=True),
        sa.Column("capabilities", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_worker_sessions_token_id", "worker_sessions", ["token_id"])


def downgrade() -> None:
    op.drop_index("ix_worker_sessions_token_id", table_name="worker_sessions")
    op.drop_table("worker_sessions")
    op.drop_index("ix_worker_tokens_user_id", table_name="worker_tokens")
    op.drop_table("worker_tokens")
