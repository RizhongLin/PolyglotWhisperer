"""auth + workspaces + vocab

Revision ID: 0002_auth_workspaces_vocab
Revises: 0001_baseline
Create Date: 2026-05-07

P2 schema. Tables:
  users, sessions,
  workspaces, workspace_jobs,
  vocab_entries, vocab_occurrences

Embed columns on ``workspaces`` are provisioned here but populated in
P4. Same goes for ``workspace_jobs`` — schema lives here, behavior
filled out in P3.

Written via ``op.create_table`` so it works on Postgres (production)
and SQLite (tests, local dev).
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_auth_workspaces_vocab"
down_revision: Union[str, None] = "0001_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("uq_users_email", "users", ["email"], unique=True)

    op.create_table(
        "sessions",
        sa.Column("token", sa.String(length=64), primary_key=True),
        sa.Column(
            "user_id",
            sa.BigInteger(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("ip", sa.String(length=45), nullable=True),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])
    op.create_index("ix_sessions_expires_at", "sessions", ["expires_at"])

    op.create_table(
        "workspaces",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "owner_id",
            sa.BigInteger(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("source_language", sa.Text(), nullable=True),
        sa.Column("target_language", sa.Text(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("embed_provider", sa.Text(), nullable=True),
        sa.Column("embed_url", sa.Text(), nullable=True),
        sa.Column("embed_blocked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fs_path", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("owner_id", "slug", "timestamp", name="uq_workspace_owner_path"),
    )
    op.create_index("ix_workspaces_owner_id", "workspaces", ["owner_id"])
    op.create_index("ix_workspaces_owner_created", "workspaces", ["owner_id", "created_at"])

    op.create_table(
        "workspace_jobs",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "owner_id",
            sa.BigInteger(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "workspace_id",
            sa.BigInteger(),
            sa.ForeignKey("workspaces.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("inputs", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("progress", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("stage", sa.Text(), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_workspace_jobs_owner_id", "workspace_jobs", ["owner_id"])
    op.create_index(
        "ix_workspace_jobs_owner_created",
        "workspace_jobs",
        ["owner_id", "created_at"],
    )

    op.create_table(
        "vocab_entries",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.BigInteger(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("language", sa.Text(), nullable=False),
        sa.Column("lemma", sa.Text(), nullable=False),
        sa.Column("pos", sa.Text(), nullable=True),
        sa.Column("zipf", sa.Float(), nullable=True),
        sa.Column("cefr", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "user_id", "language", "lemma", "pos", name="uq_vocab_entry_natural_key"
        ),
    )
    op.create_index("ix_vocab_entries_user_id", "vocab_entries", ["user_id"])
    op.create_index("ix_vocab_entries_user_lang", "vocab_entries", ["user_id", "language"])

    op.create_table(
        "vocab_occurrences",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "entry_id",
            sa.BigInteger(),
            sa.ForeignKey("vocab_entries.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "workspace_id",
            sa.BigInteger(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("segment_index", sa.Integer(), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("end_seconds", sa.Float(), nullable=False),
        sa.Column("surface", sa.Text(), nullable=False),
        sa.Column("context", sa.Text(), nullable=True),
        sa.Column("translation", sa.Text(), nullable=True),
    )
    op.create_index("ix_vocab_occurrences_entry_id", "vocab_occurrences", ["entry_id"])
    op.create_index("ix_vocab_occurrences_workspace_id", "vocab_occurrences", ["workspace_id"])


def downgrade() -> None:
    op.drop_table("vocab_occurrences")
    op.drop_table("vocab_entries")
    op.drop_table("workspace_jobs")
    op.drop_table("workspaces")
    op.drop_table("sessions")
    op.drop_index("uq_users_email", table_name="users")
    op.drop_table("users")
