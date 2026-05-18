"""user credentials and preferences

Revision ID: 0006_user_credentials
Revises: 0005_flashcard_refinement
Create Date: 2026-05-18

Adds:
- ``user_credentials`` table for encrypted per-user LLM/Whisper API keys.
- ``preferences`` JSON column on ``users`` for user-level defaults.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006_user_credentials"
down_revision: Union[str, None] = "0005_flashcard_refinement"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_credentials",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.BigInteger(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("service", sa.String(length=20), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("encrypted_value", sa.Text(), nullable=False),
        sa.Column("api_base", sa.String(length=500), nullable=True),
        sa.Column("api_model", sa.String(length=200), nullable=True),
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
    op.create_index(
        "ix_user_credentials_user_id",
        "user_credentials",
        ["user_id"],
    )

    with op.batch_alter_table("users") as batch:
        batch.add_column(
            sa.Column(
                "preferences",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'{}'"),
            )
        )


def downgrade() -> None:
    op.drop_index("ix_user_credentials_user_id", table_name="user_credentials")
    op.drop_table("user_credentials")
    with op.batch_alter_table("users") as batch:
        batch.drop_column("preferences")
