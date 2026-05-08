"""flashcards + fsrs review log

Revision ID: 0004_flashcards
Revises: 0003_workers
Create Date: 2026-05-08

P6 schema. Tables:
  flashcards, flashcard_reviews

Card-level state (stability/difficulty/due) lives on the row so the
review queue is a single indexed scan. Per-grade history lives in
``flashcard_reviews`` so we can re-derive state if FSRS is retuned.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_flashcards"
down_revision: Union[str, None] = "0003_workers"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "flashcards",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.BigInteger(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "workspace_id",
            sa.BigInteger(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "vocab_entry_id",
            sa.BigInteger(),
            sa.ForeignKey("vocab_entries.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("front", sa.Text(), nullable=False),
        sa.Column("back", sa.Text(), nullable=False),
        sa.Column("language", sa.Text(), nullable=False),
        sa.Column("audio_start_ms", sa.Integer(), nullable=True),
        sa.Column("audio_end_ms", sa.Integer(), nullable=True),
        sa.Column("fsrs_stability", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("fsrs_difficulty", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "fsrs_due",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
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
    op.create_index("ix_flashcards_user_due", "flashcards", ["user_id", "fsrs_due"])
    op.create_index("ix_flashcards_workspace", "flashcards", ["workspace_id"])

    op.create_table(
        "flashcard_reviews",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "flashcard_id",
            sa.BigInteger(),
            sa.ForeignKey("flashcards.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rating", sa.SmallInteger(), nullable=False),
        sa.Column("elapsed_ms", sa.Integer(), nullable=True),
        sa.Column("fsrs_stability_after", sa.Float(), nullable=False),
        sa.Column("fsrs_difficulty_after", sa.Float(), nullable=False),
        sa.Column("fsrs_due_after", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_flashcard_reviews_card_time",
        "flashcard_reviews",
        ["flashcard_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_flashcard_reviews_card_time", table_name="flashcard_reviews")
    op.drop_table("flashcard_reviews")
    op.drop_index("ix_flashcards_workspace", table_name="flashcards")
    op.drop_index("ix_flashcards_user_due", table_name="flashcards")
    op.drop_table("flashcards")
