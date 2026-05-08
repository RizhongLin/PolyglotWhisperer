"""flashcard LLM refinement

Revision ID: 0005_flashcard_refinement
Revises: 0004_flashcards
Create Date: 2026-05-08

Adds the columns the LLM refinement pass writes (lemma/pos/definition/
example_source/example_target/mnemonic) plus a small state machine
(``refine_status``, ``refine_attempts``, ``refine_model``,
``refined_at``).

A separate ``flashcard_refinements`` table caches the language-agnostic
parts (definition + mnemonic) keyed on ``(language, lemma, pos)`` so a
re-run on the same lemma is free; example sentences are NOT cached
because they're context-dependent.

Existing rows get ``refine_status='skipped'`` so we don't accidentally
LLM-blast the entire backfill on the first POST after upgrade — a
deliberate ``pgw maintenance refine-flashcards`` invocation flips them.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005_flashcard_refinement"
down_revision: Union[str, None] = "0004_flashcards"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── flashcards: new content + state columns ──
    with op.batch_alter_table("flashcards") as batch:
        batch.add_column(sa.Column("lemma", sa.Text(), nullable=True))
        batch.add_column(sa.Column("pos", sa.String(length=16), nullable=True))
        batch.add_column(sa.Column("definition", sa.Text(), nullable=True))
        batch.add_column(sa.Column("example_source", sa.Text(), nullable=True))
        batch.add_column(sa.Column("example_target", sa.Text(), nullable=True))
        batch.add_column(sa.Column("mnemonic", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column(
                "refine_status",
                sa.String(length=16),
                nullable=False,
                server_default="skipped",
            )
        )
        batch.add_column(
            sa.Column(
                "refine_attempts",
                sa.SmallInteger(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
        batch.add_column(sa.Column("refine_model", sa.Text(), nullable=True))
        batch.add_column(sa.Column("refined_at", sa.DateTime(timezone=True), nullable=True))

    op.create_index(
        "ix_flashcards_refine_status",
        "flashcards",
        ["refine_status"],
    )

    # ── flashcard_refinements: per-(language, lemma, pos) cache ──
    op.create_table(
        "flashcard_refinements",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("language", sa.String(length=8), nullable=False),
        sa.Column("lemma", sa.Text(), nullable=False),
        sa.Column("pos", sa.String(length=16), nullable=True),
        sa.Column("definition", sa.Text(), nullable=False),
        sa.Column("mnemonic", sa.Text(), nullable=True),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "language", "lemma", "pos", name="uq_flashcard_refinements_natural_key"
        ),
    )
    op.create_index(
        "ix_flashcard_refinements_lookup",
        "flashcard_refinements",
        ["language", "lemma", "pos"],
    )


def downgrade() -> None:
    op.drop_index("ix_flashcard_refinements_lookup", table_name="flashcard_refinements")
    op.drop_table("flashcard_refinements")
    op.drop_index("ix_flashcards_refine_status", table_name="flashcards")
    with op.batch_alter_table("flashcards") as batch:
        for col in (
            "refined_at",
            "refine_model",
            "refine_attempts",
            "refine_status",
            "mnemonic",
            "example_target",
            "example_source",
            "definition",
            "pos",
            "lemma",
        ):
            batch.drop_column(col)
