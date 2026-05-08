"""baseline

Revision ID: 0001_baseline
Revises:
Create Date: 2026-05-07

P1 ships an intentionally empty baseline so a fresh clone +
``alembic upgrade head`` is a no-op no matter how many phases land.
P2's ``0002_auth_workspaces_vocab`` is the first migration with real
tables.
"""

from __future__ import annotations

from typing import Sequence, Union

revision: str = "0001_baseline"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
