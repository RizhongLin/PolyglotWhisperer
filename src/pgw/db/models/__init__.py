"""ORM model registry.

Importing this package registers every model against ``Base.metadata``,
which is what Alembic's ``target_metadata`` reads for autogenerate.
"""

from __future__ import annotations

from pgw.db.models.flashcard import Flashcard, FlashcardRefinement, FlashcardReview
from pgw.db.models.user import Session, User
from pgw.db.models.vocab import VocabEntry, VocabOccurrence
from pgw.db.models.worker import WorkerSession, WorkerToken
from pgw.db.models.workspace import Workspace, WorkspaceJob

__all__ = [
    "User",
    "Session",
    "Workspace",
    "WorkspaceJob",
    "VocabEntry",
    "VocabOccurrence",
    "WorkerToken",
    "WorkerSession",
    "Flashcard",
    "FlashcardRefinement",
    "FlashcardReview",
]
