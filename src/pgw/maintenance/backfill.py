"""Filesystem → DB backfill.

Imports every workspace under ``base_dir`` as a row owned by ``user``.
Then walks each workspace's ``vocabulary.*.json`` and upserts entries +
occurrences into the cross-workspace vocab index.

Idempotent: relies on the natural-key UNIQUE constraints
``(owner_id, slug, timestamp)`` for workspaces and
``(user_id, language, lemma, pos)`` for vocab entries. Re-running the
backfill on a partially-imported tree skips existing rows.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as SqlaSession

from pgw.db.models.user import User
from pgw.db.models.vocab import VocabEntry, VocabOccurrence
from pgw.db.models.workspace import Workspace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackfillReport:
    workspaces_imported: int
    workspaces_skipped: int
    vocab_entries_imported: int
    vocab_occurrences_imported: int
    workspaces_failed: int = 0


def run(db: SqlaSession, *, owner: User, base_dir: Path) -> BackfillReport:
    """Import everything under ``base_dir`` into the DB.

    The discovery pass is delegated to ``server.templates`` so the CLI
    and the web library see the same workspaces.
    """
    from pgw.server.templates import _discover_workspaces

    if owner.id is None:
        raise ValueError("backfill requires a persisted owner with a real id")

    rows = _discover_workspaces(base_dir, backfill_metadata=False)
    imported = 0
    skipped = 0
    failed = 0
    vocab_entries = 0
    vocab_occurrences = 0

    for row in rows:
        # Per-workspace SAVEPOINT — one corrupt vocab JSON or a stray
        # FK collision rolls back just that workspace, not the entire
        # batch. Counters are only bumped after the savepoint commits.
        try:
            with db.begin_nested():
                ws, is_new = _upsert_workspace(db, owner=owner, row=row)
                e_added, o_added = _upsert_vocab(db, owner=owner, workspace=ws, ws_path=row["path"])
        except IntegrityError:
            logger.warning(
                "backfill: integrity error on workspace %s/%s — skipped",
                row.get("slug"),
                row.get("timestamp"),
            )
            failed += 1
            continue
        except Exception:  # noqa: BLE001
            logger.exception(
                "backfill: unexpected error on workspace %s/%s — skipped",
                row.get("slug"),
                row.get("timestamp"),
            )
            failed += 1
            continue
        if is_new:
            imported += 1
        else:
            skipped += 1
        vocab_entries += e_added
        vocab_occurrences += o_added

    db.commit()
    report = BackfillReport(
        workspaces_imported=imported,
        workspaces_skipped=skipped,
        vocab_entries_imported=vocab_entries,
        vocab_occurrences_imported=vocab_occurrences,
        workspaces_failed=failed,
    )
    logger.info("backfill complete: %s", report)
    return report


def _upsert_workspace(db: SqlaSession, *, owner: User, row: dict) -> tuple[Workspace, bool]:
    """Return ``(workspace_row, was_newly_inserted)``."""
    existing = db.scalar(
        select(Workspace).where(
            Workspace.owner_id == owner.id,
            Workspace.slug == row["slug"],
            Workspace.timestamp == row["timestamp"],
        )
    )
    if existing is not None:
        return existing, False
    ws = Workspace(
        owner_id=owner.id,
        slug=row["slug"],
        timestamp=row["timestamp"],
        title=row.get("title") or row["slug"],
        source_url=row.get("source_url"),
        source_language=row.get("language") or None,
        target_language=row.get("target_language") or None,
        duration_seconds=row.get("duration"),
        fs_path=str(row["path"]),
        metadata_json={
            "uploader": row.get("uploader"),
            "thumbnail": row.get("thumbnail"),
            "upload_date": row.get("upload_date"),
            "difficulty": row.get("difficulty"),
        },
    )
    db.add(ws)
    db.flush()  # populate ws.id without ending the transaction
    return ws, True


def _upsert_vocab(
    db: SqlaSession, *, owner: User, workspace: Workspace, ws_path: Path
) -> tuple[int, int]:
    entries_added = 0
    occurrences_added = 0
    for vocab_file in sorted(ws_path.glob("vocabulary.*.json")):
        language = _language_from_filename(vocab_file.name)
        if language is None:
            continue
        try:
            data = json.loads(vocab_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("skipping unreadable vocab file: %s", vocab_file)
            continue
        for word in _iter_words(data):
            entry, e_added = _get_or_create_entry(db, owner=owner, language=language, word=word)
            entries_added += int(e_added)
            occurrences_added += _record_occurrence(db, entry=entry, workspace=workspace, word=word)
    return entries_added, occurrences_added


def _language_from_filename(name: str) -> str | None:
    """``vocabulary.fr.json`` → ``"fr"``."""
    parts = name.split(".")
    if len(parts) != 3 or parts[0] != "vocabulary" or parts[2] != "json":
        return None
    return parts[1] or None


def _iter_words(data: dict) -> Iterable[dict]:
    """Pull individual word entries from a vocab JSON.

    The shape is what ``vocab/summary.py`` writes — ``top_rare_words``
    is the list we backfill from.
    """
    rare = data.get("top_rare_words")
    if isinstance(rare, list):
        yield from (w for w in rare if isinstance(w, dict))


def _get_or_create_entry(
    db: SqlaSession, *, owner: User, language: str, word: dict
) -> tuple[VocabEntry, bool]:
    lemma = (word.get("lemma") or word.get("word") or "").strip()
    pos = (word.get("pos") or "").strip() or None
    if not lemma:
        # Synthesize a placeholder so the tuple still types correctly;
        # caller will skip via the natural-key uniqueness lookup.
        return VocabEntry(user_id=owner.id, language=language, lemma="", pos=pos), False
    existing = db.scalar(
        select(VocabEntry).where(
            VocabEntry.user_id == owner.id,
            VocabEntry.language == language,
            VocabEntry.lemma == lemma,
            VocabEntry.pos.is_(pos) if pos is None else VocabEntry.pos == pos,
        )
    )
    if existing is not None:
        return existing, False
    entry = VocabEntry(
        user_id=owner.id,
        language=language,
        lemma=lemma,
        pos=pos,
        zipf=word.get("zipf"),
        cefr=word.get("difficulty"),
    )
    db.add(entry)
    db.flush()
    return entry, True


def _record_occurrence(
    db: SqlaSession,
    *,
    entry: VocabEntry,
    workspace: Workspace,
    word: dict,
) -> int:
    """Append an occurrence row if one isn't already present.

    Vocab JSON only stores one example sentence per word so we treat
    that as a single sighting at segment_index 0 with start=end=0 if
    timing isn't known. Backfill is best-effort — runtime updates from
    P5 attach the real segment+timestamps.
    """
    if not entry.lemma:
        return 0
    surface = (word.get("word") or "").strip()
    context = word.get("context")
    translation = word.get("translation")
    existing = db.scalar(
        select(VocabOccurrence).where(
            VocabOccurrence.entry_id == entry.id,
            VocabOccurrence.workspace_id == workspace.id,
            VocabOccurrence.surface == surface,
        )
    )
    if existing is not None:
        return 0
    db.add(
        VocabOccurrence(
            entry_id=entry.id,
            workspace_id=workspace.id,
            segment_index=0,
            start_seconds=0.0,
            end_seconds=0.0,
            surface=surface,
            context=context,
            translation=translation,
        )
    )
    return 1
