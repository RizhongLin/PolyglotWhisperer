"""Post-pipeline DB sync: read workspace files → upsert vocab + workspace rows."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session as SqlaSession

from pgw.db.models.vocab import VocabEntry, VocabOccurrence
from pgw.db.models.workspace import Workspace


def sync_workspace_to_db(
    db: SqlaSession,
    workspace_path: Path,
    owner_id: int,
) -> int:
    """Read workspace files from disk and upsert vocacb + workspace row.

    Called after pipeline completion in server mode.  The pipeline already
    wrote ``metadata.json``, ``transcription.<lang>.vtt``, and
    ``vocabulary.<lang>.json`` to disk — this function reads those files
    and populates the corresponding DB rows.

    Returns the workspace row id.
    """
    meta_path = workspace_path / "metadata.json"
    meta = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    slug = workspace_path.parent.name
    timestamp = workspace_path.name

    # ── Upsert workspace row ──────────────────────────────────────────
    existing = db.scalar(
        select(Workspace).where(
            Workspace.owner_id == owner_id,
            Workspace.slug == slug,
            Workspace.timestamp == timestamp,
        )
    )
    if existing is not None:
        ws_id = existing.id
        # Update mutable fields
        existing.title = str(meta.get("title") or slug)
        existing.source_url = meta.get("source_url") or None
        existing.source_language = meta.get("language") or None
        existing.target_language = meta.get("target_language") or None
        existing.duration_seconds = meta.get("source_duration")
        existing.metadata_json = {
            "uploader": meta.get("uploader"),
            "thumbnail": meta.get("thumbnail"),
            "upload_date": meta.get("upload_date"),
            "difficulty": meta.get("difficulty"),
        }
    else:
        ws = Workspace(
            owner_id=owner_id,
            slug=slug,
            timestamp=timestamp,
            title=str(meta.get("title") or slug),
            source_url=meta.get("source_url") or None,
            source_language=meta.get("language") or None,
            target_language=meta.get("target_language") or None,
            duration_seconds=meta.get("source_duration"),
            fs_path=str(workspace_path),
            metadata_json={
                "uploader": meta.get("uploader"),
                "thumbnail": meta.get("thumbnail"),
                "upload_date": meta.get("upload_date"),
                "difficulty": meta.get("difficulty"),
            },
        )
        db.add(ws)
        db.flush()
        ws_id = ws.id

    # ── Upsert vocabulary ────────────────────────────────────────────
    lang = meta.get("language") or "fr"
    for vocab_file in sorted(workspace_path.glob("vocabulary.*.json")):
        try:
            vdata = json.loads(vocab_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        _upsert_vocab(db, ws_id, owner_id, vdata, lang)

    db.commit()
    return ws_id


def _upsert_vocab(
    db: SqlaSession,
    workspace_id: int,
    user_id: int,
    vdata: dict,
    language: str,
) -> None:
    """Insert or update vocabulary entries + occurrences from a vocab dict."""
    top_words = vdata.get("top_rare_words") or []
    if not top_words:
        return

    # Remove old occurrences for this workspace (idempotent re-sync)
    db.execute(delete(VocabOccurrence).where(VocabOccurrence.workspace_id == workspace_id))

    for w in top_words:
        word = str(w.get("word", "")).strip()
        lemma = str(w.get("lemma", word)).strip()
        pos = w.get("pos") or None
        translation = w.get("translation") or None
        context = w.get("context") or None
        difficulty = w.get("difficulty") or None
        zipf = w.get("zipf")
        if zipf is not None:
            try:
                zipf = float(zipf)
            except (TypeError, ValueError):
                zipf = None

        entry = _get_or_create_entry(db, user_id=user_id, language=language, lemma=lemma, pos=pos)
        if zipf is not None:
            entry.zipf = zipf
        if difficulty and not entry.cefr:
            entry.cefr = difficulty

        occ = VocabOccurrence(
            entry_id=entry.id,
            workspace_id=workspace_id,
            segment_index=0,
            start_seconds=0.0,
            end_seconds=0.0,
            surface=word,
            context=context,
            translation=translation,
        )
        db.add(occ)


def _get_or_create_entry(
    db: SqlaSession,
    *,
    user_id: int,
    language: str,
    lemma: str,
    pos: str | None,
) -> VocabEntry:
    existing = db.scalar(
        select(VocabEntry).where(
            VocabEntry.user_id == user_id,
            VocabEntry.language == language,
            VocabEntry.lemma == lemma,
            VocabEntry.pos == pos,
        )
    )
    if existing is not None:
        return existing
    entry = VocabEntry(
        user_id=user_id,
        language=language,
        lemma=lemma,
        pos=pos,
    )
    db.add(entry)
    db.flush()
    return entry
