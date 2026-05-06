"""HTML rendering and data assembly for the web player and library."""

from __future__ import annotations

import html
import json
import re
import typing
import urllib.parse
from importlib.resources import files
from pathlib import Path

from pgw.core.languages import language_name
from pgw.utils.console import console
from pgw.utils.paths import (
    GITHUB_URL,
    GLOB_BILINGUAL_VTT,
    GLOB_TRANSCRIPTION_VTT,
    GLOB_TRANSLATION_VTT,
    GLOB_VOCABULARY_JSON,
    METADATA_FILE,
    find_video,
)
from pgw.utils.text import BYTES_PER_KB, BYTES_PER_MB, SECONDS_PER_HOUR

# ── Template loading ──
_PLAYER_TEMPLATE = (files("pgw.templates") / "player.html").read_text(encoding="utf-8")
_PLAYER_CSS = (files("pgw.templates") / "player.css").read_text(encoding="utf-8")
_LIBRARY_TEMPLATE = (files("pgw.templates") / "library.html").read_text(encoding="utf-8")
_LIBRARY_CSS = (files("pgw.templates") / "library.css").read_text(encoding="utf-8")
_PLAYER_JS = (files("pgw.templates") / "player.js").read_text(encoding="utf-8")
_ICON_PNG = (files("pgw.templates") / "icon.png").read_bytes()
_LOGO_PNG = (files("pgw.templates") / "logo.png").read_bytes()

# ── Constants ──
_DESC_MAX_CHARS = 200
_DEFAULT_VIDEO_EXT = ".mp4"
_METADATA_FIELDS = ("upload_date", "uploader", "thumbnail", "description")
_SIBLING_PREFIX = "sibling:"


def _lang_short(code: str) -> str:
    """Format a language code as uppercase abbreviation (e.g. 'FR')."""
    return code.upper()


def _lang_full(code: str) -> str:
    """Format a language code as full name with code (e.g. 'French (fr)')."""
    name = language_name(code).title()
    if name.lower() == code.lower():
        return name
    return f"{name} ({code})"


def _discover_tracks(
    workspace: Path, sibling_paths: list[Path] | None = None
) -> list[dict[str, str]]:
    """Find subtitle files across a workspace and its siblings."""
    all_dirs: list[tuple[Path, str]] = [(workspace, "")]
    if sibling_paths:
        for sp in sibling_paths:
            all_dirs.append((sp, f"{_SIBLING_PREFIX}{sp.name}/"))

    tracks: list[dict[str, str]] = []
    seen_labels: set[str] = set()

    for ws_dir, prefix in all_dirs:
        for f in sorted(ws_dir.glob(GLOB_BILINGUAL_VTT)):
            parts = f.stem.split(".")
            if len(parts) >= 2:
                lang_pair = parts[1]
                label = f"Bilingual ({lang_pair})"
                if label not in seen_labels:
                    seen_labels.add(label)
                    tracks.append({"file": prefix + f.name, "label": label, "lang": lang_pair})

        for f in sorted(ws_dir.glob(GLOB_TRANSCRIPTION_VTT)):
            parts = f.stem.split(".")
            if len(parts) >= 2:
                lang = parts[1]
                label = f"Original ({lang})"
                if label not in seen_labels:
                    seen_labels.add(label)
                    tracks.append({"file": prefix + f.name, "label": label, "lang": lang})

        for f in sorted(ws_dir.glob(GLOB_TRANSLATION_VTT)):
            parts = f.stem.split(".")
            if len(parts) >= 2:
                lang = parts[1]
                label = f"Translation ({lang})"
                if label not in seen_labels:
                    seen_labels.add(label)
                    tracks.append({"file": prefix + f.name, "label": label, "lang": lang})

    return tracks


_MIME_MAP = {
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".ts": "video/mp2t",
    ".flv": "video/x-flv",
}


def _format_duration(seconds: float | None) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    if not seconds:
        return ""
    total = int(seconds)
    h, remainder = divmod(total, SECONDS_PER_HOUR)
    m, s = divmod(remainder, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _format_file_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if size_bytes < BYTES_PER_KB:
        return f"{size_bytes} B"
    if size_bytes < BYTES_PER_MB:
        return f"{size_bytes / BYTES_PER_KB:.0f} KB"
    return f"{size_bytes / BYTES_PER_MB:.1f} MB"


def _load_metadata(workspace: Path) -> dict:
    """Load metadata.json from workspace, return empty dict on failure."""
    meta_path = workspace / METADATA_FILE
    if meta_path.is_file():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _build_metadata_rows(meta: dict) -> str:
    """Build <dt>/<dd> pairs for the metadata card."""
    rows = []

    def add(icon: str, label: str, value: str) -> None:
        if value:
            rows.append(
                f'<dt><i data-lucide="{icon}"></i> {html.escape(label)}</dt>' f"<dd>{value}</dd>"
            )

    lang = meta.get("language", "")
    target = meta.get("target_language", "")
    if lang and target:
        add("languages", "Languages", f"{_lang_full(lang)} &rarr; {_lang_full(target)}")
    elif lang:
        add("languages", "Language", _lang_full(lang))

    uploader = meta.get("uploader", "")
    if uploader:
        add("user", "Channel", html.escape(uploader))

    upload_date = meta.get("upload_date", "")
    if upload_date and len(upload_date) == 8:
        formatted = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
        add("calendar-plus", "Uploaded", html.escape(formatted))

    dur = _format_duration(meta.get("source_duration"))
    add("clock", "Duration", html.escape(dur))

    whisper = html.escape(meta.get("whisper_model", ""))
    if whisper:
        add("mic", "Whisper", f"<code>{whisper}</code>")
    llm = html.escape(meta.get("llm_model", ""))
    if llm:
        add("brain", "LLM", f"<code>{llm}</code>")

    source_url = meta.get("source_url")
    if source_url:
        safe_url = html.escape(source_url)
        try:
            domain = urllib.parse.urlparse(source_url).netloc
        except Exception:
            domain = source_url
        add(
            "external-link",
            "Source",
            f'<a href="{safe_url}" target="_blank" rel="noopener"'
            f' title="{safe_url}">{html.escape(domain)}</a>',
        )

    desc = meta.get("description", "")
    if desc:
        short = desc[:_DESC_MAX_CHARS].rstrip() + ("…" if len(desc) > _DESC_MAX_CHARS else "")
        add("text", "Description", html.escape(short))

    created = meta.get("created_at", "")
    if created:
        add("calendar", "Created", html.escape(created[:10]))

    return "\n        ".join(rows)


_FILE_ICONS = {
    ".vtt": "captions",
    ".srt": "captions",
    ".txt": "file-text",
    ".json": "file-json-2",
    ".pdf": "file-text",
    ".epub": "book-open",
}


def _file_icon(suffix: str) -> str:
    """Return a Lucide icon name for a file extension."""
    return _FILE_ICONS.get(suffix, "file")


def _friendly_name(filename: str) -> str:
    """Convert a workspace filename to a human-readable label."""
    stem, _, ext = filename.rpartition(".")
    parts = stem.split(".")
    base = parts[0] if parts else stem
    langs = parts[1] if len(parts) > 1 else ""
    if langs:
        lang_codes = langs.split("-")
        lang_display = " \u2192 ".join(language_name(c).title() for c in lang_codes)
    else:
        lang_display = ""

    names = {
        "bilingual": "Bilingual Subtitles",
        "transcription": "Original Transcription",
        "translation": "Translation",
        "parallel": "Parallel Text",
        "vocabulary": "Vocabulary Analysis",
    }
    label = names.get(base, base.replace("_", " ").title())
    if lang_display:
        label += f" ({lang_display})"
    return label


def _build_download_rows(
    workspace: Path,
    meta: dict,
    url_prefix: str = "",
    sibling_paths: list[Path] | None = None,
) -> str:
    """Build <li> entries for downloadable workspace files."""
    skip = {METADATA_FILE}
    from pgw.utils.paths import AUDIO_FILE

    skip.add(AUDIO_FILE)

    def _files_for_dir(ws_dir: Path, prefix: str) -> list[tuple[str, str]]:
        """Return (dedup_key, html_row) pairs."""
        if ws_dir == workspace:
            ws_meta = meta.get("files", {})
        else:
            ws_meta = _load_metadata(ws_dir).get("files", {})
        items = []
        for f in sorted(ws_dir.iterdir()):
            if f.name.startswith(".") or f.name in skip:
                continue
            if f.suffix.lower() in _MIME_MAP:
                continue
            if not f.is_file():
                continue

            size = ws_meta.get(f.name, {}).get("size_bytes", f.stat().st_size)
            size_str = _format_file_size(size)
            safe_name = html.escape(f.name)
            icon = _file_icon(f.suffix.lower())
            label = _friendly_name(f.name)
            ext = f.suffix.lstrip(".").upper()
            dedup_key = f"{label}.{ext}"
            href = f"{url_prefix}/{prefix}{safe_name}"
            row_html = (
                f'<li><a href="{href}" download>'
                f'<i data-lucide="{icon}"></i>'
                f'<span class="dl-name">{html.escape(label)}'
                f'<span class="dl-ext">{ext}</span></span></a>'
                f'<span class="size">{size_str}</span></li>'
            )
            items.append((dedup_key, row_html))
        return items

    seen_keys: set[str] = set()
    rows: list[str] = []

    for entry in _files_for_dir(workspace, ""):
        seen_keys.add(entry[0])
        rows.append(entry[1])

    if sibling_paths:
        for sp in sibling_paths:
            for key, row_html in _files_for_dir(sp, f"{_SIBLING_PREFIX}{sp.name}/"):
                if key not in seen_keys:
                    seen_keys.add(key)
                    rows.append(row_html)

    return "\n        ".join(rows) if rows else "<li>No files available</li>"


_DIFFICULTY_COLORS = {
    "A1": "#2e7d32",
    "A2": "#558b2f",
    "B1": "#f57f17",
    "B2": "#e65100",
    "C1": "#c62828",
    "C2": "#6a1b9a",
}

_DIFFICULTY_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]


def _build_vocab_section(workspace: Path) -> str:
    """Build the vocabulary analysis card HTML, or empty string if unavailable."""
    vocab_files = list(workspace.glob(GLOB_VOCABULARY_JSON))
    if not vocab_files:
        return ""

    try:
        summary = json.loads(vocab_files[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""

    dist = summary.get("difficulty_distribution", {})
    total_types = sum(dist.values()) or 1
    estimated = summary.get("estimated_difficulty", "?")
    unique = summary.get("unique_lemmas", 0)

    bar_parts = []
    for level in _DIFFICULTY_ORDER:
        count = dist.get(level, 0)
        if count == 0:
            continue
        pct = count / total_types * 100
        color = _DIFFICULTY_COLORS[level]
        bar_parts.append(
            f'<span style="width:{pct:.1f}%;background:{color}" '
            f'title="{level}: {count}"></span>'
        )

    legend_parts = []
    for level in _DIFFICULTY_ORDER:
        count = dist.get(level, 0)
        if count == 0:
            continue
        color = _DIFFICULTY_COLORS[level]
        legend_parts.append(
            f'<span class="vocab-chip">'
            f'<span style="background:{color}" class="vocab-dot"></span>'
            f"{level} <small>({count})</small></span>"
        )

    words = summary.get("top_rare_words", [])[:10]
    word_rows = []
    for w in words:
        diff = w.get("difficulty", "")
        color = _DIFFICULTY_COLORS.get(diff, "#888")
        ctx = html.escape(w.get("context", ""))
        trans = html.escape(w.get("translation", ""))
        word_rows.append(
            f"<tr>"
            f'<td class="vocab-word">{html.escape(w["word"])}</td>'
            f'<td><span class="vocab-badge" '
            f'style="background:{color}">{diff}</span></td>'
            f'<td class="vocab-ctx">{ctx}</td>'
            f'<td class="vocab-trans">{trans}</td>'
            f"</tr>"
        )

    words_table = ""
    if word_rows:
        words_table = (
            '<table class="vocab-table">'
            "<thead><tr><th>Word</th><th></th>"
            "<th>Context</th><th>Translation</th></tr></thead>"
            f'<tbody>{"".join(word_rows)}</tbody></table>'
        )

    return (
        "<article>"
        '<div class="section-label">'
        '<i data-lucide="book-open"></i> Vocabulary</div>'
        '<div class="vocab-stats">'
        f'<span class="vocab-stat">'
        f"<strong>{unique:,}</strong> unique lemmas</span>"
        f'<span class="vocab-stat">Difficulty '
        f'<strong style="color:{_DIFFICULTY_COLORS.get(estimated, "#888")}">'
        f"{estimated}</strong></span>"
        "</div>"
        f'<div class="vocab-bar">{"".join(bar_parts)}</div>'
        f'<div class="vocab-legend">{"".join(legend_parts)}</div>'
        f"{words_table}"
        "</article>"
    )


def _find_sibling_workspaces(workspace: Path, base_dir: Path) -> list[Path]:
    """Find other workspaces for the same source video."""
    meta = _load_metadata(workspace)
    source_url = meta.get("source_url", "")
    if not source_url:
        return []

    siblings: list[Path] = []
    for slug_dir in base_dir.iterdir():
        if not slug_dir.is_dir() or slug_dir.name.startswith("."):
            continue
        for ts_dir in slug_dir.iterdir():
            if not ts_dir.is_dir() or ts_dir == workspace:
                continue
            if not re.match(r"\d{8}_\d{6}$", ts_dir.name):
                continue
            other_meta = _load_metadata(ts_dir)
            if other_meta.get("source_url") == source_url:
                siblings.append(ts_dir)
    return siblings


def _build_html(
    workspace: Path,
    video_path: Path | None,
    url_prefix: str = "",
    sibling_paths: list[Path] | None = None,
    library_url: str = "",
) -> str:
    """Generate the player HTML for a workspace."""
    tracks = _discover_tracks(workspace, sibling_paths)
    meta = _load_metadata(workspace)
    title = meta.get("title") or workspace.parent.name

    track_tags = []
    for i, t in enumerate(tracks):
        default = " default" if i == 0 else ""
        tag = (
            f'<track kind="subtitles" src="{url_prefix}/{t["file"]}" '
            f'srclang="{t["lang"]}" label="{t["label"]}"{default}>'
        )
        track_tags.append(tag)

    if video_path is not None:
        video_filename = f"{url_prefix}/{video_path.name}"
        video_mime = _MIME_MAP.get(video_path.suffix.lower(), "video/mp4")
        video_section = (
            f'<video id="player" controls autoplay>\n'
            f'    <source src="{video_filename}" type="{video_mime}">\n'
            f"    {chr(10).join(track_tags)}\n"
            f"  </video>"
        )
    else:
        source_url = meta.get("source_url", "")
        thumbnail = meta.get("thumbnail", "")
        if source_url:
            redownload_btn = (
                '<button class="redownload-btn outline" onclick="redownload()">'
                '<i data-lucide="download"></i> Re-download video'
                "</button>"
            )
        else:
            redownload_btn = ""
        if thumbnail:
            backdrop = (
                f'<div class="vm-backdrop">'
                f'<img src="{html.escape(thumbnail)}" alt="">'
                f"</div>"
            )
        else:
            backdrop = ""
        video_section = (
            f'<div class="video-missing">'
            f"{backdrop}"
            f'<div class="vm-content">'
            f'<i data-lucide="video-off"></i>'
            f"<p><strong>Video not available</strong></p>"
            f"<p>The source file may have been moved or cleaned from cache.</p>"
            f"{redownload_btn}"
            f"</div>"
            f"</div>"
        )

    icon_src = f"{url_prefix}/icon.png"
    if library_url:
        brand = (
            f'<p><a href="{html.escape(library_url)}" class="brand-link">'
            f'<i data-lucide="arrow-left" class="brand-back"></i>'
            f'<img src="{icon_src}" class="brand-logo" alt="">'
            f"PolyglotWhisperer</a></p>"
        )
    else:
        brand = f'<p><img src="{icon_src}" class="brand-logo" alt="">' f"PolyglotWhisperer</p>"

    return _PLAYER_TEMPLATE.format(
        title=html.escape(title),
        base_url=url_prefix,
        brand=brand,
        video_section=video_section,
        metadata_rows=_build_metadata_rows(meta),
        vocab_section=_build_vocab_section(workspace),
        download_rows=_build_download_rows(workspace, meta, url_prefix, sibling_paths),
        github_url=GITHUB_URL,
    )


def _merge_workspaces(workspaces: list[dict]) -> list[dict]:
    """Merge workspaces for the same source video into a single entry."""
    groups: dict[str, list[dict]] = {}
    no_url: list[dict] = []

    for ws in workspaces:
        meta = _load_metadata(ws["path"])
        source_url = meta.get("source_url", "") if meta else ""

        if not source_url:
            no_url.append(ws)
            continue

        groups.setdefault(source_url, []).append(ws)

    merged: list[dict] = []
    for _url, group in groups.items():

        def _score(w: dict) -> tuple:
            has_video = bool(w.get("has_video"))
            file_count = sum(1 for f in w["path"].iterdir() if not f.name.startswith("."))
            return (has_video, file_count, w["timestamp"])

        group.sort(key=_score, reverse=True)
        primary = group[0]

        lang_pairs: list[tuple[str, str]] = []
        sibling_paths: list[Path] = []
        for ws in group:
            lang = ws.get("language", "")
            target = ws.get("target_language", "")
            pair = (lang, target)
            if pair not in lang_pairs:
                lang_pairs.append(pair)
            if ws["path"] != primary["path"]:
                sibling_paths.append(ws["path"])

        translated_sources = {src for src, tgt in lang_pairs if tgt}
        lang_pairs = [p for p in lang_pairs if p[1] or p[0] not in translated_sources]

        primary["lang_pairs"] = lang_pairs
        primary["sibling_paths"] = sibling_paths
        merged.append(primary)

    for ws in no_url:
        lang = ws.get("language", "")
        target = ws.get("target_language", "")
        ws["lang_pairs"] = [(lang, target)]
        ws["sibling_paths"] = []

    return merged + no_url


def _discover_workspaces(base_dir: Path, backfill_metadata: bool = True) -> list[dict]:
    """Find all workspaces under base_dir, return metadata + paths."""
    workspaces = []
    if not base_dir.is_dir():
        return workspaces
    needs_backfill: list[Path] = []
    for slug_dir in sorted(base_dir.iterdir()):
        if not slug_dir.is_dir() or slug_dir.name.startswith("."):
            continue
        for ts_dir in sorted(slug_dir.iterdir(), reverse=True):
            if not ts_dir.is_dir() or not re.match(r"\d{8}_\d{6}$", ts_dir.name):
                continue
            meta = _load_metadata(ts_dir)
            if not meta:
                continue
            if (
                backfill_metadata
                and meta.get("source_url")
                and not all(meta.get(k) for k in _METADATA_FIELDS)
            ):
                needs_backfill.append(ts_dir)
            lang = meta.get("language", "")
            target = meta.get("target_language", "")

            # Peek at vocabulary file for difficulty estimate
            difficulty = ""
            for vocab_file in sorted(ts_dir.glob(GLOB_VOCABULARY_JSON)):
                try:
                    vdata = json.loads(vocab_file.read_text(encoding="utf-8"))
                    difficulty = vdata.get("estimated_difficulty", "")
                except (json.JSONDecodeError, OSError):
                    pass
                break

            workspaces.append(
                {
                    "path": ts_dir,
                    "slug": slug_dir.name,
                    "timestamp": ts_dir.name,
                    "title": meta.get("title", slug_dir.name),
                    "language": lang,
                    "target_language": target,
                    "duration": meta.get("source_duration"),
                    "created_at": meta.get("created_at", ""),
                    "has_video": find_video(ts_dir) is not None,
                    "upload_date": meta.get("upload_date", ""),
                    "uploader": meta.get("uploader", ""),
                    "thumbnail": meta.get("thumbnail", ""),
                    "difficulty": difficulty,
                }
            )

    workspaces = _merge_workspaces(workspaces)

    if needs_backfill:
        import threading

        def _backfill() -> None:
            for ws_path in needs_backfill:
                try:
                    _refresh_metadata(ws_path)
                except Exception as exc:
                    console.print(
                        f"[dim red]Metadata refresh failed for {ws_path.name}: {exc}[/dim red]"
                    )

        threading.Thread(target=_backfill, daemon=True).start()
        console.print(
            f"[dim]Refreshing metadata for {len(needs_backfill)} workspace(s) in background…[/dim]"
        )

    return workspaces


def _build_library_html(workspaces: list[dict]) -> str:
    """Generate the library HTML listing all workspaces."""
    if not workspaces:
        content = (
            '<div class="ws-empty">'
            '<i data-lucide="folder-open"></i>'
            "<p>No workspaces found.</p>"
            "<p>Run <code>pgw run</code> to process a video first.</p>"
            "</div>"
        )
    else:
        cards = []
        for ws in workspaces:
            title = html.escape(ws["title"])
            slug = html.escape(ws["slug"])
            ts = html.escape(ws["timestamp"])
            icon = "video" if ws["has_video"] else "file-audio"
            thumb = ws.get("thumbnail", "")

            meta_chips = []
            lang_pairs = ws.get("lang_pairs", [])
            if lang_pairs:
                lang_labels = []
                for lang, target in lang_pairs:
                    if lang and target:
                        lang_labels.append(f"{_lang_short(lang)} &rarr; {_lang_short(target)}")
                    elif lang:
                        lang_labels.append(_lang_short(lang))
                if lang_labels:
                    joined = " · ".join(lang_labels)
                    meta_chips.append(f'<span><i data-lucide="languages"></i>{joined}</span>')

            difficulty = ws.get("difficulty", "")
            if difficulty:
                color = _DIFFICULTY_COLORS.get(difficulty, "#888")
                meta_chips.append(
                    f'<span class="difficulty-chip" style="background:{color}">'
                    f"{html.escape(difficulty)}</span>"
                )

            uploader = ws.get("uploader", "")
            if uploader:
                meta_chips.append(
                    f'<span><i data-lucide="user"></i>' f"{html.escape(uploader)}</span>"
                )

            dur = _format_duration(ws["duration"])
            if dur:
                meta_chips.append(f'<span><i data-lucide="clock"></i>' f"{html.escape(dur)}</span>")

            upload_date = ws.get("upload_date", "")
            if upload_date and len(upload_date) == 8:
                formatted = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
                meta_chips.append(
                    f'<span><i data-lucide="calendar-plus"></i>' f"{html.escape(formatted)}</span>"
                )
            else:
                created = ws["created_at"]
                if created:
                    meta_chips.append(
                        f'<span><i data-lucide="calendar"></i>'
                        f"{html.escape(created[:10])}</span>"
                    )

            meta_html = "\n".join(meta_chips)
            if thumb:
                thumb_html = (
                    f'<div class="ws-thumb">'
                    f'<img src="{html.escape(thumb)}" alt="" loading="lazy">'
                    f"</div>"
                )
            else:
                thumb_html = (
                    f'<div class="ws-thumb ws-thumb-placeholder">'
                    f'<i data-lucide="{icon}"></i>'
                    f"</div>"
                )
            cards.append(
                f'<a href="/ws/{slug}/{ts}/" class="ws-card">'
                f"<article>"
                f"{thumb_html}"
                f'<div class="ws-card-body">'
                f"<h3>{title}</h3>"
                f'<p class="ws-meta">{meta_html}</p>'
                f"</div>"
                f"</article></a>"
            )
        content = f'<div class="ws-grid">{"".join(cards)}</div>'

    return _LIBRARY_TEMPLATE.format(
        content=content,
        github_url=GITHUB_URL,
    )


def _update_workspace_meta(workspace: Path, source: object) -> None:
    """Backfill metadata.json with fields from a VideoSource (or info dict)."""
    meta_path = workspace / METADATA_FILE
    if not meta_path.is_file():
        return
    import json as _json

    existing = _json.loads(meta_path.read_text(encoding="utf-8"))
    changed = False
    for key in _METADATA_FIELDS:
        val = getattr(source, key, None) if hasattr(source, key) else source.get(key)
        if val and not existing.get(key):
            existing[key] = val
            changed = True
    for src_key, meta_key in (("title", "title"), ("duration", "source_duration")):
        val = getattr(source, src_key, None) if hasattr(source, src_key) else source.get(src_key)
        if val and not existing.get(meta_key):
            existing[meta_key] = val
            changed = True
    if changed:
        meta_path.write_text(
            _json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _refresh_metadata(workspace: Path) -> bool:
    """Fetch metadata from source URL without downloading the video."""
    meta = _load_metadata(workspace)
    source_url = meta.get("source_url")
    if not source_url:
        return False

    if all(meta.get(k) for k in _METADATA_FIELDS):
        return False

    try:
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(source_url, download=False)

        from types import SimpleNamespace

        source = SimpleNamespace(
            upload_date=info.get("upload_date"),
            uploader=info.get("channel") or info.get("uploader"),
            thumbnail=info.get("thumbnail"),
            description=info.get("description"),
            title=info.get("title"),
            duration=info.get("duration"),
        )
        _update_workspace_meta(workspace, source)
        return True
    except Exception as exc:
        console.print(f"[dim red]Metadata refresh failed: {exc}[/dim red]")
        return False


def _redownload_video_streaming(workspace: Path, send_event: typing.Callable[[str], None]) -> bool:
    """Re-download video with progress events streamed via send_event."""
    meta = _load_metadata(workspace)
    source_url = meta.get("source_url")
    if not source_url:
        send_event(json.dumps({"progress": 0, "status": "error", "detail": "No source URL"}))
        return False

    try:
        import yt_dlp

        from pgw.core.config import load_config
        from pgw.utils.cache import link_or_copy

        config = load_config()
        output_dir = Path(config.download_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fmt = config.download.format

        def progress_hook(d: dict) -> None:
            if d["status"] == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded = d.get("downloaded_bytes", 0)
                pct = (downloaded / total * 100) if total > 0 else 0
                speed = d.get("speed")
                eta = d.get("eta")
                detail_parts = []
                if speed:
                    if speed > BYTES_PER_MB:
                        detail_parts.append(f"{speed / BYTES_PER_MB:.1f} MB/s")
                    else:
                        detail_parts.append(f"{speed / BYTES_PER_KB:.0f} KB/s")
                if eta is not None:
                    m, s = divmod(int(eta), 60)
                    detail_parts.append(f"ETA {m}:{s:02d}")
                send_event(
                    json.dumps(
                        {
                            "progress": round(pct, 1),
                            "status": "downloading",
                            "detail": " · ".join(detail_parts),
                        }
                    )
                )
            elif d["status"] == "finished":
                send_event(
                    json.dumps(
                        {
                            "progress": 100,
                            "status": "processing",
                            "detail": "Merging formats…",
                        }
                    )
                )

        opts = {
            "format": fmt,
            "outtmpl": str(output_dir / "%(title)s_%(id)s.%(ext)s"),
            "progress_hooks": [progress_hook],
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }

        send_event(json.dumps({"progress": 0, "status": "starting", "detail": "Connecting…"}))

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(source_url, download=True)
            video_path = Path(ydl.prepare_filename(info))

        if not video_path.is_file():
            candidates = sorted(
                (p for p in output_dir.iterdir() if not p.name.startswith(".")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                video_path = candidates[0]
            else:
                msg = {"progress": 0, "status": "error", "detail": "No file found"}
                send_event(json.dumps(msg))
                return False

        video_ext = video_path.suffix or _DEFAULT_VIDEO_EXT
        video_dest = workspace / f"video{video_ext}"
        if video_dest.is_symlink():
            video_dest.unlink()
        link_or_copy(video_path, video_dest)

        from types import SimpleNamespace

        source = SimpleNamespace(
            upload_date=info.get("upload_date"),
            uploader=info.get("channel") or info.get("uploader"),
            thumbnail=info.get("thumbnail"),
            description=info.get("description"),
            title=info.get("title"),
            duration=info.get("duration"),
        )
        _update_workspace_meta(workspace, source)

        send_event(json.dumps({"progress": 100, "status": "done", "detail": "Complete"}))
        return True
    except Exception as e:
        send_event(json.dumps({"progress": 0, "status": "error", "detail": str(e)}))
        return False
