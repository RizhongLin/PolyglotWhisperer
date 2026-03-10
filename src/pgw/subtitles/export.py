"""Parallel text and vocabulary export — PDF/EPUB generation."""

from __future__ import annotations

import hashlib
import html
from pathlib import Path

from pgw.core.languages import language_name
from pgw.core.models import SubtitleSegment
from pgw.utils.text import SECONDS_PER_MINUTE

# Segments per EPUB chapter — keeps chapter size manageable for e-readers
EPUB_SEGMENTS_PER_CHAPTER = 100


def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS for display."""
    m = int(seconds // SECONDS_PER_MINUTE)
    s = int(seconds % SECONDS_PER_MINUTE)
    return f"{m:02d}:{s:02d}"


def _lang_label(code: str) -> str:
    """Format a language code as 'Name (code)' for display."""
    name = language_name(code).title()
    if name.lower() == code.lower():
        return name
    return f"{name} ({code})"


def _build_table_row(orig: SubtitleSegment, trans: SubtitleSegment) -> str:
    """Build a single HTML table row for a parallel segment pair."""
    ts = _format_timestamp(orig.start)
    return (
        f'<tr><td class="ts">{ts}</td>'
        f'<td class="orig">{html.escape(orig.text)}</td>'
        f'<td class="trans">{html.escape(trans.text)}</td></tr>'
    )


# ── PDF CSS (WeasyPrint — full CSS support including @page) ──

_PDF_CSS = """\
@page {
  size: A4;
  margin: 2cm 1.8cm;
  @bottom-center {
    content: counter(page);
    font-size: 8pt;
    color: #aaa;
    font-family: system-ui, -apple-system, sans-serif;
  }
}

body {
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 10pt;
  color: #1a1a1a;
  line-height: 1.6;
}

.header {
  margin-bottom: 1.8em;
  padding-bottom: 1em;
  border-bottom: 2px solid #e0e0e0;
}

.header h1 {
  font-size: 18pt;
  font-weight: 600;
  margin: 0 0 0.3em 0;
  color: #111;
}

.header .subtitle {
  font-size: 10pt;
  color: #666;
  letter-spacing: 0.04em;
}

.header .meta {
  font-size: 8.5pt;
  color: #999;
  margin-top: 0.3em;
}

table {
  width: 100%;
  border-collapse: collapse;
}

thead th {
  text-align: left;
  font-size: 8pt;
  font-weight: 600;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 6px 10px;
  border-bottom: 1.5px solid #ccc;
}

tbody tr:nth-child(even) {
  background: #f8f9fa;
}

td {
  padding: 5px 10px;
  vertical-align: top;
  font-size: 10pt;
}

.ts {
  width: 40px;
  color: #aaa;
  font-size: 8pt;
  font-family: "SF Mono", "Cascadia Code", "Consolas", monospace;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  padding-top: 7px;
}

.orig {
  width: 45%;
  color: #1a1a1a;
}

.trans {
  width: 45%;
  color: #444;
}
"""


# ── EPUB CSS (conservative — wide e-reader compatibility) ──

_EPUB_CSS = b"""\
body {
  font-family: system-ui, Georgia, serif;
  font-size: 1em;
  line-height: 1.6;
  color: #1a1a1a;
}

h2 {
  font-size: 1.1em;
  font-weight: 600;
  color: #555;
  border-bottom: 1px solid #ddd;
  padding-bottom: 0.4em;
  margin-top: 1.5em;
  margin-bottom: 1em;
}

table {
  width: 100%;
  border-collapse: collapse;
}

tr:nth-child(even) {
  background: #f5f5f5;
}

td {
  padding: 0.3em 0.5em;
  vertical-align: top;
  font-size: 0.95em;
}

.ts {
  width: 3em;
  color: #999;
  font-size: 0.8em;
  font-family: monospace;
  white-space: nowrap;
  padding-top: 0.4em;
}

.orig {
  width: 45%;
  color: #1a1a1a;
}

.trans {
  width: 45%;
  color: #444;
}
"""


def build_parallel_html(
    original: list[SubtitleSegment],
    translated: list[SubtitleSegment],
    source_lang: str,
    target_lang: str,
    title: str,
    css: str = "",
) -> str:
    """Build HTML for parallel text layout."""
    if not css:
        css = _PDF_CSS

    source_label = _lang_label(source_lang)
    target_label = _lang_label(target_lang)

    table_rows = "\n".join(
        _build_table_row(orig, trans) for orig, trans in zip(original, translated, strict=True)
    )
    return f"""\
<!DOCTYPE html>
<html lang="{source_lang}">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
{css}
</style>
</head>
<body>
<div class="header">
  <h1>{html.escape(title)}</h1>
  <div class="subtitle">{source_label} &rarr; {target_label}</div>
  <div class="meta">{len(original)} segments &middot; PolyglotWhisperer</div>
</div>
<table>
<thead><tr>
  <th></th>
  <th>{source_label}</th>
  <th>{target_label}</th>
</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</body>
</html>"""


def export_parallel_pdf(
    original: list[SubtitleSegment],
    translated: list[SubtitleSegment],
    output_path: Path,
    source_lang: str,
    target_lang: str,
    title: str = "",
) -> Path:
    """Generate a side-by-side PDF using WeasyPrint.

    Each segment pair is a table row: timestamp | original | translation.

    Args:
        original: Original subtitle segments.
        translated: Translated subtitle segments (same length).
        output_path: Where to write the PDF.
        source_lang: Source language code.
        target_lang: Target language code.
        title: Document title.

    Returns:
        Path to the generated PDF.
    """
    try:
        from weasyprint import HTML
    except ImportError:
        raise ImportError("weasyprint is not installed. Install with: uv sync --extra export")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    title = title or f"Parallel Text ({source_lang}-{target_lang})"
    html_content = build_parallel_html(
        original, translated, source_lang, target_lang, title, css=_PDF_CSS
    )
    HTML(string=html_content).write_pdf(str(output_path))

    return output_path


def export_parallel_epub(
    original: list[SubtitleSegment],
    translated: list[SubtitleSegment],
    output_path: Path,
    source_lang: str,
    target_lang: str,
    title: str = "",
) -> Path:
    """Generate a parallel-text EPUB using ebooklib.

    Splits into chapters of ~100 segments for e-reader performance.
    Side-by-side layout via HTML table in XHTML chapters.

    Args:
        original: Original subtitle segments.
        translated: Translated subtitle segments (same length).
        output_path: Where to write the EPUB.
        source_lang: Source language code.
        target_lang: Target language code.
        title: Document title.

    Returns:
        Path to the generated EPUB.
    """
    try:
        from ebooklib import epub
    except ImportError:
        raise ImportError("ebooklib is not installed. Install with: uv sync --extra export")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    title = title or f"Parallel Text ({source_lang}-{target_lang})"
    source_label = _lang_label(source_lang)
    target_label = _lang_label(target_lang)

    book = epub.EpubBook()
    content_hash = hashlib.sha256(title.encode() + b"\0" + source_lang.encode()).hexdigest()[:12]
    book.set_identifier(f"pgw-{source_lang}-{target_lang}-{content_hash}")
    book.set_title(title)
    book.set_language(source_lang)

    style = epub.EpubItem(
        uid="style",
        file_name="style/main.css",
        media_type="text/css",
        content=_EPUB_CSS,
    )
    book.add_item(style)

    # Title page
    title_page = epub.EpubHtml(
        title=title,
        file_name="title.xhtml",
        lang=source_lang,
    )
    title_page.content = (
        f'<div style="text-align:center; padding-top:3em;">'
        f"<h1>{html.escape(title)}</h1>"
        f'<p style="color:#666; font-size:1.1em;">'
        f"{source_label} &rarr; {target_label}</p>"
        f'<p style="color:#999; font-size:0.9em;">'
        f"{len(original)} segments</p>"
        f"</div>"
    ).encode("utf-8")
    title_page.add_item(style)
    book.add_item(title_page)

    # Content chapters
    chapters = []
    for ch_idx in range(0, len(original), EPUB_SEGMENTS_PER_CHAPTER):
        ch_orig = original[ch_idx : ch_idx + EPUB_SEGMENTS_PER_CHAPTER]
        ch_trans = translated[ch_idx : ch_idx + EPUB_SEGMENTS_PER_CHAPTER]

        start_ts = _format_timestamp(ch_orig[0].start)
        end_ts = _format_timestamp(ch_orig[-1].end)
        ch_title = f"{start_ts} \u2014 {end_ts}"

        rows = [
            _build_table_row(orig, trans) for orig, trans in zip(ch_orig, ch_trans, strict=True)
        ]

        chapter = epub.EpubHtml(
            title=ch_title,
            file_name=f"ch_{ch_idx // EPUB_SEGMENTS_PER_CHAPTER:03d}.xhtml",
            lang=source_lang,
        )
        chapter.content = (
            f"<h2>{html.escape(ch_title)}</h2>" f'<table>{"".join(rows)}</table>'
        ).encode("utf-8")
        chapter.add_item(style)
        book.add_item(chapter)
        chapters.append(chapter)

    book.toc = [title_page, *chapters]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", title_page, *chapters]

    epub.write_epub(str(output_path), book, {})
    return output_path


# ── Vocabulary export ──

_CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]

_CEFR_COLORS = {
    "A1": "#2e7d32",
    "A2": "#558b2f",
    "B1": "#f57f17",
    "B2": "#e65100",
    "C1": "#c62828",
    "C2": "#6a1b9a",
}

_VOCAB_PDF_CSS = """\
@page {
  size: A4;
  margin: 2cm 1.8cm;
  @bottom-center {
    content: counter(page);
    font-size: 8pt;
    color: #aaa;
    font-family: system-ui, -apple-system, sans-serif;
  }
}

body {
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 10pt;
  color: #1a1a1a;
  line-height: 1.6;
}

.header {
  margin-bottom: 1.8em;
  padding-bottom: 1em;
  border-bottom: 2px solid #e0e0e0;
}

.header h1 {
  font-size: 18pt;
  font-weight: 600;
  margin: 0 0 0.3em 0;
  color: #111;
}

.header .subtitle {
  font-size: 10pt;
  color: #666;
  letter-spacing: 0.04em;
}

.header .meta {
  font-size: 8.5pt;
  color: #999;
  margin-top: 0.3em;
}

.stats-grid {
  display: flex;
  gap: 1.5em;
  margin-bottom: 1.5em;
}

.stat-box {
  text-align: center;
  padding: 0.6em 1.2em;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  background: #fafafa;
}

.stat-box .value {
  font-size: 16pt;
  font-weight: 700;
  color: #111;
}

.stat-box .label {
  font-size: 7.5pt;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #888;
}

.cefr-bar {
  display: flex;
  height: 10px;
  border-radius: 5px;
  overflow: hidden;
  margin-bottom: 0.4em;
}

.cefr-bar span {
  display: inline-block;
  height: 100%;
}

.cefr-legend {
  display: flex;
  gap: 1em;
  font-size: 7.5pt;
  color: #666;
  margin-bottom: 2em;
}

.cefr-legend .dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 3px;
  vertical-align: middle;
}

h2.level-heading {
  font-size: 12pt;
  font-weight: 600;
  margin: 1.8em 0 0.6em 0;
  padding-bottom: 0.3em;
  border-bottom: 1.5px solid #ccc;
}

.level-badge {
  display: inline-block;
  padding: 0.1em 0.5em;
  border-radius: 4px;
  color: white;
  font-size: 9pt;
  font-weight: 700;
  margin-right: 0.4em;
  vertical-align: middle;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 1em;
}

thead th {
  text-align: left;
  font-size: 8pt;
  font-weight: 600;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 6px 10px;
  border-bottom: 1.5px solid #ccc;
}

tbody tr:nth-child(even) {
  background: #f8f9fa;
}

td {
  padding: 5px 10px;
  vertical-align: top;
  font-size: 10pt;
}

td.word {
  font-weight: 600;
  white-space: nowrap;
}

td.pos {
  font-size: 8pt;
  color: #888;
  font-family: "SF Mono", "Cascadia Code", "Consolas", monospace;
  white-space: nowrap;
}

td.freq {
  font-size: 8pt;
  color: #aaa;
  font-family: "SF Mono", "Cascadia Code", "Consolas", monospace;
  text-align: right;
  white-space: nowrap;
}

td.context {
  color: #444;
  font-size: 9pt;
  font-style: italic;
}

td.translation {
  color: #666;
  font-size: 9pt;
}
"""

_VOCAB_EPUB_CSS = b"""\
body {
  font-family: system-ui, Georgia, serif;
  font-size: 1em;
  line-height: 1.6;
  color: #1a1a1a;
}

h2 {
  font-size: 1.1em;
  font-weight: 600;
  color: #555;
  border-bottom: 1px solid #ddd;
  padding-bottom: 0.4em;
  margin-top: 1.5em;
  margin-bottom: 1em;
}

.level-badge {
  display: inline-block;
  padding: 0.1em 0.4em;
  border-radius: 3px;
  color: white;
  font-weight: 700;
  margin-right: 0.3em;
}

table {
  width: 100%;
  border-collapse: collapse;
}

tr:nth-child(even) {
  background: #f5f5f5;
}

td {
  padding: 0.3em 0.5em;
  vertical-align: top;
  font-size: 0.95em;
}

td.word {
  font-weight: 600;
  white-space: nowrap;
}

td.pos {
  font-size: 0.8em;
  color: #999;
  font-family: monospace;
}

td.context {
  color: #444;
  font-style: italic;
  font-size: 0.9em;
}

td.translation {
  color: #666;
  font-size: 0.9em;
}

.summary {
  margin-bottom: 1.5em;
  color: #555;
  font-size: 0.95em;
}
"""


def _group_words_by_cefr(words: list[dict]) -> dict[str, list[dict]]:
    """Group word entries by CEFR level, preserving only non-empty levels."""
    groups: dict[str, list[dict]] = {}
    for w in words:
        level = w.get("cefr", "C2")
        groups.setdefault(level, []).append(w)
    return groups


def _build_vocab_word_row(w: dict) -> str:
    """Build a single HTML table row for a vocabulary word."""
    context = html.escape(w.get("context", ""))
    translation = html.escape(w.get("translation", ""))
    return (
        f'<tr><td class="word">{html.escape(w["word"])}</td>'
        f'<td class="pos">{html.escape(w["pos"])}</td>'
        f'<td class="freq">{w["zipf"]:.1f}</td>'
        f'<td class="context">{context}</td>'
        f'<td class="translation">{translation}</td></tr>'
    )


def build_vocab_html(summary: dict, title: str = "", css: str = "") -> str:
    """Build HTML for vocabulary summary layout."""
    if not css:
        css = _VOCAB_PDF_CSS

    language = summary.get("language", "")
    lang_label = _lang_label(language)
    title = title or f"Vocabulary \u2014 {lang_label}"

    total = summary.get("total_words", 0)
    unique = summary.get("unique_lemmas", 0)
    estimated = summary.get("estimated_level", "?")
    dist = summary.get("cefr_distribution", {})
    total_types = sum(dist.values()) or 1

    # CEFR bar segments
    bar_parts = []
    legend_parts = []
    for level in _CEFR_ORDER:
        count = dist.get(level, 0)
        if count == 0:
            continue
        pct = count / total_types * 100
        color = _CEFR_COLORS[level]
        bar_parts.append(f'<span style="width:{pct:.1f}%;background:{color}"></span>')
        legend_parts.append(
            f'<span><span class="dot" style="background:{color}"></span>'
            f"{level} ({count})</span>"
        )

    bar_html = "".join(bar_parts)
    legend_html = "".join(legend_parts)

    # Word tables grouped by CEFR level (rarest first: C2, C1, ...)
    words = summary.get("top_rare_words", [])
    groups = _group_words_by_cefr(words)
    tables_html = []
    for level in reversed(_CEFR_ORDER):
        group = groups.get(level)
        if not group:
            continue
        color = _CEFR_COLORS[level]
        rows = "\n".join(_build_vocab_word_row(w) for w in group)
        tables_html.append(
            f'<h2 class="level-heading">'
            f'<span class="level-badge" style="background:{color}">{level}</span>'
            f"{len(group)} words</h2>\n"
            f"<table>\n<thead><tr>"
            f"<th>Word</th><th>POS</th><th>Zipf</th>"
            f"<th>Context</th><th>Translation</th>"
            f"</tr></thead>\n<tbody>\n{rows}\n</tbody></table>"
        )

    def _stat(value: str, label: str) -> str:
        return (
            f'<div class="stat-box">'
            f'<div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>'
        )

    stats = "\n  ".join(
        [
            _stat(f"{total:,}", "Total tokens"),
            _stat(f"{unique:,}", "Unique lemmas"),
            _stat(estimated, "Estimated level"),
        ]
    )

    return f"""\
<!DOCTYPE html>
<html lang="{language}">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
{css}
</style>
</head>
<body>
<div class="header">
  <h1>{html.escape(title)}</h1>
  <div class="subtitle">{lang_label}</div>
  <div class="meta">{len(words)} rare words &middot; PolyglotWhisperer</div>
</div>
<div class="stats-grid">
  {stats}
</div>
<div class="cefr-bar">{bar_html}</div>
<div class="cefr-legend">{legend_html}</div>
{"".join(tables_html)}
</body>
</html>"""


def export_vocab_pdf(
    summary: dict,
    output_path: Path,
    title: str = "",
) -> Path:
    """Generate a vocabulary summary PDF using WeasyPrint."""
    try:
        from weasyprint import HTML
    except ImportError:
        raise ImportError("weasyprint is not installed. Install with: uv sync --extra export")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_content = build_vocab_html(summary, title=title, css=_VOCAB_PDF_CSS)
    HTML(string=html_content).write_pdf(str(output_path))
    return output_path


def export_vocab_epub(
    summary: dict,
    output_path: Path,
    title: str = "",
) -> Path:
    """Generate a vocabulary summary EPUB using ebooklib."""
    try:
        from ebooklib import epub
    except ImportError:
        raise ImportError("ebooklib is not installed. Install with: uv sync --extra export")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    language = summary.get("language", "en")
    lang_label = _lang_label(language)
    title = title or f"Vocabulary \u2014 {lang_label}"

    total = summary.get("total_words", 0)
    unique = summary.get("unique_lemmas", 0)
    estimated = summary.get("estimated_level", "?")

    book = epub.EpubBook()
    content_hash = hashlib.sha256(
        title.encode() + b"\0" + language.encode() + b"vocab"
    ).hexdigest()[:12]
    book.set_identifier(f"pgw-vocab-{language}-{content_hash}")
    book.set_title(title)
    book.set_language(language)

    style = epub.EpubItem(
        uid="style",
        file_name="style/main.css",
        media_type="text/css",
        content=_VOCAB_EPUB_CSS,
    )
    book.add_item(style)

    # Title page
    title_page = epub.EpubHtml(
        title=title,
        file_name="title.xhtml",
        lang=language,
    )
    title_page.content = (
        f'<div style="text-align:center; padding-top:3em;">'
        f"<h1>{html.escape(title)}</h1>"
        f'<p style="color:#666; font-size:1.1em;">{lang_label}</p>'
        f'<p style="color:#999; font-size:0.9em;">'
        f"{total:,} tokens &middot; {unique:,} lemmas &middot; {estimated}</p>"
        f"</div>"
    ).encode("utf-8")
    title_page.add_item(style)
    book.add_item(title_page)

    # One chapter per CEFR level (rarest first)
    words = summary.get("top_rare_words", [])
    groups = _group_words_by_cefr(words)
    chapters = []
    for level in reversed(_CEFR_ORDER):
        group = groups.get(level)
        if not group:
            continue

        color = _CEFR_COLORS[level]
        rows = "\n".join(_build_vocab_word_row(w) for w in group)

        chapter = epub.EpubHtml(
            title=f"{level} ({len(group)} words)",
            file_name=f"level_{level.lower()}.xhtml",
            lang=language,
        )
        chapter.content = (
            f'<h2><span class="level-badge" style="background:{color}">'
            f"{level}</span> {len(group)} words</h2>"
            f"<table><thead><tr>"
            f"<th>Word</th><th>POS</th>"
            f"<th>Context</th><th>Translation</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
        ).encode("utf-8")
        chapter.add_item(style)
        book.add_item(chapter)
        chapters.append(chapter)

    book.toc = [title_page, *chapters]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", title_page, *chapters]

    epub.write_epub(str(output_path), book, {})
    return output_path
