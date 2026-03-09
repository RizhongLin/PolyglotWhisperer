"""Parallel text export — side-by-side original + translation in PDF/EPUB."""

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
