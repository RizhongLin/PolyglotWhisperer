"""Parallel text export — side-by-side original + translation in PDF/EPUB."""

from __future__ import annotations

import hashlib
import html
from pathlib import Path

from pgw.core.models import SubtitleSegment


def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS for display."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _build_parallel_html(
    original: list[SubtitleSegment],
    translated: list[SubtitleSegment],
    source_lang: str,
    target_lang: str,
    title: str,
) -> str:
    """Build HTML for parallel text layout."""
    rows = []
    for orig, trans in zip(original, translated, strict=True):
        ts = _format_timestamp(orig.start)
        orig_text = html.escape(orig.text)
        trans_text = html.escape(trans.text)
        rows.append(
            f'<tr><td class="ts">{ts}</td>'
            f'<td class="orig">{orig_text}</td>'
            f'<td class="trans">{trans_text}</td></tr>'
        )

    table_rows = "\n".join(rows)
    return f"""\
<!DOCTYPE html>
<html lang="{source_lang}">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  @page {{ size: A4; margin: 2cm; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; font-size: 11pt;
         color: #222; line-height: 1.5; }}
  h1 {{ font-size: 16pt; margin-bottom: 0.5em; }}
  .meta {{ color: #666; font-size: 9pt; margin-bottom: 1.5em; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: left; font-size: 9pt; color: #666; padding: 4px 8px;
       border-bottom: 2px solid #333; }}
  td {{ padding: 4px 8px; vertical-align: top; border-bottom: 1px solid #eee; }}
  .ts {{ width: 50px; color: #999; font-size: 9pt; white-space: nowrap; }}
  .orig {{ width: 45%; }}
  .trans {{ width: 45%; color: #444; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="meta">{source_lang.upper()} / {target_lang.upper()} &mdash; \
{len(original)} segments</p>
<table>
<thead><tr>
  <th></th>
  <th>{source_lang.upper()}</th>
  <th>{target_lang.upper()}</th>
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
    html_content = _build_parallel_html(original, translated, source_lang, target_lang, title)
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
    book = epub.EpubBook()
    content_hash = hashlib.sha256(title.encode() + b"\0" + source_lang.encode()).hexdigest()[:12]
    book.set_identifier(f"pgw-{source_lang}-{target_lang}-{content_hash}")
    book.set_title(title)
    book.set_language(source_lang)

    style = epub.EpubItem(
        uid="style",
        file_name="style/main.css",
        media_type="text/css",
        content=b"""\
body { font-family: system-ui, sans-serif; font-size: 11pt; line-height: 1.5; }
table { width: 100%; border-collapse: collapse; }
td { padding: 4px 8px; vertical-align: top; border-bottom: 1px solid #eee; }
.ts { width: 50px; color: #999; font-size: 9pt; }
.orig { width: 45%; }
.trans { width: 45%; color: #555; }
""",
    )
    book.add_item(style)

    # Split into chapters
    chunk_size = 100
    chapters = []
    for ch_idx in range(0, len(original), chunk_size):
        ch_orig = original[ch_idx : ch_idx + chunk_size]
        ch_trans = translated[ch_idx : ch_idx + chunk_size]

        start_ts = _format_timestamp(ch_orig[0].start)
        end_ts = _format_timestamp(ch_orig[-1].end)
        ch_title = f"{start_ts} — {end_ts}"

        rows = []
        for orig, trans in zip(ch_orig, ch_trans, strict=True):
            ts = _format_timestamp(orig.start)
            rows.append(
                f'<tr><td class="ts">{ts}</td>'
                f'<td class="orig">{html.escape(orig.text)}</td>'
                f'<td class="trans">{html.escape(trans.text)}</td></tr>'
            )

        chapter = epub.EpubHtml(
            title=ch_title,
            file_name=f"ch_{ch_idx // chunk_size:03d}.xhtml",
            lang=source_lang,
        )
        chapter.content = (
            f"<h2>{html.escape(ch_title)}</h2>" f'<table>{"".join(rows)}</table>'
        ).encode("utf-8")
        chapter.add_item(style)
        book.add_item(chapter)
        chapters.append(chapter)

    book.toc = chapters
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *chapters]

    epub.write_epub(str(output_path), book, {})
    return output_path
