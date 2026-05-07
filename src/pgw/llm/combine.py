"""Single-pass LLM refinement + translation.

When both ``--refine`` and ``--translate`` are active, this module replaces
the two sequential LLM calls (refine then translate) with one call per
chunk.  The LLM receives the raw transcription and produces *both* the
refined original and the translation in one response.  This halves API
latency / token cost and gives the model full visibility into the
source→target mapping so idiomatic choices can consider the raw signal.

Output format — nested keyed JSON with one key per segment:
``{"1": {"refined": "...", "translated": "..."}, "2": {...}}``
"""

from __future__ import annotations

from typing import Callable

from pgw.core.config import LLMConfig
from pgw.core.models import SubtitleSegment
from pgw.llm.chunking import (
    find_chunk_boundaries,
    resolve_chunk_params,
)
from pgw.llm.client import complete
from pgw.llm.prompts import (
    format_json_segments,
    parse_json_response,
)
from pgw.llm.translator import TranslationResult
from pgw.utils.text import find_sentence_split

_HISTORY_SIZE = 8
_SCAN_RANGE = 5  # How far to scan for sentence boundaries around ideal split point

# ── Chunk size presets ──────────────────────────────────────────────────
# The combined task is at least as heavy as translation alone, so use
# the translator's defaults.
_COMBINE_API_DEFAULT = 150
_COMBINE_LOCAL_CAP = 64
_COMBINE_MIN_OVERLAP = 6
_COMBINE_MIN_BACK_OVERLAP = 4


def _chunk_params(config: LLMConfig, chunk_size: int | None) -> tuple[int, int, int]:
    """Resolve chunk / overlap / back_overlap for combined refine+translate."""
    return resolve_chunk_params(
        config,
        chunk_size,
        api_default=_COMBINE_API_DEFAULT,
        local_cap=_COMBINE_LOCAL_CAP,
        min_overlap=_COMBINE_MIN_OVERLAP,
        min_back_overlap=_COMBINE_MIN_BACK_OVERLAP,
    )


# ── Prompt ──────────────────────────────────────────────────────────────

_COMBINE_SYSTEM = """\
You are a bilingual subtitle editor and translator. For each segment you must
perform TWO tasks simultaneously:

────────────────────────────────────────────────────────────
1. REFINE (correct ASR errors — source language only)
────────────────────────────────────────────────────────────

Strict 1:1 mapping (CRITICAL):
- The input is a JSON object with NUMBERED string keys. The output MUST have
  the SAME keys ("1" through "N") with no keys added, dropped, merged, or
  split. These are timed subtitles — merging segments breaks audio sync.

What to fix:
- ASR transcription errors (misheard words, homophone mistakes)
- Missing or incorrect accents and diacritics (e.g. "tres" → "très")
- Missing or incorrect punctuation (sentence endings, commas, hyphens)
- Filler words and hesitations (euh, hum, ah, um) — remove unless meaningful
- Capitalization errors (sentence starts, proper nouns)
- Missing hyphens in compound expressions (e.g. "peut être" → "peut-être")
- Use surrounding segment context to resolve ambiguous ASR errors
  (e.g. "mer" vs "mère" vs "maire" depends on adjacent words)

What NOT to do in refine:
- Do NOT translate — keep the original language exactly
- Do NOT rephrase, paraphrase, or rewrite the speaker's wording
- Do NOT add information, context, or explanation
- If a line is already correct, return it unchanged

────────────────────────────────────────────────────────────
2. TRANSLATE (into natural, idiomatic {target_lang})
────────────────────────────────────────────────────────────

Cross-segment coherence:
- Consecutive segments are often parts of the same sentence. Read them
  together — both the refined source AND the translation context — to
  understand the full meaning before translating.
- Each translated segment must be a natural, meaningful phrase in
  {target_lang} — never a dangling fragment like a lone word, preposition,
  or adjective that makes no sense on its own.
- When a sentence spans multiple segments, redistribute the meaning so
  each segment reads as a complete phrase in {target_lang}. The split
  points need NOT match the source — adapt to {target_lang} grammar.

Other rules:
- Use natural {target_lang} grammar, word order, and phrasing
- Keep proper nouns (names, places, brands) in their original form
- Preserve the speaker's register, tone, and cultural references
- Keep translations concise — suitable for subtitle display
- Do NOT add extra text, explanations, or commentary

Examples — same 3-segment French input, two target languages:

Source: "1": "L'armée a abattu deux avions"
        "2": "en provenance d'Iran"
        "3": "au-dessus de la capitale."

English (similar grammar — split points align naturally):
→ "1": {{"refined": "L'armée a abattu deux avions", "translated": "The army shot down two planes"}}
   "2": {{"refined": "en provenance d'Iran", "translated": "coming from Iran"}}
   "3": {{"refined": "au-dessus de la capitale.", "translated": "over the capital."}}

Chinese (different grammar — redistributes "飞机" from key 1 to key 2):
→ "1": {{"refined": "L'armée a abattu deux avions", "translated": "军队击落了两架"}}
   "2": {{"refined": "en provenance d'Iran", "translated": "来自伊朗的飞机，"}}
   "3": {{"refined": "au-dessus de la capitale.", "translated": "就在首都上空。"}}

────────────────────────────────────────────────────────────
OUTPUT FORMAT
────────────────────────────────────────────────────────────

Return a JSON object keyed by segment number where each value is itself an
object with "refined" and "translated" keys:

{{"1": {{"refined": "corrected source text", "translated": "{target_lang} translation"}},
 "2": {{"refined": "...", "translated": "..."}},
 ...
}}

- Include ALL segments in the input. For noise- or music-only segments
  (e.g. "[music]", crowd noise, mic feedback) set both "refined" and
  "translated" to "" — the goal is matching key count, not non-empty content.
- Empty output should be RARE — only use it for genuine noise.
  Do NOT produce long runs of consecutive empty entries. Even brief or
  partial speech deserves a best-effort refine and translation.
- Segment-level integrity: every input key MUST produce exactly one
  output key — never combine two or more input segments into a single
  output, no matter how naturally they form a sentence. This is
  non-negotiable timing data.
- Keys MUST be sequential strings matching the input keys exactly.
- Do NOT add, skip, merge, or reorder segments."""

_COMBINE_USER = """\
Refine (fix ASR errors) then translate these {count} segments \
from {source_lang} to {target_lang}. Return a JSON object with \
EXACTLY the keys "1" through "{count}" — every key must appear, \
none may be added, merged, split, or skipped. Use empty strings \
"" for non-speech segments.

{context}===BEGIN===
{segments_json}
===END===
"""


def _build_combine_schema(count: int) -> dict:
    """JSON schema for the nested keyed format."""

    def _make_item_schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "refined": {"type": "string"},
                "translated": {"type": "string"},
            },
            "required": ["refined", "translated"],
            "additionalProperties": False,
        }

    properties: dict[str, dict] = {}
    required: list[str] = []
    for i in range(1, count + 1):
        key = str(i)
        properties[key] = _make_item_schema()
        required.append(key)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "combined_segments",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


# ── Combined call for one chunk ─────────────────────────────────────────


def _process_chunk(
    texts: list[str],
    language: str,
    target_lang: str,
    config: LLMConfig,
    context: str = "",
    _depth: int = 0,
) -> list[tuple[str, str]]:
    """Process one chunk: refine + translate → list of (refined, translated)."""
    MAX_DEPTH = 3

    schema = _build_combine_schema(len(texts))
    segments_json = format_json_segments(
        [SubtitleSegment(text=t, start=0.0, end=0.0) for t in texts]
    )

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": _COMBINE_SYSTEM.format(count=len(texts), target_lang=target_lang),
        },
        {
            "role": "user",
            "content": _COMBINE_USER.format(
                source_lang=language,
                target_lang=target_lang,
                segments_json=segments_json,
                context=context,
                count=len(texts),
            ),
        },
    ]

    raw = complete(messages, config, json_schema=schema, expected_count=len(texts))
    parsed = parse_json_response(raw)

    # Extract refined + translated per segment
    items: list[tuple[str, str]] = []
    for i in range(1, len(texts) + 1):
        entry = parsed.get(str(i))
        if isinstance(entry, dict):
            items.append((str(entry.get("refined", "")), str(entry.get("translated", ""))))
        else:
            items.append(("", ""))

    # Retry on count mismatch
    if len(items) != len(texts) and len(texts) > 2 and _depth < MAX_DEPTH:
        correction = (
            f"You returned {len(items) if items else len(parsed)} items "
            f"but I need exactly {len(texts)}. "
            f'Please return a JSON object with keys "1" through "{len(texts)}" '
            f'each containing {{"refined": "...", "translated": "..."}}.'
        )
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": correction})
        retry_raw = complete(messages, config, json_schema=schema, expected_count=len(texts))
        retry_parsed = parse_json_response(retry_raw)
        items = []
        for i in range(1, len(texts) + 1):
            entry = retry_parsed.get(str(i))
            if isinstance(entry, dict):
                items.append((str(entry.get("refined", "")), str(entry.get("translated", ""))))
            else:
                items.append(("", ""))

    # Binary split on persistent mismatch
    if len(items) != len(texts) and _depth < MAX_DEPTH:
        if len(texts) <= 2:
            return [(t, "") for t in texts]

        mid = find_sentence_split(texts)
        if mid <= 0 or mid >= len(texts):
            return [(t, "") for t in texts]

        first_half = _process_chunk(texts[:mid], language, target_lang, config, context, _depth + 1)

        # Enriched context for second half: preceding refined+translated pairs
        second_context = context
        prev = first_half[-min(2, len(first_half)) :]
        for r, t in prev:
            second_context += f"\n[preceding] refined: {r}\n[preceding] translated: {t}"

        second_half = _process_chunk(
            texts[mid:], language, target_lang, config, second_context, _depth + 1
        )
        items = first_half + second_half

    return items


# ── Public API ──────────────────────────────────────────────────────────


def refine_and_translate(
    segments: list[SubtitleSegment],
    source_lang: str,
    target_lang: str,
    config: LLMConfig,
    chunk_size: int | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> TranslationResult:
    """Refine and translate subtitles in a single LLM pass per chunk.

    Combines what would normally be two sequential LLM calls (refine then
    translate) into one, halving latency and token cost.  The LLM receives
    the raw transcription and emits both the corrected original and the
    translation for every segment.

    Returns:
        TranslationResult with ``.original`` (refined) and ``.translated``
        segments preserving all original timestamps.
    """
    chunk_size, overlap, back_overlap = _chunk_params(config, chunk_size)

    refined_segs: list[SubtitleSegment] = []
    translated_segs: list[SubtitleSegment] = []

    boundaries = find_chunk_boundaries(
        segments, chunk_size, overlap=overlap, scan_range=_SCAN_RANGE
    )
    total_chunks = len(boundaries)

    for chunk_idx, keep_start in enumerate(boundaries):
        keep_end = boundaries[chunk_idx + 1] if chunk_idx + 1 < len(boundaries) else len(segments)

        # Backward overlap for non-first chunks
        if chunk_idx > 0:
            translate_start = max(0, keep_start - back_overlap)
        else:
            translate_start = keep_start

        # Forward overlap for lookahead
        translate_end = min(keep_end + overlap, len(segments))
        chunk = segments[translate_start:translate_end]

        # Build bilingual context
        context_parts: list[str] = []

        # Preceding refined+translated pairs as read-only context
        if translate_start > 0:
            before_start = max(0, translate_start - _HISTORY_SIZE)
            pairs = []
            for j in range(before_start, translate_start):
                if j < len(refined_segs) and j < len(translated_segs):
                    pairs.append(
                        f"[preceding] src: {refined_segs[j].text}\n"
                        f"[preceding] tgt: {translated_segs[j].text}"
                    )
            if pairs:
                context_parts.insert(
                    0,
                    "Previously refined + translated (for reference only):\n" + "\n".join(pairs),
                )

        # Following segments as read-only source context
        follow_start = translate_end
        follow_end = min(follow_start + overlap, len(segments))
        if follow_start < follow_end:
            following = segments[follow_start:follow_end]
            after_lines = [f"[following] {seg.text}" for seg in following]
            context_parts.append(
                "Following segments (for reference only):\n" + "\n".join(after_lines)
            )

        context = "\n".join(context_parts) + "\n" if context_parts else ""

        texts = [seg.text for seg in chunk]

        pairs = _process_chunk(texts, source_lang, target_lang, config, context=context)

        # Extract keep region (discard lookahead-only segments)
        keep_offset = keep_start - translate_start
        keep_count = keep_end - keep_start
        keep_pairs = pairs[keep_offset : keep_offset + keep_count]

        for i, (ref_text, trans_text) in enumerate(keep_pairs):
            seg_idx = keep_start + i
            orig = segments[seg_idx]

            if chunk_idx > 0 and seg_idx < len(refined_segs):
                # Backward overlap — overwrite with newly processed result
                refined_segs[seg_idx] = SubtitleSegment(
                    text=ref_text or orig.text,
                    start=orig.start,
                    end=orig.end,
                    speaker=orig.speaker,
                )
                translated_segs[seg_idx] = SubtitleSegment(
                    text=trans_text,
                    start=orig.start,
                    end=orig.end,
                    speaker=orig.speaker,
                )
            else:
                refined_segs.append(
                    SubtitleSegment(
                        text=ref_text or orig.text,
                        start=orig.start,
                        end=orig.end,
                        speaker=orig.speaker,
                    )
                )
                translated_segs.append(
                    SubtitleSegment(
                        text=trans_text,
                        start=orig.start,
                        end=orig.end,
                        speaker=orig.speaker,
                    )
                )

        if on_progress:
            on_progress((chunk_idx + 1) / total_chunks)

    return TranslationResult(
        original=refined_segs,
        translated=translated_segs,
        source_language=source_lang,
        target_language=target_lang,
    )
