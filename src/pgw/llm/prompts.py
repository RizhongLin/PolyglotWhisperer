"""Prompt templates for subtitle refinement and translation."""

import json

# Prefix for segments where translation failed or was missing
UNTRANSLATED_MARKER = "[?] "


# ── JSON schema builders ──


def _build_keyed_schema(count: int, name: str) -> dict:
    """Strict JSON schema requiring keys ``"1"`` through ``"N"``, all strings.

    Keyed dicts give the model a per-item anchor it must echo back, so a
    drop or merge is structurally visible to the model itself rather than
    only via a length count. Works in three tiers:

    - ``json_schema`` (strict): each key required → hard guarantee.
    - ``json_object``: prompt + format keep the model honest.
    - ``none``: same prompt; parser handles via ``parse_json_response``.
    """
    keys = [str(i) for i in range(1, count + 1)]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {k: {"type": "string"} for k in keys},
                "required": keys,
                "additionalProperties": False,
            },
        },
    }


def build_refine_schema(count: int) -> dict:
    """Build a strict JSON schema for refinement output with {count} items."""
    return _build_keyed_schema(count, "refinement")


def build_translation_schema(count: int) -> dict:
    """Build a strict JSON schema for translation output with {count} items."""
    return _build_keyed_schema(count, "translation")


CLASSIFY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "difficulty_classify",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "level": {
                                "type": "string",
                                "enum": ["A1", "A2", "B1", "B2", "C1", "C2"],
                            },
                        },
                        "required": ["word", "level"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
    },
}


REFINE_SYSTEM = """\
You are a professional subtitle editor. Your task is to refine automatic \
speech recognition (ASR) output into broadcast-quality subtitles.

Strict 1:1 mapping (CRITICAL):
- The input is a JSON object with NUMBERED string keys ("1", "2", ...). \
The output MUST be a JSON object with the SAME keys ("1" through "N"), in \
the same order, with no keys added, dropped, merged, or split.
- These are timed subtitles — each segment is synced to audio. \
Merging or splitting segments would break the timing.
- An empty string ("") is acceptable for a noise-only line (e.g. "[music]", \
crowd noise, mic feedback) — the goal is matching count, not non-empty content. \
Prefer "" over deleting the key.

What to fix:
- ASR transcription errors (misheard words, homophone mistakes)
- Missing or incorrect accents and diacritics (e.g. "tres" → "très")
- Missing or incorrect punctuation (sentence endings, commas, hyphens)
- Filler words and hesitations (euh, hum, ah, um) — remove unless \
they carry meaning or are part of a quote
- Capitalization errors (sentence starts, proper nouns)
- Missing hyphens in compound expressions (e.g. "peut être" → "peut-être")

What NOT to do:
- Do NOT translate — keep the original language
- Do NOT rephrase, paraphrase, or rewrite — preserve the speaker's words
- Do NOT add information, context, or explanation
- Do NOT merge or split segments
- If a line is already correct, return it unchanged

Cross-segment awareness:
- Consecutive segments are often parts of the same sentence. \
Read them together to understand context before correcting.
- Use context to resolve ambiguous ASR errors \
(e.g. "mer" vs "mère" vs "maire" depends on surrounding words).

Output format — a JSON object keyed by segment number:
{{"1": "text 1", "2": "text 2", ...}}

Example:
Input: {{"1": "euh bonjour comment allez vous", \
"2": "je suis tres content de vous voire", \
"3": "merci beaucoup pour votre aide"}}
Output: {{"1": "Bonjour, comment allez-vous ?", \
"2": "Je suis très content de vous voir.", \
"3": "Merci beaucoup pour votre aide."}}
"""

REFINE_USER = """\
Refine these {count} subtitle segments in {language}. \
Return a JSON object with EXACTLY the keys "1" through "{count}" — \
every key must appear, none may be added, merged, or skipped. \
Use an empty string "" if a line is pure noise.

{context}===BEGIN===
{json_segments}
===END===
"""

TRANSLATION_SYSTEM = """\
You are a professional subtitle translator. Translate subtitle segments \
into natural, idiomatic {target_lang} — not word-for-word from {source_lang}.

Strict 1:1 mapping (CRITICAL):
- The input is a JSON object with NUMBERED string keys ("1", "2", ...). \
The output MUST be a JSON object with the SAME keys ("1" through "N"), in \
the same order, with no keys added, dropped, merged, or split.
- These are timed subtitles — each segment is synced to audio. \
Merging segments would break the timing.
- An empty string ("") is acceptable for a noise-only segment (e.g. "[music]", \
crowd noise, mic feedback, untranslatable filler) — the goal is matching the \
key count, not non-empty content. Prefer "" over dropping the key.

Cross-segment coherence:
- Consecutive segments are often parts of the same sentence. \
Read them together to understand the full meaning before translating.
- Each translated segment must be a natural, meaningful phrase \
in {target_lang} — never a dangling fragment like a lone word, \
preposition, or adjective that makes no sense on its own.
- When a sentence spans segments, redistribute the meaning so each \
segment reads well on its own. The split point in {target_lang} need \
not match {source_lang} — adapt it to {target_lang} grammar.

Other rules:
- Translate ONLY the segments between the ===BEGIN=== and ===END=== markers
- Use natural {target_lang} grammar, word order, and phrasing
- Keep proper nouns (names, places, brands) in their original form
- Keep translations concise — suitable for subtitle display
- Do NOT add extra text, explanations, or commentary

Example 1 — similar languages (split points align naturally):
Input: {{"1": "L'armée a abattu deux avions", \
"2": "en provenance d'Iran", \
"3": "au-dessus de la capitale."}}
Good: {{"1": "The army shot down two planes", \
"2": "coming from Iran", \
"3": "over the capital."}}
Bad (merged into key 1, drops 2 and 3): \
{{"1": "The army shot down two planes from Iran over the capital."}}

Example 2 — distant languages (split points SHIFT to fit target grammar):
Same input → Chinese:
Good (meaning redistributed across the SAME 3 keys): \
{{"1": "军队击落了两架", \
"2": "来自伊朗的飞机，", \
"3": "就在首都上空。"}}
Bad (word-for-word, dangling modifier): \
{{"1": "军队击落了两架飞机", \
"2": "来自伊朗的", \
"3": "在首都上空。"}}
Note: "飞机" (planes) moved from segment 1 to 2 so each segment is a \
complete phrase — but the key count stays at 3.

Output format — a JSON object keyed by segment number:
{{"1": "text 1", "2": "text 2", ...}}
"""

TRANSLATION_USER = """\
Translate these {count} segments from {source_lang} to {target_lang}. \
Return a JSON object with EXACTLY the keys "1" through "{count}" — \
every key must appear, none may be added, merged, or skipped. \
Use an empty string "" if a segment is pure noise.

{context}===BEGIN===
{json_segments}
===END===
"""


def format_numbered_segments(texts: list[str]) -> str:
    """Format a list of subtitle texts as numbered lines for LLM input.

    Collapses newlines within each text to spaces so the model sees
    exactly one line per numbered item (subtitle line wraps are visual only).
    """
    return "\n".join(f"{i + 1}. {' '.join(text.split())}" for i, text in enumerate(texts))


def format_json_segments(texts: list[str]) -> str:
    """Format subtitle texts as a numbered-key JSON object.

    Output: {"1": "text one", "2": "text two", ...}

    Numbered keys give the model a per-item anchor to echo back, making
    silent merges or drops structurally visible — the parser can detect
    a missing "47" key, whereas a length-N array can shrink invisibly.
    Collapses newlines within each text to spaces so each value is one
    line (subtitle line wraps are visual only).
    """
    items = {str(i + 1): " ".join(text.split()) for i, text in enumerate(texts)}
    return json.dumps(items, ensure_ascii=False)


def format_history_context(
    source_texts: list[str],
    translated_texts: list[str],
) -> str:
    """Format previously translated pairs as JSON reference context.

    Provides the LLM with its own recent translation style and terminology
    to maintain consistency across batches.
    """
    if not source_texts or not translated_texts:
        return ""
    pairs = [
        {" ".join(src.split()): " ".join(tgt.split())}
        for src, tgt in zip(source_texts, translated_texts)
    ]
    return (
        "Previous translations for style reference (do NOT re-translate these):\n"
        + json.dumps(pairs, ensure_ascii=False)
        + "\n\n"
    )


def _normalize_parsed(parsed: list[str], expected_count: int) -> tuple[list[str], bool]:
    """Truncate or pad a parsed list to expected_count, returning exact_match flag."""
    exact_match = len(parsed) == expected_count
    if len(parsed) > expected_count:
        parsed = parsed[:expected_count]
    while len(parsed) < expected_count:
        parsed.append("")
    return parsed, exact_match


def parse_numbered_response(response: str, expected_count: int) -> tuple[list[str], bool]:
    """Parse a numbered LLM response back into a list of texts.

    Only extracts lines with explicit numbering (1. / 1) / 1:).
    Non-numbered lines (explanations, context echoes) are ignored.

    Returns:
        Tuple of (parsed texts, exact_match) where exact_match is True
        if the parsed count matches expected_count exactly.
    """
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]

    parsed = []
    for line in lines:
        for sep in [". ", ") ", ": "]:
            parts = line.split(sep, 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                parsed.append(parts[1].strip())
                break

    return _normalize_parsed(parsed, expected_count)


def parse_json_response(response: str, expected_count: int) -> tuple[list[str], bool]:
    """Parse a JSON object response from the LLM.

    Preferred: numbered-key format ``{"1": "t1", "2": "t2", ...}`` —
    matches what the prompt asks for and gives the parser per-key parity.
    Fallback: array-wrapped format ``{"translations": [...]}`` /
    ``{"refined": [...]}`` for older models that ignore the key spec.
    """
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return [], False

    if not isinstance(data, dict):
        return [], False

    # Preferred: keyed format {"1": "t1", "2": "t2", ...}
    numeric_keys: dict[int, str] = {}
    for k, v in data.items():
        try:
            numeric_keys[int(k)] = str(v).strip()
        except (ValueError, TypeError):
            pass
    if numeric_keys:
        max_key = max(numeric_keys)
        parsed = [numeric_keys.get(i + 1, "") for i in range(max_key)]
        return _normalize_parsed(parsed, expected_count)

    # Fallback: array format {"translations": [...]} / {"refined": [...]}
    for key in ("translations", "refined", "translated", "results"):
        if key in data and isinstance(data[key], list):
            parsed = [str(item).strip() for item in data[key]]
            return _normalize_parsed(parsed, expected_count)

    return [], False


def filter_empty_segments(texts: list[str]) -> tuple[list[int], list[str]]:
    """Filter out empty/whitespace-only texts for LLM processing.

    Returns:
        Tuple of (non-empty indices, non-empty texts).
    """
    non_empty_idx = [j for j, t in enumerate(texts) if t.strip()]
    non_empty_texts = [texts[j] for j in non_empty_idx]
    return non_empty_idx, non_empty_texts


def reconstruct_with_empties(
    texts: list[str],
    non_empty_idx: list[int],
    result_texts: list[str],
) -> list[str]:
    """Reconstruct full text list, restoring empty positions with originals as fallback."""
    reconstructed = [""] * len(texts)
    for j, idx in enumerate(non_empty_idx):
        reconstructed[idx] = result_texts[j] if j < len(result_texts) else texts[idx]
    return reconstructed


def format_bilingual_context(
    source_texts: list[str],
    translated_texts: list[str],
    label: str = "preceding",
) -> str:
    """Format bilingual pairs as JSON overlap context.

    Shows how preceding/following segments were translated, giving
    the LLM concrete examples of boundary translations.
    """
    if not source_texts:
        return ""
    pairs = {
        " ".join(src.split()): " ".join(tgt.split())
        for src, tgt in zip(source_texts, translated_texts)
    }
    return f"{label}: " + json.dumps(pairs, ensure_ascii=False)
