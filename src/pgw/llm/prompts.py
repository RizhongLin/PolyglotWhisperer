"""Prompt templates for subtitle cleanup and translation."""

import json

# Prefix for segments where translation failed or was missing
UNTRANSLATED_MARKER = "[?] "

CLEANUP_SYSTEM = """\
You are a subtitle editor. Your task is to clean up automatic speech recognition (ASR) \
output while preserving the original meaning and timing structure.

Rules:
- First check if each line actually contains errors before modifying it. \
If a line is already correct, return it unchanged.
- Fix spelling and grammar errors from ASR mistakes
- Remove filler words (euh, hum, ah, um) unless they carry meaning
- Normalize punctuation (proper sentence endings, quotation marks)
- Do NOT merge or split segments — return the EXACT same number of lines
- Do NOT translate — keep the original language
- Do NOT add information that wasn't in the original
- Return ONLY the cleaned lines, numbered exactly as the input

Examples:
Input:
1. euh bonjour comment allez vous
2. je suis tres content de vous voire
3. merci beaucoup pour votre aide

Output:
1. Bonjour, comment allez-vous ?
2. Je suis très content de vous voir.
3. Merci beaucoup pour votre aide.
"""

CLEANUP_USER = """\
Clean up these {count} subtitle segments in {language}. \
Return exactly {count} numbered lines, one per input line.

{numbered_segments}
"""

TRANSLATION_SYSTEM = """\
You are a professional subtitle translator. Translate subtitle segments \
into natural, idiomatic {target_lang} — not word-for-word from {source_lang}.

Rules:
- Each key in the input MUST have exactly one translation in the output — \
do NOT merge or split segments. Every key ("1", "2", ...) must appear in the output.
- Translate ONLY the segments between the ===BEGIN=== and ===END=== markers
- Consecutive segments are part of the same speech and a sentence may span multiple segments. \
Keep each segment's core meaning roughly aligned with its source (subtitles are timed to audio), \
but always ensure the translated segments read coherently and fluently when joined together
- Use natural {target_lang} grammar, word order, and phrasing \
— avoid mimicking {source_lang} structure
- Keep proper nouns (names, places, brands) in their original form
- Keep translations concise — suitable for subtitle display
- Do NOT add extra text, explanations, or commentary

Example — a sentence split across two segments:
Input:
{{"1": "Le président a annoncé une nouvelle taxe", \
"2": "sur les importations en provenance d'Asie."}}
Good output (1:1 mapping, each key preserved):
{{"1": "The president announced a new tax", \
"2": "on imports from Asia."}}
Bad output (key "2" merged into "1"):
{{"1": "The president announced a new tax on imports from Asia.", \
"2": ""}}

Output format — return a JSON object with the SAME numbered keys as the input:
{{"1": "translation 1", "2": "translation 2", ...}}
"""

TRANSLATION_USER = """\
Translate these {count} segments from {source_lang} to {target_lang}. \
Return a JSON object with keys "1" through "{count}", each mapped to its translation.

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
    """Format subtitle texts as a JSON object with numbered string keys.

    Output: {"1": "text one", "2": "text two", ...}
    Collapses newlines within each text to spaces.
    """
    segments = {str(i + 1): " ".join(text.split()) for i, text in enumerate(texts)}
    return json.dumps(segments, ensure_ascii=False)


def format_history_context(
    source_texts: list[str],
    translated_texts: list[str],
) -> str:
    """Format previously translated pairs as reference context.

    Provides the LLM with its own recent translation style and terminology
    to maintain consistency across batches.
    """
    if not source_texts or not translated_texts:
        return ""
    pairs = []
    for src, tgt in zip(source_texts, translated_texts):
        pairs.append(f"  {src} → {tgt}")
    return (
        "Previous translations for style reference (do NOT re-translate these):\n"
        + "\n".join(pairs)
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
        # Only accept lines with explicit numbering
        for sep in [". ", ") ", ": "]:
            parts = line.split(sep, 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                parsed.append(parts[1].strip())
                break
        # Skip non-numbered lines entirely

    return _normalize_parsed(parsed, expected_count)


def parse_json_response(response: str, expected_count: int) -> tuple[list[str], bool]:
    """Try to parse a JSON object response from the LLM.

    Supports two formats:
    1. Keyed: {"1": "t1", "2": "t2", ...} — preferred, forces 1:1 mapping
    2. Array: {"translations": ["t1", "t2", ...]} — legacy fallback

    Falls back gracefully — returns ([], False) if response is not valid JSON.
    """
    text = response.strip()
    # Strip markdown code fences if present
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

    # Try keyed format first: {"1": "t1", "2": "t2", ...}
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

    # Fallback: array format {"translations": ["t1", "t2", ...]}
    translations = None
    for key in ("translations", "translated", "results"):
        if key in data and isinstance(data[key], list):
            translations = data[key]
            break

    if translations is None:
        return [], False

    parsed = [str(item).strip() for item in translations]
    return _normalize_parsed(parsed, expected_count)


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
    """Format bilingual pairs as overlap context.

    Shows how preceding/following segments were translated, giving
    the LLM concrete examples of boundary translations.
    """
    if not source_texts:
        return ""
    parts = []
    for src, tgt in zip(source_texts, translated_texts):
        parts.append(f"[{label}] {src} -> {tgt}")
    return "\n".join(parts)
