"""Prompt templates for subtitle cleanup and translation."""

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
You are a professional subtitle translator. Translate subtitle segments accurately \
while keeping them natural and concise for on-screen reading.

Rules:
- Translate each numbered line from {source_lang} to {target_lang}
- Keep translations concise — suitable for subtitle display
- Preserve the tone and register of the original
- Handle idioms and colloquialisms naturally in the target language
- Do NOT merge or split segments — return the EXACT same number of lines
- Return ONLY the translated lines, numbered exactly as the input

Examples (fr → en):
Input:
1. Bonjour, comment allez-vous ?
2. Je suis très content de vous voir.
3. Il fait un temps magnifique aujourd'hui.

Output:
1. Hello, how are you?
2. I'm very happy to see you.
3. The weather is wonderful today.
"""

TRANSLATION_USER = """\
Translate these {count} subtitle segments from {source_lang} to {target_lang}. \
Return exactly {count} numbered lines, one per input line.

{numbered_segments}
"""


def format_numbered_segments(texts: list[str]) -> str:
    """Format a list of subtitle texts as numbered lines for LLM input."""
    return "\n".join(f"{i + 1}. {text}" for i, text in enumerate(texts))


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


def parse_numbered_response(response: str, expected_count: int) -> tuple[list[str], bool]:
    """Parse a numbered LLM response back into a list of texts.

    Handles various formats:
    - "1. Text here"
    - "1) Text here"
    - "1: Text here"
    - Plain lines (fallback)

    Returns:
        Tuple of (parsed texts, exact_match) where exact_match is True
        if the parsed count matches expected_count exactly.
    """
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]

    parsed = []
    for line in lines:
        # Try stripping common numbering patterns
        for sep in [". ", ") ", ": "]:
            parts = line.split(sep, 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                parsed.append(parts[1].strip())
                break
        else:
            # No numbering found, use the line as-is
            parsed.append(line)

    exact_match = len(parsed) == expected_count

    # If we got more lines than expected, take only the first N
    if len(parsed) > expected_count:
        parsed = parsed[:expected_count]

    # If we got fewer, pad with empty strings (will fall back to original)
    while len(parsed) < expected_count:
        parsed.append("")

    return parsed, exact_match
