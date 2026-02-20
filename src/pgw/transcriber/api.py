"""API transcription backend via LiteLLM.

Uses litellm.transcription() to call cloud Whisper APIs (OpenAI, Groq, etc.)
and regroups word-level timestamps into subtitle segments — no stable-ts
dependency required.
"""

from __future__ import annotations

from pathlib import Path

from pgw.core.config import WhisperConfig
from pgw.core.models import SubtitleSegment
from pgw.utils.console import console

# 25 MB limit for OpenAI/Groq Whisper API
_MAX_FILE_SIZE = 25 * 1024 * 1024


def transcribe(audio_path: Path, config: WhisperConfig) -> list[SubtitleSegment]:
    """Transcribe audio via a cloud Whisper API.

    Args:
        audio_path: Path to audio file (WAV recommended).
        config: Whisper configuration with model set to a LiteLLM model string.

    Returns:
        List of subtitle segments with timestamps.

    Raises:
        ImportError: If litellm is not installed.
        ValueError: If file exceeds 25 MB API limit.
    """
    try:
        import litellm
    except ImportError:
        raise ImportError("litellm is not installed. Install with: uv sync --extra llm")

    audio_path = Path(audio_path)
    file_size = audio_path.stat().st_size
    if file_size > _MAX_FILE_SIZE:
        size_mb = file_size / (1024 * 1024)
        raise ValueError(
            f"Audio file is {size_mb:.1f} MB, exceeding the 25 MB API limit. "
            "Use --start/--duration to clip, or switch to --backend local."
        )

    console.print(f"[bold]Transcribing via API:[/bold] {config.model}")

    # Drop unsupported top-level params; pass timestamp_granularities via
    # extra_body so it reaches providers that support it (e.g. Groq, OpenAI)
    litellm.drop_params = True

    call_kwargs: dict = {
        "model": config.model,
        "response_format": "verbose_json",
        "language": config.language,
        "extra_body": {"timestamp_granularities": ["word"]},
    }
    if config.api_base:
        call_kwargs["api_base"] = config.api_base

    with open(audio_path, "rb") as f:
        response = litellm.transcription(file=f, **call_kwargs)

    segments = _response_to_segments(response)
    console.print(f"[green]Transcription complete:[/green] {len(segments)} segments")
    return segments


def _response_to_segments(response) -> list[SubtitleSegment]:
    """Convert a LiteLLM transcription response to SubtitleSegments.

    Prefers word-level timestamps for subtitle-optimized regrouping.
    Falls back to segment-level timestamps if words are missing.
    """
    # LiteLLM returns a TranscriptionResponse; extract the inner dict
    data = response.model_dump() if hasattr(response, "model_dump") else response

    # Try word-level timestamps first
    words = data.get("words") or []
    if words:
        return _regroup_words(words)

    # Fallback: use segment-level timestamps
    raw_segments = data.get("segments") or []
    if raw_segments:
        return [
            SubtitleSegment(
                text=_get(seg, "text", "").strip(),
                start=float(_get(seg, "start", 0)),
                end=float(_get(seg, "end", 0)),
            )
            for seg in raw_segments
            if _get(seg, "text", "").strip()
        ]

    # Last resort: full text as single segment
    text = data.get("text", "").strip()
    if text:
        return [SubtitleSegment(text=text, start=0.0, end=0.0)]
    return []


def _regroup_words(
    words: list[dict],
    max_chars: int = 50,
    max_dur: float = 8.0,
) -> list[SubtitleSegment]:
    """Regroup word-level timestamps into subtitle-sized segments.

    Mirrors the logic of regroup_for_subtitles() in postprocess.py but
    operates on plain word dicts — no stable-ts dependency.

    Split rules (in priority order):
    1. Sentence punctuation (. ? ! 。 ？ ！)
    2. Speech gaps > 0.5s
    3. Comma/semicolon if segment has 4+ words
    4. Max chars or max duration exceeded

    Merge rule:
    - Short fragments (< 3 words) merged if gap < 0.15s
    """
    if not words:
        return []

    _SENTENCE_PUNCT = set(".?!。？！")
    _CLAUSE_PUNCT = set(",;，；")

    # Build initial 1-word-per-segment list
    raw: list[dict] = []
    for w in words:
        text = _get(w, "word", "").strip()
        if not text:
            continue
        raw.append(
            {
                "text": text,
                "start": float(_get(w, "start", 0)),
                "end": float(_get(w, "end", 0)),
            }
        )

    if not raw:
        return []

    # Phase 1: Merge all into one, then split by sentence punctuation
    groups: list[list[dict]] = [[raw[0]]]
    for w in raw[1:]:
        prev_text = groups[-1][-1]["text"]
        if prev_text and prev_text[-1] in _SENTENCE_PUNCT:
            groups.append([w])
        else:
            groups[-1].append(w)

    # Phase 2: Split by speech gaps > 0.5s
    split = []
    for group in groups:
        current = [group[0]]
        for w in group[1:]:
            gap = w["start"] - current[-1]["end"]
            if gap > 0.5:
                split.append(current)
                current = [w]
            else:
                current.append(w)
        split.append(current)
    groups = split

    # Phase 3: Split at comma/semicolon if segment has 4+ words
    split = []
    for group in groups:
        current = [group[0]]
        for w in group[1:]:
            prev_text = current[-1]["text"]
            if len(current) >= 4 and prev_text and prev_text[-1] in _CLAUSE_PUNCT:
                split.append(current)
                current = [w]
            else:
                current.append(w)
        split.append(current)
    groups = split

    # Phase 4: Split by max_chars and max_dur
    split = []
    for group in groups:
        current = [group[0]]
        cur_text_len = len(group[0]["text"])
        for w in group[1:]:
            new_len = cur_text_len + 1 + len(w["text"])  # +1 for space
            cur_dur = w["end"] - current[0]["start"]
            if new_len > max_chars or cur_dur > max_dur:
                split.append(current)
                current = [w]
                cur_text_len = len(w["text"])
            else:
                current.append(w)
                cur_text_len = new_len
        split.append(current)
    groups = split

    # Phase 5: Merge short fragments (< 3 words) if gap < 0.15s
    merged: list[list[dict]] = [groups[0]]
    for group in groups[1:]:
        prev = merged[-1]
        gap = group[0]["start"] - prev[-1]["end"]
        combined_words = len(prev) + len(group)
        combined_text = " ".join(w["text"] for w in prev + group)
        if (
            len(prev) < 3
            and gap < 0.15
            and combined_words <= 10
            and len(combined_text) <= max_chars
        ):
            merged[-1].extend(group)
        else:
            merged.append(group)
    groups = merged

    # Convert groups to SubtitleSegments
    segments = []
    for group in groups:
        text = " ".join(w["text"] for w in group)
        segments.append(
            SubtitleSegment(
                text=text,
                start=group[0]["start"],
                end=group[-1]["end"],
            )
        )

    return segments


def _get(obj, key: str, default=None):
    """Get a value from a dict or object attribute."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
