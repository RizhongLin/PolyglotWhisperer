"""API transcription backend via LiteLLM.

Uses litellm.transcription() to call cloud Whisper APIs (OpenAI, Groq, etc.)
and regroups word-level timestamps into subtitle segments — no stable-ts
dependency required.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pgw.core.config import WhisperConfig
from pgw.core.models import SubtitleSegment
from pgw.utils.cache import cache_key, find_cached_file, get_cache_dir
from pgw.utils.console import debug
from pgw.utils.text import (
    CLAUSE_PUNCT,
    MAX_MERGE_LEAD_WORDS,
    MAX_MERGE_TRAIL_WORDS,
    MAX_SEGMENT_CHARS,
    MAX_SEGMENT_DURATION,
    MERGE_CHAR_SLACK,
    MERGE_GAP_THRESHOLD,
    SENTENCE_END_CHARS,
    SPEECH_GAP_THRESHOLD,
)

# 25 MB upload limit for Groq/OpenAI Whisper API
_MAX_FILE_SIZE = 25 * 1024 * 1024


def transcribe(
    audio_path: Path,
    config: WhisperConfig,
    workspace_dir: Path | None = None,
    content_hash: str | None = None,
) -> list[SubtitleSegment]:
    """Transcribe audio via a cloud Whisper API.

    Args:
        audio_path: Path to audio file (WAV recommended).
        config: Whisper configuration with model set to a LiteLLM model string.
        workspace_dir: Base workspace directory for shared cache.

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
        audio_path = _compress_for_api(audio_path, workspace_dir, content_hash=content_hash)

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

    return response_to_segments(response)


def _compress_for_api(
    audio_path: Path,
    workspace_dir: Path | None = None,
    content_hash: str | None = None,
) -> Path:
    """Compress audio to MP3 to fit within the API upload limit.

    Uses the shared cache at .cache/compressed/ when workspace_dir is provided.
    Whisper APIs accept MP3, and the quality loss at 64kbps mono is
    negligible for speech recognition.
    """
    # Check shared cache (content-based key first, metadata fallback)
    params = dict(codec="mp3", bitrate="64k")
    if workspace_dir is not None:
        cache_dir = get_cache_dir(workspace_dir, "compressed")
        hit = find_cached_file(
            cache_dir,
            ".mp3",
            content_hash=content_hash,
            file_path=audio_path,
            **params,
        )
        if hit is not None:
            return hit

    debug(f"Compressing audio: {audio_path.stat().st_size / (1024 * 1024):.1f} MB WAV → MP3")

    # Determine write path
    if workspace_dir is not None:
        if content_hash:
            write_key = cache_key(content_hash=content_hash, **params)
        else:
            write_key = cache_key(audio_path, **params)
        mp3_path = cache_dir / f"{write_key}.mp3"
    else:
        mp3_path = audio_path.with_suffix(".api.mp3")
    cmd = [
        "ffmpeg",
        "-i",
        str(audio_path),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "64k",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-y",
        str(mp3_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr_msg = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Audio compression failed: {stderr_msg}")

    new_size_mb = mp3_path.stat().st_size / (1024 * 1024)
    if mp3_path.stat().st_size > _MAX_FILE_SIZE:
        raise ValueError(
            f"Compressed audio is still {new_size_mb:.1f} MB, exceeding the 25 MB API limit. "
            "Use --start/--duration to clip, or switch to --backend local."
        )
    debug(f"Compressed: {new_size_mb:.1f} MB")
    return mp3_path


def response_to_segments(response) -> list[SubtitleSegment]:
    """Convert a LiteLLM transcription response to SubtitleSegments.

    Prefers word-level timestamps for subtitle-optimized regrouping.
    Falls back to segment-level timestamps if words are missing.
    """
    # LiteLLM returns a TranscriptionResponse; extract the inner dict
    data = response.model_dump() if hasattr(response, "model_dump") else response

    # Try word-level timestamps first
    words = data.get("words") or []
    if words:
        return regroup_words(words)

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


def regroup_words(
    words: list[dict],
    max_chars: int = MAX_SEGMENT_CHARS,
    max_dur: float = MAX_SEGMENT_DURATION,
) -> list[SubtitleSegment]:
    """Regroup word-level timestamps into subtitle-sized segments.

    Mirrors the logic of regroup_for_subtitles() in postprocess.py but
    operates on plain word dicts — no stable-ts dependency.

    Split rules (in priority order):
    1. Sentence punctuation (. ? ! 。 ？ ！)
    2. Speech gaps > 0.5s
    3. Comma/semicolon if segment has 4+ words
    4. Max chars or max duration exceeded

    Merge rules:
    - Short trailing fragments (≤2 words) merged into previous segment
    - Short leading fragments (≤2 words) merged into next segment
    """
    if not words:
        return []

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
        if prev_text and prev_text[-1] in SENTENCE_END_CHARS:
            groups.append([w])
        else:
            groups[-1].append(w)

    # Phase 2: Split by speech gaps
    split = []
    for group in groups:
        current = [group[0]]
        for w in group[1:]:
            gap = w["start"] - current[-1]["end"]
            if gap > SPEECH_GAP_THRESHOLD:
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
            if len(current) >= 4 and prev_text and prev_text[-1] in CLAUSE_PUNCT:
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

    # Phase 5a: Merge short trailing fragments into PREVIOUS segment.
    # Uses MERGE_CHAR_SLACK to allow slightly longer combined segments
    # rather than leaving dangling 1-2 word fragments.
    merge_limit = max_chars + MERGE_CHAR_SLACK
    merged: list[list[dict]] = [groups[0]]
    for group in groups[1:]:
        prev = merged[-1]
        prev_text = " ".join(w["text"] for w in prev)
        gap = group[0]["start"] - prev[-1]["end"]
        combined_text = " ".join(w["text"] for w in prev + group)
        combined_dur = group[-1]["end"] - prev[0]["start"]
        if (
            len(group) <= MAX_MERGE_TRAIL_WORDS
            and gap < SPEECH_GAP_THRESHOLD
            and prev_text
            and prev_text[-1] not in SENTENCE_END_CHARS
            and len(combined_text) <= merge_limit
            and combined_dur <= max_dur
        ):
            merged[-1].extend(group)
        else:
            merged.append(group)
    groups = merged

    # Phase 5b: Merge short leading fragments into NEXT segment.
    merged = [groups[0]]
    for group in groups[1:]:
        prev = merged[-1]
        gap = group[0]["start"] - prev[-1]["end"]
        combined_text = " ".join(w["text"] for w in prev + group)
        if (
            len(prev) <= MAX_MERGE_LEAD_WORDS
            and gap < MERGE_GAP_THRESHOLD
            and len(prev) + len(group) <= 10
            and len(combined_text) <= merge_limit
        ):
            merged[-1].extend(group)
        else:
            merged.append(group)
    groups = merged

    # Phase 6: Convert groups to SubtitleSegments and fix overlapping timestamps
    segments = []
    for group in groups:
        text = " ".join(w["text"] for w in group)
        start = group[0]["start"]
        end = group[-1]["end"]
        # Clamp start to previous segment's end to prevent overlap
        if segments and start < segments[-1].end:
            start = segments[-1].end
        segments.append(
            SubtitleSegment(
                text=text,
                start=start,
                end=max(end, start),
            )
        )

    return segments


def _get(obj, key: str, default=None):
    """Get a value from a dict or object attribute."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
