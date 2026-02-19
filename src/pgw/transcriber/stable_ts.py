"""Transcription using stable-ts with multi-backend support.

Backends (selected automatically based on platform):
- MLX: Apple Silicon (fastest on Mac)
- Vanilla Whisper: CUDA or CPU fallback
"""

from __future__ import annotations

import gc
import platform
from pathlib import Path

from pgw.core.config import WhisperConfig
from pgw.utils.console import console


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _select_backend(device: str) -> str:
    """Select the best transcription backend for the current platform."""
    if device in ("mps", "auto") and _is_apple_silicon():
        try:
            import mlx.core  # noqa: F401

            return "mlx"
        except ImportError:
            console.print("[yellow]MLX not installed, falling back to vanilla Whisper.[/yellow]")

    if device in ("cuda", "auto"):
        try:
            import torch

            if torch.cuda.is_available():
                return "vanilla"
        except ImportError:
            pass

    return "vanilla"


def _resolve_device(device: str, backend: str) -> str | None:
    """Resolve the compute device string for the chosen backend.

    Returns None for MLX (no device parameter needed).
    """
    if backend == "mlx":
        return None

    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    if device == "mps":
        console.print("[yellow]Note:[/yellow] MPS not optimal for Whisper, using CPU.")
        return "cpu"

    return device


def _load_model(config: WhisperConfig):
    """Load a Whisper model using the best available backend."""
    import stable_whisper

    backend = _select_backend(config.device)
    device = _resolve_device(config.device, backend)

    if backend == "mlx":
        console.print(f"[bold]Loading model:[/bold] {config.model_size} (MLX, Apple Silicon)")
        return stable_whisper.load_mlx_whisper(config.model_size)

    console.print(f"[bold]Loading model:[/bold] {config.model_size} on {device}")
    return stable_whisper.load_model(config.model_size, device=device)


# POS tags that should not dangle at the end of a subtitle segment.
# DET = determiners (le/la/les/the/der/die/das/el/la/los...)
# ADP = adpositions/prepositions (de/en/à/of/in/to/von/mit...)
_DANGLING_POS = {"DET", "ADP"}

# Mapping from pgw language codes to spaCy model names.
# Languages without a model here gracefully skip the function-word fix.
_SPACY_MODELS: dict[str, str] = {
    "ca": "ca_core_news_sm",
    "da": "da_core_news_sm",
    "de": "de_core_news_sm",
    "el": "el_core_news_sm",
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "fi": "fi_core_news_sm",
    "fr": "fr_core_news_sm",
    "hr": "hr_core_news_sm",
    "it": "it_core_news_sm",
    "ja": "ja_core_news_sm",
    "ko": "ko_core_news_sm",
    "lt": "lt_core_news_sm",
    "mk": "mk_core_news_sm",
    "nb": "nb_core_news_sm",
    "nl": "nl_core_news_sm",
    "pl": "pl_core_news_sm",
    "pt": "pt_core_news_sm",
    "ro": "ro_core_news_sm",
    "ru": "ru_core_news_sm",
    "sl": "sl_core_news_sm",
    "sv": "sv_core_news_sm",
    "uk": "uk_core_news_sm",
    "zh": "zh_core_web_sm",
}

_spacy_cache: dict[str, object] = {}


def _load_spacy_model(language: str):
    """Load a spaCy model for POS tagging, auto-downloading if needed.

    Returns the loaded model, or None if spaCy is not installed or the
    language has no model available.  Results are cached per language.
    """
    if language in _spacy_cache:
        return _spacy_cache[language]

    try:
        import spacy
    except ImportError:
        _spacy_cache[language] = None
        return None

    model_name = _SPACY_MODELS.get(language)
    if model_name is None:
        _spacy_cache[language] = None
        return None

    try:
        nlp = spacy.load(model_name, disable=["parser", "lemmatizer", "ner"])
    except OSError:
        # Model not installed — auto-download
        console.print(f"[bold]Downloading spaCy model:[/bold] {model_name}")
        try:
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name, disable=["parser", "lemmatizer", "ner"])
        except (SystemExit, Exception):
            console.print(f"[yellow]Could not load spaCy model {model_name}, skipping.[/yellow]")
            _spacy_cache[language] = None
            return None

    _spacy_cache[language] = nlp
    return nlp


def _regroup_for_subtitles(result, max_chars: int = 50) -> None:
    """Rebuild segments from word-level timestamps for subtitle display.

    Merges all segments, then re-splits using punctuation, speech gaps,
    and length constraints.  This produces subtitle-friendly segments
    with correct word-level timestamps.
    """
    result.merge_all_segments()
    result.split_by_punctuation([(".", " "), "?", "!", "。", "？", "！"])
    result.split_by_gap(0.5)
    result.split_by_punctuation([(",", " "), ";", "，", "；"])
    result.split_by_length(max_chars=max_chars)
    result.merge_by_gap(0.15, max_words=3)


def _fix_dangling_function_words(result, language: str) -> None:
    """Move dangling function words from segment ends to the next segment.

    Uses spaCy POS tagging to detect determiners (DET) and adpositions
    (ADP) at segment boundaries.  Works for any language with a spaCy
    model.  Falls back silently if spaCy is not installed.
    """
    nlp = _load_spacy_model(language)
    if nlp is None:
        return

    segments = result.segments
    for i in range(len(segments) - 1):
        words = segments[i].words
        if len(words) <= 1:
            continue  # Don't empty a segment

        # Run POS tagger on full segment text for context-aware tagging
        segment_text = segments[i].text.strip()
        if not segment_text:
            continue

        doc = nlp(segment_text)
        if not doc:
            continue

        # Check if the last token is a function word (DET or ADP)
        last_token = doc[-1]
        if last_token.pos_ not in _DANGLING_POS:
            continue

        # Move word from end of current segment to start of next
        word = words.pop()
        segments[i].reassign_ids()

        segments[i + 1].words.insert(0, word)
        word.segment = segments[i + 1]
        segments[i + 1].reassign_ids()

    # Drop segments that became empty (all words moved out)
    result.segments = [s for s in result.segments if s.words]


def transcribe(audio_path: Path, config: WhisperConfig):
    """Transcribe audio using stable-ts.

    Returns the raw stable-ts WhisperResult, which supports built-in
    export methods: to_srt_vtt(), to_txt(), save_as_json(), to_dict().

    Args:
        audio_path: Path to audio file (WAV, 16kHz mono recommended).
        config: Whisper configuration.

    Returns:
        stable_whisper.result.WhisperResult
    """
    try:
        import stable_whisper  # noqa: F401
    except ImportError:
        raise ImportError("stable-ts is not installed. Install with: uv sync --extra transcribe")

    model = _load_model(config)

    console.print("[bold]Transcribing...[/bold]")
    result = model.transcribe(
        str(audio_path),
        language=config.language,
        word_timestamps=config.word_timestamps,
        regroup=False,  # We apply our own subtitle-optimized regrouping
    )

    # Unload model and free GPU memory before regrouping
    del model
    gc.collect()
    _clear_gpu_cache()

    # Regroup for subtitle display (uses word-level timestamps)
    if config.word_timestamps:
        _regroup_for_subtitles(result)
        _fix_dangling_function_words(result, config.language)

    seg_count = len(result.segments) if result.segments else 0
    console.print(f"[green]Transcription complete:[/green] {seg_count} segments")

    return result


def _clear_gpu_cache() -> None:
    """Release GPU/accelerator memory after model use."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    try:
        import mlx.core as mx

        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        else:
            mx.metal.reset_peak_memory()
    except (ImportError, AttributeError):
        pass
