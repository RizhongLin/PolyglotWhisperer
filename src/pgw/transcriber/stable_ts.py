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
from pgw.transcriber.postprocess import fix_dangling_function_words, regroup_for_subtitles
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
        regroup_for_subtitles(result)
        fix_dangling_function_words(result, config.language)

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
