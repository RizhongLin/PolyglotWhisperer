"""WhisperX transcription with word-level timestamp alignment."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from pgw.core.config import WhisperConfig
from pgw.core.models import SubtitleSegment, TranscriptionResult, WordSegment

console = Console()


def _resolve_device(device: str) -> str:
    """Auto-detect the best available compute device."""
    if device != "auto":
        return device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _compute_type_for_device(device: str, requested: str) -> str:
    """Adjust compute type based on device capabilities.

    MPS and CPU don't support float16 well in all cases.
    """
    if device == "cpu" and requested == "float16":
        return "int8"
    return requested


def _parse_segments(raw_segments: list[dict]) -> list[SubtitleSegment]:
    """Convert raw WhisperX segment dicts to SubtitleSegment models."""
    segments = []
    for seg in raw_segments:
        words = []
        for w in seg.get("words", []):
            # WhisperX sometimes omits start/end for low-confidence words
            if "start" in w and "end" in w:
                words.append(
                    WordSegment(
                        word=w["word"],
                        start=w["start"],
                        end=w["end"],
                        score=w.get("score", 0.0),
                    )
                )

        segments.append(
            SubtitleSegment(
                text=seg.get("text", "").strip(),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                words=words,
                speaker=seg.get("speaker"),
            )
        )
    return segments


def transcribe(audio_path: Path, config: WhisperConfig) -> TranscriptionResult:
    """Transcribe audio using WhisperX with word-level alignment.

    Args:
        audio_path: Path to audio file (WAV, 16kHz mono recommended).
        config: Whisper configuration.

    Returns:
        TranscriptionResult with segments and word-level timestamps.
    """
    try:
        import whisperx
    except ImportError:
        raise ImportError("WhisperX is not installed. Install with: uv sync --extra transcribe")

    device = _resolve_device(config.device)
    compute_type = _compute_type_for_device(device, config.compute_type)

    console.print(f"[bold]Loading WhisperX model:[/bold] {config.model_size} on {device}")
    model = whisperx.load_model(
        config.model_size,
        device=device,
        compute_type=compute_type,
        language=config.language,
    )

    console.print("[bold]Loading audio...[/bold]")
    audio = whisperx.load_audio(str(audio_path))

    console.print("[bold]Transcribing...[/bold]")
    result = model.transcribe(audio, batch_size=config.batch_size)

    # Word-level alignment via wav2vec2
    if config.word_timestamps:
        console.print("[bold]Aligning words...[/bold]")
        align_model, align_metadata = whisperx.load_align_model(
            language_code=result.get("language", config.language),
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            device=device,
            return_char_alignments=False,
        )

    segments = _parse_segments(result.get("segments", []))

    console.print(f"[green]Transcription complete:[/green] {len(segments)} segments")

    return TranscriptionResult(
        segments=segments,
        language=config.language,
        audio_path=audio_path,
        model_used=config.model_size,
    )
