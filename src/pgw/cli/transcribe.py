"""pgw transcribe command — transcribe video/audio to subtitles."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from pgw.core.config import load_config
from pgw.subtitles.converter import save_subtitles
from pgw.utils.audio import extract_audio

console = Console()


def transcribe(
    input_path: Annotated[
        Path,
        typer.Argument(help="Path to video or audio file."),
    ],
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Source language code."),
    ] = "fr",
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="WhisperX model size."),
    ] = "large-v3",
    device: Annotated[
        str,
        typer.Option(help="Compute device: cpu, cuda, mps, or auto."),
    ] = "auto",
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: srt, vtt, ass."),
    ] = "srt",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path. Default: <input>.<lang>.<fmt>"),
    ] = None,
    no_txt: Annotated[
        bool,
        typer.Option("--no-txt", help="Skip generating plain text file."),
    ] = False,
) -> None:
    """Transcribe a video or audio file to subtitles with word-level timestamps."""
    from pgw.transcriber.whisperx import transcribe as whisperx_transcribe

    if not input_path.is_file():
        console.print(f"[red]File not found:[/red] {input_path}")
        raise typer.Exit(1)

    config = load_config(
        **{
            "whisper.model_size": model,
            "whisper.language": language,
            "whisper.device": device,
        }
    )

    # Extract audio if input is a video file
    audio_suffixes = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    if input_path.suffix.lower() in audio_suffixes:
        audio_path = input_path
    else:
        console.print("[bold]Extracting audio...[/bold]")
        audio_path = extract_audio(input_path)

    # Transcribe
    result = whisperx_transcribe(audio_path, config.whisper)

    # Determine output paths
    if output is not None:
        sub_path = output
    else:
        sub_path = input_path.with_suffix(f".{language}.{fmt}")

    # Save subtitle file
    save_subtitles(result.segments, sub_path, fmt=fmt)
    console.print(f"[green]Saved:[/green] {sub_path}")

    # Also save plain text version
    if not no_txt:
        txt_path = sub_path.with_suffix(".txt")
        # Avoid overwriting if fmt is already txt
        if txt_path != sub_path:
            save_subtitles(result.segments, txt_path, fmt="txt")
            console.print(f"[green]Saved:[/green] {txt_path}")
