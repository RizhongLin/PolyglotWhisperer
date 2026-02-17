"""pgw transcribe command — transcribe video/audio to subtitles."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from pgw.core.config import load_config
from pgw.downloader.resolver import is_url, resolve
from pgw.subtitles.converter import save_subtitles
from pgw.utils.audio import extract_audio

console = Console()


def transcribe(
    input_path: Annotated[
        str,
        typer.Argument(help="Path to video/audio file, or a URL."),
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
    cleanup: Annotated[
        bool,
        typer.Option("--cleanup/--no-cleanup", help="Clean up transcription with LLM."),
    ] = False,
    provider: Annotated[
        Optional[str],
        typer.Option(help="LLM provider for cleanup (e.g. ollama_chat/qwen3:8b)."),
    ] = None,
) -> None:
    """Transcribe a video or audio file (or URL) to subtitles with word-level timestamps."""
    from pgw.transcriber.whisperx import transcribe as whisperx_transcribe

    overrides: dict[str, object] = {
        "whisper.model_size": model,
        "whisper.language": language,
        "whisper.device": device,
    }
    if provider is not None:
        overrides["llm.provider"] = provider
    config = load_config(**overrides)

    # Resolve input: URL → download, local path → use directly
    if is_url(input_path):
        console.print(f"[bold]Downloading:[/bold] {input_path}")
        source = resolve(input_path, output_dir=config.download.output_dir)
        video_path = source.video_path
    else:
        video_path = Path(input_path)
        if not video_path.is_file():
            console.print(f"[red]File not found:[/red] {input_path}")
            raise typer.Exit(1)

    # Extract audio if input is a video file
    audio_suffixes = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    if video_path.suffix.lower() in audio_suffixes:
        audio_path = video_path
    else:
        console.print("[bold]Extracting audio...[/bold]")
        audio_path = extract_audio(video_path)

    # Transcribe
    result = whisperx_transcribe(audio_path, config.whisper)
    segments = result.segments

    # Optional LLM cleanup
    if cleanup:
        from pgw.llm.cleanup import cleanup_subtitles

        console.print("[bold]Cleaning up with LLM...[/bold]")
        segments = cleanup_subtitles(segments, language, config.llm)

    # Determine output paths
    if output is not None:
        sub_path = output
    else:
        sub_path = video_path.with_suffix(f".{language}.{fmt}")

    # Save subtitle file
    save_subtitles(segments, sub_path, fmt=fmt)
    console.print(f"[green]Saved:[/green] {sub_path}")

    # Also save plain text version
    if not no_txt:
        txt_path = sub_path.with_suffix(".txt")
        if txt_path != sub_path:
            save_subtitles(segments, txt_path, fmt="txt")
            console.print(f"[green]Saved:[/green] {txt_path}")
