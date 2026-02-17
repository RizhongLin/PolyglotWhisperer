"""pgw transcribe command — transcribe video/audio to subtitles."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from pgw.core.config import load_config
from pgw.downloader.resolver import is_url, resolve
from pgw.utils.audio import extract_audio

console = Console()


def transcribe(
    input_path: Annotated[
        str,
        typer.Argument(help="Path to video/audio file, or a URL."),
    ],
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Source language code (see 'pgw languages')."),
    ] = "fr",
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Whisper model size."),
    ] = "large-v3-turbo",
    device: Annotated[
        str,
        typer.Option(help="Compute device: cpu, cuda, mps, or auto."),
    ] = "auto",
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: vtt, srt, ass."),
    ] = "vtt",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path. Default: <input>.<lang>.<fmt>"),
    ] = None,
    no_txt: Annotated[
        bool,
        typer.Option("--no-txt", help="Skip generating plain text file."),
    ] = False,
    start: Annotated[
        Optional[str],
        typer.Option("--start", help="Start time (e.g. '00:01:00' or '60')."),
    ] = None,
    duration: Annotated[
        Optional[str],
        typer.Option("--duration", help="Duration to process (e.g. '00:05:00' or '300')."),
    ] = None,
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
    from pgw.core.languages import validate_language
    from pgw.transcriber.stable_ts import transcribe as do_transcribe

    try:
        validate_language(language)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

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
        audio_path = extract_audio(video_path, start=start, duration=duration)

    # Transcribe — returns raw stable-ts WhisperResult
    result = do_transcribe(audio_path, config.whisper)

    # Determine output path
    if output is not None:
        sub_path = output
    else:
        sub_path = video_path.with_suffix(f".{language}.{fmt}")

    if cleanup:
        # LLM cleanup path: convert to segments, clean, save via pysubs2
        from pgw.llm.cleanup import cleanup_subtitles
        from pgw.subtitles.converter import result_to_segments, save_subtitles

        segments = result_to_segments(result)
        console.print("[bold]Cleaning up with LLM...[/bold]")
        segments = cleanup_subtitles(segments, language, config.llm)
        save_subtitles(segments, sub_path, fmt=fmt)
    else:
        # Direct export via stable-ts built-in methods
        if fmt in ("srt", "vtt"):
            result.to_srt_vtt(str(sub_path), vtt=(fmt == "vtt"))
        elif fmt == "ass":
            result.to_ass(str(sub_path))
        else:
            result.to_srt_vtt(str(sub_path), vtt=True)

    console.print(f"[green]Saved:[/green] {sub_path}")

    # Also save plain text version
    if not no_txt:
        txt_path = sub_path.with_suffix(".txt")
        if txt_path != sub_path:
            if cleanup:
                save_subtitles(segments, txt_path, fmt="txt")
            else:
                result.to_txt(str(txt_path))
            console.print(f"[green]Saved:[/green] {txt_path}")
