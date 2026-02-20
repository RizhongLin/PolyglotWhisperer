"""pgw transcribe command — transcribe video/audio to subtitles."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from pgw.core.config import load_config
from pgw.downloader.resolver import is_url, resolve
from pgw.utils.audio import extract_audio
from pgw.utils.console import console


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
        Optional[str],
        typer.Option("--model", "-m", help="Model for the active backend."),
    ] = None,
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
    llm_model: Annotated[
        Optional[str],
        typer.Option("--llm-model", help="LLM model for cleanup (e.g. ollama_chat/qwen3:8b)."),
    ] = None,
    backend: Annotated[
        Optional[str],
        typer.Option(help="Transcription backend: local or api."),
    ] = None,
) -> None:
    """Transcribe a video or audio file (or URL) to subtitles with word-level timestamps."""
    from pgw.core.languages import validate_language

    try:
        validate_language(language)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    overrides: dict[str, object] = {
        "whisper.language": language,
        "whisper.device": device,
    }
    if model is not None:
        model_key = "whisper.api_model" if backend == "api" else "whisper.local_model"
        overrides[model_key] = model
    if llm_model is not None:
        overrides["llm.model"] = llm_model
    if backend is not None:
        overrides["whisper.backend"] = backend
    config = load_config(**overrides)

    # Resolve input: URL → download, local path → use directly
    if is_url(input_path):
        console.print(f"[bold]Downloading:[/bold] {input_path}")
        source = resolve(input_path, output_dir=config.download_dir)
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

    # Determine output path
    if output is not None:
        sub_path = output
    else:
        sub_path = video_path.with_suffix(f".{language}.{fmt}")

    use_api = config.whisper.backend == "api"

    if use_api:
        # API transcription — returns segments directly
        from pgw.subtitles.converter import save_subtitles
        from pgw.transcriber.api import transcribe as api_transcribe
        from pgw.transcriber.postprocess import fix_dangling_clitics

        segments = api_transcribe(audio_path, config.whisper)
        segments = fix_dangling_clitics(segments, language)

        if cleanup:
            from pgw.llm.cleanup import cleanup_subtitles

            console.print("[bold]Cleaning up with LLM...[/bold]")
            segments = cleanup_subtitles(segments, language, config.llm)

        save_subtitles(segments, sub_path, fmt=fmt)
    else:
        # Local transcription — returns raw stable-ts WhisperResult
        from pgw.transcriber.stable_ts import transcribe as do_transcribe

        result = do_transcribe(audio_path, config.whisper)

        if cleanup:
            from pgw.llm.cleanup import cleanup_subtitles
            from pgw.subtitles.converter import result_to_segments, save_subtitles
            from pgw.transcriber.postprocess import fix_dangling_clitics

            segments = result_to_segments(result)
            segments = fix_dangling_clitics(segments, language)
            console.print("[bold]Cleaning up with LLM...[/bold]")
            segments = cleanup_subtitles(segments, language, config.llm)
            save_subtitles(segments, sub_path, fmt=fmt)
        else:
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
            if use_api or cleanup:
                from pgw.subtitles.converter import save_subtitles

                save_subtitles(segments, txt_path, fmt="txt")
            else:
                result.to_txt(str(txt_path))
            console.print(f"[green]Saved:[/green] {txt_path}")
