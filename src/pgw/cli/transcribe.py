"""pgw transcribe command — transcribe video/audio to subtitles."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from pgw.cli.utils import build_config_overrides, expand_inputs, print_batch_summary
from pgw.core.config import PGWConfig, load_config
from pgw.downloader.resolver import is_url, resolve
from pgw.utils.audio import extract_audio
from pgw.utils.console import console, error, saved, stage, warning


def transcribe(
    inputs: Annotated[
        list[str],
        typer.Argument(help="URLs, file paths, or glob patterns. Accepts multiple inputs."),
    ],
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Source language code (see 'pgw languages')."),
    ] = "fr",
    whisper_model: Annotated[
        Optional[str],
        typer.Option("--whisper-model", "-w", help="Whisper model (e.g. large-v3-turbo)."),
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
    refine: Annotated[
        bool,
        typer.Option("--refine/--no-refine", help="Refine transcription with LLM."),
    ] = False,
    llm_model: Annotated[
        Optional[str],
        typer.Option("--llm-model", help="LLM model for refinement (e.g. ollama_chat/qwen3:8b)."),
    ] = None,
    llm_backend: Annotated[
        Optional[str],
        typer.Option(help="LLM backend: local or api."),
    ] = None,
    backend: Annotated[
        Optional[str],
        typer.Option(help="Transcription backend: local or api."),
    ] = None,
    subs: Annotated[
        bool,
        typer.Option("--subs/--no-subs", help="Download subtitles from video pages."),
    ] = False,
) -> None:
    """Transcribe video/audio files (or URLs) to subtitles with word-level timestamps.

    Accepts multiple inputs — files, URLs, glob patterns (*.mp4), or .txt
    files containing one URL/path per line.
    """
    from pgw.core.languages import validate_language

    try:
        validate_language(language)
    except ValueError as e:
        error(str(e))
        raise typer.Exit(1)

    overrides = build_config_overrides(
        language=language,
        device=device,
        whisper_model=whisper_model,
        llm_model=llm_model,
        llm_backend=llm_backend,
        backend=backend,
        subs=subs,
    )
    config = load_config(**overrides)

    expanded = expand_inputs(inputs)
    if not expanded:
        error("No inputs resolved. Check your paths or patterns.")
        raise typer.Exit(1)

    # Single input — original behavior
    if len(expanded) == 1:
        _transcribe_single(
            expanded[0],
            config,
            language,
            fmt,
            output,
            not no_txt,
            refine,
            start,
            duration,
        )
        return

    # Batch mode — output flag ignored, process each
    if output is not None:
        warning("--output ignored in batch mode (auto-naming per file).")

    results: list[tuple[str, str, str]] = []
    console.print(f"[bold]Batch transcribing {len(expanded)} inputs...[/bold]\n")

    for i, input_path in enumerate(expanded, 1):
        console.rule(f"[bold][{i}/{len(expanded)}] {input_path}[/bold]")
        try:
            _transcribe_single(
                input_path,
                config,
                language,
                fmt,
                None,
                not no_txt,
                refine,
                start,
                duration,
            )
            results.append((input_path, "success", ""))
        except Exception as e:
            error(f"Failed: {e}")
            results.append((input_path, "failed", str(e)))

    # Summary table
    console.print()
    print_batch_summary(results, total=len(expanded))


def _transcribe_single(
    input_path: str,
    config: PGWConfig,
    language: str,
    fmt: str,
    output: Path | None,
    save_txt: bool,
    refine: bool,
    start: str | None,
    duration: str | None,
) -> None:
    """Transcribe a single input file or URL."""
    # Resolve input: URL → download, local path → use directly
    source = None
    if is_url(input_path):
        sub_language = language if config.download.subtitles else None
        source = resolve(
            input_path,
            output_dir=config.download_dir,
            fmt=config.download.format,
            language=sub_language,
        )
        video_path = source.video_path
    else:
        video_path = Path(input_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"File not found: {input_path}")

    # Extract audio if input is a video file
    audio_suffixes = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    if video_path.suffix.lower() in audio_suffixes:
        audio_path = video_path
    else:
        stage("Extracting audio")
        audio_path = extract_audio(video_path, start=start, duration=duration)

    # Determine output path
    if output is not None:
        sub_path = output
    else:
        sub_path = video_path.with_suffix(f".{language}.{fmt}")

    # Use downloaded subtitles if available (skip Whisper)
    if source and source.subtitle_path:
        from pgw.subtitles.converter import load_subtitles, save_subtitles

        kind = "auto-generated" if source.subtitle_is_auto else "human-made"
        stage("Using downloaded subtitles", f"{kind}, skipping Whisper")
        segments = load_subtitles(source.subtitle_path)
        if source.subtitle_is_auto:
            from pgw.transcriber.postprocess import postprocess_segments

            segments = postprocess_segments(segments, language)

        if refine:
            from pgw.llm.refine import refine_subtitles

            stage("Refining", config.llm.model)
            segments = refine_subtitles(segments, language, config.llm)

        save_subtitles(segments, sub_path, fmt=fmt)
        saved(sub_path)

        if save_txt:
            txt_path = sub_path.with_suffix(".txt")
            if txt_path != sub_path:
                save_subtitles(segments, txt_path, fmt="txt")
                saved(txt_path)
        return

    use_api = config.whisper.backend == "api"

    if use_api:
        # API transcription — returns segments directly
        from pgw.subtitles.converter import save_subtitles
        from pgw.transcriber.api import transcribe as api_transcribe
        from pgw.transcriber.postprocess import postprocess_segments

        segments = api_transcribe(audio_path, config.whisper, config.workspace_dir)
        segments = postprocess_segments(segments, language)

        if refine:
            from pgw.llm.refine import refine_subtitles

            stage("Refining", config.llm.model)
            segments = refine_subtitles(segments, language, config.llm)

        save_subtitles(segments, sub_path, fmt=fmt)
    else:
        # Local transcription — returns raw stable-ts WhisperResult
        from pgw.transcriber.stable_ts import transcribe as do_transcribe

        result = do_transcribe(audio_path, config.whisper)

        if refine:
            from pgw.llm.refine import refine_subtitles
            from pgw.subtitles.converter import result_to_segments, save_subtitles
            from pgw.transcriber.postprocess import postprocess_segments

            segments = result_to_segments(result)
            segments = postprocess_segments(segments, language)
            stage("Refining", config.llm.model)
            segments = refine_subtitles(segments, language, config.llm)
            save_subtitles(segments, sub_path, fmt=fmt)
        else:
            if fmt in ("srt", "vtt"):
                result.to_srt_vtt(str(sub_path), vtt=(fmt == "vtt"))
            elif fmt == "ass":
                result.to_ass(str(sub_path))
            else:
                result.to_srt_vtt(str(sub_path), vtt=True)

    saved(sub_path)

    # Also save plain text version
    if save_txt:
        txt_path = sub_path.with_suffix(".txt")
        if txt_path != sub_path:
            if use_api or refine:
                from pgw.subtitles.converter import save_subtitles

                save_subtitles(segments, txt_path, fmt="txt")
            else:
                result.to_txt(str(txt_path))
            saved(txt_path)
