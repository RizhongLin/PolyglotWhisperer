"""Pipeline orchestrator — download, transcribe, translate, play."""

from __future__ import annotations

import gc
from pathlib import Path

from pgw.core.config import PGWConfig
from pgw.downloader.resolver import is_url, resolve
from pgw.utils.audio import extract_audio_cached
from pgw.utils.cache import link_or_copy
from pgw.utils.console import console
from pgw.utils.paths import create_workspace, save_metadata, workspace_paths


def run_pipeline(
    input_path: str,
    config: PGWConfig,
    translate: str | None = None,
    cleanup: bool = True,
    play: bool = True,
    start: str | None = None,
    duration: str | None = None,
) -> Path:
    """Run the full processing pipeline.

    Args:
        input_path: URL or local file path.
        config: Full application config.
        translate: Target language code, or None to skip translation.
        cleanup: Whether to clean up transcription with LLM.
        play: Whether to play the video with subtitles after processing.
        start: Start time for audio clipping (ffmpeg format).
        duration: Duration to extract (ffmpeg format).

    Returns:
        Path to the workspace directory.
    """
    language = config.whisper.language

    # Step 1: Resolve input
    if is_url(input_path):
        console.print(f"[bold]Downloading:[/bold] {input_path}")
        source = resolve(input_path, output_dir=config.download_dir)
    else:
        from pgw.core.models import VideoSource

        path = Path(input_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {input_path}")
        source = VideoSource(video_path=path, title=path.stem)

    # Step 2: Create workspace
    title = source.title or source.video_path.stem
    workspace = create_workspace(title, base_dir=config.workspace_dir)
    paths = workspace_paths(workspace, language, target_lang=translate)
    console.print(f"[bold]Workspace:[/bold] {workspace}")

    # Link video into workspace (symlink to save disk, copy as fallback)
    video_dest = paths["video"]
    if not video_dest.is_file() and not video_dest.is_symlink():
        link_or_copy(source.video_path, video_dest)

    # Step 3: Extract audio (with cross-workspace cache)
    audio_path = paths["audio"]
    if not audio_path.is_file():
        clip_msg = ""
        if start or duration:
            parts = []
            if start:
                parts.append(f"from {start}")
            if duration:
                parts.append(f"duration {duration}")
            clip_msg = f" ({', '.join(parts)})"
        console.print(f"[bold]Extracting audio{clip_msg}...[/bold]")
        _, cache_hit = extract_audio_cached(
            source.video_path,
            output_path=audio_path,
            workspace_dir=config.workspace_dir,
            start=start,
            duration=duration,
        )
        if cache_hit:
            console.print("[dim]Audio found in cache.[/dim]")
    else:
        console.print("[dim]Audio already extracted, skipping.[/dim]")

    # Step 4: Transcribe (with resume support)
    vtt_path = paths["transcription_vtt"]
    txt_path = paths["transcription_txt"]
    json_path = workspace / "transcription.json"
    needs_llm = (cleanup and config.llm.cleanup_enabled) or translate
    llm_was_used = False
    use_api = config.whisper.backend == "api"

    if not vtt_path.is_file():
        if use_api:
            # API transcription — returns segments directly, no WhisperResult
            from pgw.transcriber.api import transcribe as api_transcribe

            segments = api_transcribe(audio_path, config.whisper)

        elif json_path.is_file() and needs_llm:
            # Resume: transcription JSON cached from interrupted run
            console.print("[dim]Transcription JSON found, loading segments...[/dim]")
            import stable_whisper

            from pgw.subtitles.converter import result_to_segments

            result = stable_whisper.WhisperResult(str(json_path))
            segments = result_to_segments(result)
            del result
        else:
            from pgw.transcriber.stable_ts import transcribe

            result = transcribe(audio_path, config.whisper)

            # Save full result as JSON for reprocessing
            result.save_as_json(str(json_path))
            console.print(f"[green]Saved:[/green] {json_path}")

            if needs_llm:
                from pgw.subtitles.converter import result_to_segments

                segments = result_to_segments(result)
            else:
                # Use stable-ts built-in export — auto-detects VTT from extension
                result.to_srt_vtt(str(vtt_path))
                console.print(f"[green]Saved:[/green] {vtt_path}")
                result.to_txt(str(txt_path))
                console.print(f"[green]Saved:[/green] {txt_path}")

            # Free transcription result to reclaim memory before LLM steps
            del result
            gc.collect()

        # Post-processing and save for segment-based paths (API or LLM)
        if use_api or needs_llm:
            from pgw.subtitles.converter import save_subtitles
            from pgw.transcriber.postprocess import fix_dangling_clitics

            segments = fix_dangling_clitics(segments, language)

            if cleanup and config.llm.cleanup_enabled:
                from pgw.llm.cleanup import cleanup_subtitles

                console.print("[bold]Cleaning up transcription...[/bold]")
                segments = cleanup_subtitles(segments, language, config.llm)
                llm_was_used = True

            save_subtitles(segments, vtt_path, fmt="vtt")
            console.print(f"[green]Saved:[/green] {vtt_path}")
            save_subtitles(segments, txt_path, fmt="txt")
            console.print(f"[green]Saved:[/green] {txt_path}")
    else:
        console.print("[dim]Transcription found, skipping.[/dim]")
        if needs_llm:
            from pgw.subtitles.converter import load_subtitles
            from pgw.transcriber.postprocess import fix_dangling_clitics

            segments = load_subtitles(vtt_path)
            segments = fix_dangling_clitics(segments, language)

    # Step 5: Optional translation
    if translate:
        trans_vtt = paths["translation_vtt"]
        if not trans_vtt.is_file():
            from pgw.llm.translator import translate_subtitles
            from pgw.subtitles.converter import save_bilingual_vtt, save_subtitles

            console.print(f"[bold]Translating to {translate}...[/bold]")
            trans_result = translate_subtitles(segments, language, translate, config.llm)
            llm_was_used = True

            save_subtitles(trans_result.translated, trans_vtt, fmt="vtt")
            console.print(f"[green]Saved:[/green] {trans_vtt}")

            trans_txt = paths["translation_txt"]
            save_subtitles(trans_result.translated, trans_txt, fmt="txt")
            console.print(f"[green]Saved:[/green] {trans_txt}")

            # Bilingual VTT: original at bottom, translation at top
            bi_vtt = paths["bilingual_vtt"]
            save_bilingual_vtt(segments, trans_result.translated, bi_vtt)
            console.print(f"[green]Saved:[/green] {bi_vtt}")
        else:
            console.print("[dim]Translation found, skipping.[/dim]")

    # Unload Ollama model from GPU only if LLM was actually used
    if llm_was_used:
        from pgw.llm.client import unload_ollama_model

        unload_ollama_model(config.llm.model)

    # Step 6: Save metadata
    save_metadata(
        workspace,
        source_url=source.source_url,
        title=title,
        language=language,
        target_language=translate,
        cleanup=cleanup,
        whisper_model=config.whisper.model,
        whisper_device=config.whisper.device,
        llm_model=config.llm.model,
        start=start,
        duration=duration,
        source_duration=source.duration,
    )

    # Step 7: Optional playback
    if play:
        from pgw.player.mpv_player import check_mpv
        from pgw.player.mpv_player import play as mpv_play

        if check_mpv():
            mpv_play(
                video_dest,
                primary_subs=vtt_path,
                bilingual_subs=paths.get("bilingual_vtt"),
                config=config.player,
            )
        else:
            console.print("[yellow]mpv not found, skipping playback.[/yellow]")

    console.print(f"\n[bold green]Done![/bold green] Workspace: {workspace}")
    return workspace
