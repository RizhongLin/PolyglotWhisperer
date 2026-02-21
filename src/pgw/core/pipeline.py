"""Pipeline orchestrator — download, transcribe, translate, play."""

from __future__ import annotations

import gc
from pathlib import Path

from pgw.core.config import PGWConfig
from pgw.core.events import EventCallback, PipelineEvent
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
    on_event: EventCallback | None = None,
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
        on_event: Optional callback for streaming progress events.

    Returns:
        Path to the workspace directory.
    """

    def emit(stage: str, progress: float, message: str, data: dict | None = None) -> None:
        if on_event:
            on_event(PipelineEvent(stage=stage, progress=progress, message=message, data=data))

    language = config.whisper.language
    segments: list | None = None
    trans_result = None

    # Step 1: Resolve input
    emit("download", 0.0, f"Resolving input: {input_path}")
    if is_url(input_path):
        console.print(f"[bold]Downloading:[/bold] {input_path}")
        source = resolve(input_path, output_dir=config.download_dir)
    else:
        from pgw.core.models import VideoSource

        path = Path(input_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {input_path}")
        source = VideoSource(video_path=path, title=path.stem)
    emit("download", 1.0, "Input resolved")

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
    emit("audio", 0.0, "Extracting audio...")
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
    emit("audio", 1.0, "Audio ready")

    # Step 4: Transcribe (with resume support)
    vtt_path = paths["transcription_vtt"]
    txt_path = paths["transcription_txt"]
    json_path = workspace / "transcription.json"
    needs_llm = (cleanup and config.llm.cleanup_enabled) or translate
    llm_was_used = False
    use_api = config.whisper.backend == "api"

    emit("transcribe", 0.0, "Transcribing...")
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

                emit("transcribe", 0.5, "Cleaning up transcription...")
                console.print("[bold]Cleaning up transcription...[/bold]")

                def _on_cleanup_progress(frac: float) -> None:
                    emit("transcribe", 0.5 + frac * 0.5, f"Cleaning ({frac:.0%})...")

                segments = cleanup_subtitles(
                    segments,
                    language,
                    config.llm,
                    on_progress=_on_cleanup_progress,
                )
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

    emit("transcribe", 1.0, "Transcription complete")

    # Step 5: Optional translation
    if translate:
        trans_vtt = paths["translation_vtt"]
        if not trans_vtt.is_file():
            from pgw.llm.translator import translate_subtitles
            from pgw.subtitles.converter import save_bilingual_vtt, save_subtitles

            emit("translate", 0.0, f"Translating to {translate}...")
            console.print(f"[bold]Translating to {translate}...[/bold]")

            def _on_translate_progress(frac: float) -> None:
                emit("translate", frac, f"Translating ({frac:.0%})...")

            trans_result = translate_subtitles(
                segments,
                language,
                translate,
                config.llm,
                on_progress=_on_translate_progress,
            )
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
            emit("translate", 1.0, "Translation complete")

            # Parallel text export (best-effort, requires export extras)
            try:
                from pgw.subtitles.export import export_parallel_pdf

                pdf_path = workspace / f"parallel.{language}-{translate}.pdf"
                export_parallel_pdf(
                    segments,
                    trans_result.translated,
                    pdf_path,
                    language,
                    translate,
                    title=title,
                )
                console.print(f"[green]Saved:[/green] {pdf_path}")
            except ImportError:
                pass
            except Exception as e:
                console.print(f"[yellow]PDF export skipped:[/yellow] {e}")
        else:
            console.print("[dim]Translation found, skipping.[/dim]")

    # Step 5.5: Vocabulary summary (best-effort)
    emit("vocab", 0.0, "Generating vocabulary summary...")
    try:
        from pgw.vocab.summary import generate_vocab_summary

        trans_segs = trans_result.translated if trans_result is not None else None
        # Ensure segments are loaded (may not be if no LLM processing was done)
        if not segments:
            from pgw.subtitles.converter import load_subtitles

            segments = load_subtitles(vtt_path)

        summary = generate_vocab_summary(segments, language, translated_segments=trans_segs)

        import json

        summary_path = workspace / f"vocabulary.{language}.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(
            f"[green]Vocabulary:[/green] {summary['unique_lemmas']} unique lemmas, "
            f"estimated {summary['estimated_level']}"
        )
    except ImportError:
        pass  # wordfreq or spacy not installed
    except Exception as e:
        console.print(f"[yellow]Vocabulary summary skipped:[/yellow] {e}")
    emit("vocab", 1.0, "Vocabulary summary done")

    # Unload Ollama model from GPU only if LLM was actually used
    if llm_was_used:
        from pgw.llm.client import unload_ollama_model

        unload_ollama_model(config.llm.model)

    # Step 6: Save metadata
    emit("save", 0.0, "Saving metadata...")
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

    emit("save", 1.0, "Done", data={"workspace": str(workspace)})
    console.print(f"\n[bold green]Done![/bold green] Workspace: {workspace}")
    return workspace
