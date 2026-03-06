"""Pipeline orchestrator — download, transcribe, translate, play."""

from __future__ import annotations

import gc
import json
from pathlib import Path

from pgw.core.config import PGWConfig
from pgw.core.events import EventCallback, PipelineEvent
from pgw.downloader.resolver import is_url, resolve
from pgw.utils.audio import extract_audio_cached
from pgw.utils.cache import (
    atomic_write_text,
    cache_key,
    find_cached_file,
    get_cache_dir,
    link_or_copy,
)
from pgw.utils.console import cache_hit, debug, stage, warning, workspace_done
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
    chunk_size: int | None = None,
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

    def emit(stg: str, progress: float, message: str, data: dict | None = None) -> None:
        if on_event:
            on_event(PipelineEvent(stage=stg, progress=progress, message=message, data=data))

    language = config.whisper.language
    segments: list | None = None
    trans_result = None
    saved: list[Path] = []

    # Step 1: Resolve input (always goes through resolve() for content_hash)
    emit("download", 0.0, f"Resolving input: {input_path}")
    if is_url(input_path):
        stage("Downloading", input_path)
    sub_language = language if config.download.subtitles else None
    source = resolve(
        input_path,
        output_dir=config.download_dir,
        fmt=config.download.format,
        language=sub_language,
    )
    emit("download", 1.0, "Input resolved")

    # Step 2: Create workspace
    title = source.title or source.video_path.stem
    video_ext = source.video_path.suffix or ".mp4"
    workspace = create_workspace(title, base_dir=config.workspace_dir)
    paths = workspace_paths(workspace, language, target_lang=translate, video_ext=video_ext)

    # Link video into workspace (symlink to save disk, copy as fallback)
    video_dest = paths["video"]
    if not video_dest.is_file():
        if video_dest.is_symlink():
            video_dest.unlink()  # Remove broken symlink
        link_or_copy(source.video_path, video_dest)

    # Step 3: Extract audio (with cross-workspace cache)
    emit("audio", 0.0, "Extracting audio...")
    audio_path = paths["audio"]
    if not audio_path.is_file():
        clip_detail = ""
        if start or duration:
            parts = []
            if start:
                parts.append(f"from {start}")
            if duration:
                parts.append(f"duration {duration}")
            clip_detail = ", ".join(parts)
        stage("Extracting audio", clip_detail)
        _, was_cached = extract_audio_cached(
            source.video_path,
            output_path=audio_path,
            workspace_dir=config.workspace_dir,
            start=start,
            duration=duration,
            content_hash=source.content_hash,
        )
        if was_cached:
            cache_hit()
    else:
        cache_hit("Audio cached")
    emit("audio", 1.0, "Audio ready")

    # Step 3.5: Use downloaded subtitles if available (skip Whisper)
    vtt_path = paths["transcription_vtt"]
    txt_path = paths["transcription_txt"]
    if source.subtitle_path and not vtt_path.is_file():
        from pgw.subtitles.converter import load_subtitles, save_subtitles

        emit("transcribe", 0.0, "Loading downloaded subtitles...")
        kind = "auto-generated" if source.subtitle_is_auto else "human-made"
        stage("Transcribing", f"using downloaded subtitles ({kind})")
        segments = load_subtitles(source.subtitle_path)
        # Only postprocess auto-generated subs; human-made are already well-segmented
        if source.subtitle_is_auto:
            from pgw.transcriber.postprocess import postprocess_segments

            segments = postprocess_segments(segments, language)
        save_subtitles(segments, vtt_path, fmt="vtt")
        saved.append(vtt_path)
        save_subtitles(segments, txt_path, fmt="txt")
        saved.append(txt_path)
        emit("transcribe", 1.0, "Downloaded subtitles ready")

    # Step 4: Transcribe (with shared cache)
    needs_llm = (cleanup and config.llm.cleanup_enabled) or translate
    llm_was_used = False
    use_api = config.whisper.backend == "api"

    # Shared transcription cache: .cache/transcriptions/<hash>.json
    # Derive audio identity from video content hash + extraction params
    audio_identity = None
    if source.content_hash:
        audio_identity = cache_key(
            content_hash=source.content_hash,
            sample_rate=16000,
            start=start,
            duration=duration,
        )

    trans_cache_dir = get_cache_dir(config.workspace_dir, "transcriptions")
    trans_params = dict(model=config.whisper.model, backend=config.whisper.backend)
    trans_cache_path = find_cached_file(
        trans_cache_dir,
        ".json",
        content_hash=audio_identity,
        file_path=audio_path,
        **trans_params,
    )

    # Determine write path for new cache entries (prefer content-based key)
    if audio_identity:
        trans_write_key = cache_key(content_hash=audio_identity, **trans_params)
    else:
        trans_write_key = cache_key(audio_path, **trans_params)
    trans_write_path = trans_cache_dir / f"{trans_write_key}.json"

    # Validate cached transcription JSON (could be corrupted from interrupted write)
    if trans_cache_path is not None:
        try:
            json.loads(trans_cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            debug("Cached transcription corrupted, regenerating...")
            trans_cache_path = None

    emit("transcribe", 0.0, "Transcribing...")
    if not vtt_path.is_file():
        if use_api and trans_cache_path is not None:
            # Cache hit: load API segments from shared cache
            stage("Transcribing", config.whisper.model)
            cache_hit()
            from pgw.core.models import SubtitleSegment

            raw = json.loads(trans_cache_path.read_text(encoding="utf-8"))
            segments = [SubtitleSegment(**s) for s in raw]

        elif use_api:
            # API transcription — returns segments directly, no WhisperResult
            stage("Transcribing", config.whisper.model)
            from pgw.transcriber.api import transcribe as api_transcribe

            segments = api_transcribe(
                audio_path,
                config.whisper,
                config.workspace_dir,
                content_hash=audio_identity,
            )

            # Save to shared cache (atomic write to prevent corruption)
            raw = [{"text": s.text, "start": s.start, "end": s.end} for s in segments]
            atomic_write_text(trans_write_path, json.dumps(raw, ensure_ascii=False, indent=2))
            debug("Transcription cached.")

        elif trans_cache_path is not None and needs_llm:
            # Cache hit: load local transcription from shared cache
            stage("Transcribing", config.whisper.model)
            cache_hit()
            import stable_whisper

            from pgw.subtitles.converter import result_to_segments

            result = stable_whisper.WhisperResult(str(trans_cache_path))
            segments = result_to_segments(result)
            del result
        else:
            from pgw.transcriber.stable_ts import transcribe

            result = transcribe(audio_path, config.whisper)

            # Save full result to shared cache (atomic to prevent corruption)
            tmp_path = trans_write_path.with_suffix(".tmp")
            result.save_as_json(str(tmp_path))
            tmp_path.replace(trans_write_path)
            debug("Transcription cached.")

            if needs_llm:
                from pgw.subtitles.converter import result_to_segments

                segments = result_to_segments(result)
            else:
                # Use stable-ts built-in export — auto-detects VTT from extension
                result.to_srt_vtt(str(vtt_path))
                saved.append(vtt_path)
                result.to_txt(str(txt_path))
                saved.append(txt_path)

            # Free transcription result to reclaim memory before LLM steps
            del result
            gc.collect()

        # Post-processing and save for segment-based paths (API or LLM)
        if use_api or needs_llm:
            from pgw.subtitles.converter import save_subtitles
            from pgw.transcriber.postprocess import postprocess_segments

            segments = postprocess_segments(segments, language)

            if cleanup and config.llm.cleanup_enabled:
                from pgw.llm.cleanup import cleanup_subtitles

                emit("transcribe", 0.5, "Cleaning up transcription...")
                stage("Cleaning up", config.llm.model)

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
            saved.append(vtt_path)
            save_subtitles(segments, txt_path, fmt="txt")
            saved.append(txt_path)
    else:
        cache_hit("Transcription cached")
        if needs_llm:
            from pgw.subtitles.converter import load_subtitles
            from pgw.transcriber.postprocess import postprocess_segments

            segments = load_subtitles(vtt_path)
            segments = postprocess_segments(segments, language)

    emit("transcribe", 1.0, "Transcription complete")

    # Step 5: Optional translation
    if translate:
        trans_vtt = paths["translation_vtt"]
        if not trans_vtt.is_file():
            from pgw.llm.translator import translate_subtitles
            from pgw.subtitles.converter import save_bilingual_vtt, save_subtitles

            emit("translate", 0.0, f"Translating to {translate}...")
            stage(f"Translating to {translate}", config.llm.model)

            def _on_translate_progress(frac: float) -> None:
                emit("translate", frac, f"Translating ({frac:.0%})...")

            trans_result = translate_subtitles(
                segments,
                language,
                translate,
                config.llm,
                chunk_size=chunk_size,
                on_progress=_on_translate_progress,
            )
            llm_was_used = True

            save_subtitles(trans_result.translated, trans_vtt, fmt="vtt")
            saved.append(trans_vtt)

            trans_txt = paths["translation_txt"]
            save_subtitles(trans_result.translated, trans_txt, fmt="txt")
            saved.append(trans_txt)

            # Bilingual VTT: original at bottom, translation at top
            bi_vtt = paths["bilingual_vtt"]
            save_bilingual_vtt(segments, trans_result.translated, bi_vtt)
            saved.append(bi_vtt)
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
                saved.append(pdf_path)
            except ImportError:
                pass
            except Exception as e:
                warning(f"PDF export skipped: {e}")

            try:
                from pgw.subtitles.export import export_parallel_epub

                epub_path = workspace / f"parallel.{language}-{translate}.epub"
                export_parallel_epub(
                    segments,
                    trans_result.translated,
                    epub_path,
                    language,
                    translate,
                    title=title,
                )
                saved.append(epub_path)
            except ImportError:
                pass
            except Exception as e:
                warning(f"EPUB export skipped: {e}")
        else:
            cache_hit("Translation cached")

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

        summary_path = workspace / f"vocabulary.{language}.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        saved.append(summary_path)
        stage(
            "Vocabulary",
            f"{summary['unique_lemmas']} unique lemmas, estimated {summary['estimated_level']}",
        )
    except ImportError:
        pass  # wordfreq or spacy not installed
    except Exception as e:
        warning(f"Vocabulary summary skipped: {e}")
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
            warning("mpv not found, skipping playback.")

    emit("save", 1.0, "Done", data={"workspace": str(workspace)})
    workspace_done(workspace, saved)
    return workspace
