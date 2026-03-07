"""Workspace directory management for per-video output."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

# ── Project metadata ──
GITHUB_URL = "https://github.com/RizhongLin/PolyglotWhisperer"

# ── Workspace file names ──
# Fixed filenames
METADATA_FILE = "metadata.json"
AUDIO_FILE = "audio.wav"

# Stem prefixes used to build filenames like transcription.fr.vtt
STEM_BILINGUAL = "bilingual"
STEM_TRANSCRIPTION = "transcription"
STEM_TRANSLATION = "translation"
STEM_VOCABULARY = "vocabulary"
STEM_PARALLEL = "parallel"

# Glob patterns for discovering workspace files
GLOB_BILINGUAL_VTT = f"{STEM_BILINGUAL}.*.vtt"
GLOB_TRANSCRIPTION_VTT = f"{STEM_TRANSCRIPTION}.*.vtt"
GLOB_TRANSLATION_VTT = f"{STEM_TRANSLATION}.*.vtt"
GLOB_VOCABULARY_JSON = f"{STEM_VOCABULARY}.*.json"


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text[:80].strip("-")


def create_workspace(
    title: str,
    base_dir: Path = Path("./pgw_workspace"),
) -> Path:
    """Create a timestamped workspace directory for a video.

    Structure: <base_dir>/<slug>/<YYYYMMDD_HHMMSS>/
    Groups multiple runs of the same source under one parent slug dir.
    """
    slug = slugify(title) or "untitled"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = base_dir / slug / timestamp
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".ts", ".flv")


def find_video(workspace: Path) -> Path | None:
    """Find the video file in a workspace by globbing known extensions.

    Returns the first match, or None if no video is found.
    """
    for ext in _VIDEO_EXTENSIONS:
        candidates = sorted(workspace.glob(f"video{ext}"))
        for c in candidates:
            if c.is_file():  # Skip broken symlinks
                return c
    return None


def workspace_paths(
    workspace: Path, language: str, target_lang: str | None = None, video_ext: str = ".mp4"
) -> dict:
    """Generate standard output paths for a workspace.

    Returns a dict with keys: video, audio, transcription_vtt, transcription_txt,
    translation_vtt, translation_txt, bilingual_vtt, metadata.
    """
    paths = {
        "video": workspace / f"video{video_ext}",
        "audio": workspace / AUDIO_FILE,
        "transcription_vtt": workspace / f"{STEM_TRANSCRIPTION}.{language}.vtt",
        "transcription_txt": workspace / f"{STEM_TRANSCRIPTION}.{language}.txt",
        "metadata": workspace / METADATA_FILE,
    }
    if target_lang:
        paths["translation_vtt"] = workspace / f"{STEM_TRANSLATION}.{target_lang}.vtt"
        paths["translation_txt"] = workspace / f"{STEM_TRANSLATION}.{target_lang}.txt"
        paths["bilingual_vtt"] = workspace / f"{STEM_BILINGUAL}.{language}-{target_lang}.vtt"
    return paths


def save_metadata(workspace: Path, **kwargs: object) -> Path:
    """Save processing metadata to the workspace.

    Creates a comprehensive metadata.json with source info, processing
    parameters, output file inventory, and timing.
    """
    meta_path = workspace / METADATA_FILE

    # Build file inventory from workspace contents
    files = {}
    for f in sorted(workspace.iterdir()):
        if f.name == METADATA_FILE or f.name.startswith("."):
            continue
        files[f.name] = {
            "size_bytes": f.stat().st_size,
            "type": _classify_file(f.name),
        }

    data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    data.update({k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()})

    meta_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


def _classify_file(name: str) -> str:
    """Classify a workspace file by its name."""
    if name.startswith("video"):
        return "source_video"
    if name.startswith("audio"):
        return "extracted_audio"
    if name.startswith("bilingual") and name.endswith(".vtt"):
        return "bilingual_subtitle"
    if name.startswith("transcription") and name.endswith((".srt", ".vtt")):
        return "transcription_subtitle"
    if name.startswith("transcription") and name.endswith(".txt"):
        return "transcription_text"
    if name.startswith("translation") and name.endswith((".srt", ".vtt")):
        return "translation_subtitle"
    if name.startswith("translation") and name.endswith(".txt"):
        return "translation_text"
    return "other"
