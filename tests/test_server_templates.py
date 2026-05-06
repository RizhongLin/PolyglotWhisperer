"""Tests for server template rendering functions."""

import json

from pgw.server.templates import (
    _build_html,
    _build_library_html,
    _build_metadata_rows,
    _discover_tracks,
    _format_duration,
    _format_file_size,
    _friendly_name,
    _lang_full,
    _lang_short,
    _load_metadata,
)


def test_lang_short():
    assert _lang_short("en") == "EN"
    assert _lang_short("fr") == "FR"
    assert _lang_short("zh") == "ZH"


def test_lang_full_known():
    assert "French" in _lang_full("fr")
    assert "fr" in _lang_full("fr")


def test_lang_full_unknown():
    result = _lang_full("xyz")
    assert "(" not in result


def test_format_duration():
    assert _format_duration(None) == ""
    assert _format_duration(0) == ""
    assert _format_duration(65) == "1:05"
    assert _format_duration(3661) == "1:01:01"


def test_format_file_size():
    assert _format_file_size(500) == "500 B"
    assert _format_file_size(1500) == "1 KB"
    assert _format_file_size(1_500_000) == "1.4 MB"


def test_load_metadata_missing_file(tmp_path):
    result = _load_metadata(tmp_path)
    assert result == {}


def test_load_metadata_valid(tmp_path):
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps({"title": "Test", "language": "fr"}))
    result = _load_metadata(tmp_path)
    assert result["title"] == "Test"
    assert result["language"] == "fr"


def test_load_metadata_invalid_json(tmp_path):
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text("not json")
    result = _load_metadata(tmp_path)
    assert result == {}


def test_friendly_name_transcription():
    name = _friendly_name("transcription.fr.vtt")
    assert "Original Transcription" in name
    assert "French" in name


def test_friendly_name_bilingual():
    name = _friendly_name("bilingual.fr-en.vtt")
    assert "Bilingual" in name


def test_friendly_name_vocabulary():
    name = _friendly_name("vocabulary.fr.json")
    assert "Vocabulary" in name


def test_build_metadata_rows_basic():
    meta = {
        "language": "fr",
        "whisper_model": "whisper-large-v3",
        "source_duration": 123.0,
    }
    rows = _build_metadata_rows(meta)
    assert "French" in rows
    assert "whisper-large-v3" in rows
    assert "2:03" in rows


def test_build_metadata_rows_with_target():
    meta = {"language": "fr", "target_language": "en"}
    rows = _build_metadata_rows(meta)
    assert "Languages" in rows
    assert "French" in rows
    assert "English" in rows


def test_build_metadata_rows_with_description():
    meta = {"language": "en", "description": "A" * 300}
    rows = _build_metadata_rows(meta)
    assert "…" in rows


def test_build_metadata_rows_empty():
    rows = _build_metadata_rows({})
    assert rows == ""


def test_build_html_player(tmp_path, monkeypatch):
    """Smoke test: _build_html produces valid HTML structure."""
    # Create workspace with metadata
    workspace = tmp_path / "slug" / "20260310_120000"
    workspace.mkdir(parents=True)
    (workspace / "metadata.json").write_text(json.dumps({"title": "Test Video", "language": "fr"}))

    html = _build_html(workspace, video_path=None)
    assert "<!DOCTYPE html>" in html or "<html" in html
    assert "Test Video" in html
    # Should show video-missing section when no video
    assert "Video not available" in html


def test_build_html_with_video(tmp_path, monkeypatch):
    """Smoke test: _build_html with a video path produces a video tag."""
    workspace = tmp_path / "slug" / "20260310_120000"
    workspace.mkdir(parents=True)
    (workspace / "metadata.json").write_text(json.dumps({"title": "Test", "language": "en"}))
    video = workspace / "test.mp4"
    video.touch()

    html = _build_html(workspace, video_path=video)
    assert '<video id="player"' in html
    assert "test.mp4" in html


def test_build_html_with_library_url(tmp_path):
    workspace = tmp_path / "slug" / "20260310_120000"
    workspace.mkdir(parents=True)
    (workspace / "metadata.json").write_text(json.dumps({"title": "Test", "language": "en"}))

    html = _build_html(workspace, video_path=None, library_url="/")
    assert "arrow-left" in html  # back arrow icon


def test_build_library_html_empty():
    html = _build_library_html([])
    assert "No workspaces found" in html
    assert "pgw run" in html


def test_build_library_html_with_workspaces(tmp_path):
    ws = tmp_path / "slug" / "20260310_120000"
    ws.mkdir(parents=True)
    (ws / "metadata.json").write_text(
        json.dumps(
            {
                "title": "My Video",
                "language": "fr",
                "source_duration": 180,
                "uploader": "TestChannel",
                "created_at": "2026-03-10T12:00:00",
            }
        )
    )

    workspaces = [
        {
            "path": ws,
            "slug": "slug",
            "timestamp": "20260310_120000",
            "title": "My Video",
            "language": "fr",
            "target_language": "",
            "duration": 180,
            "created_at": "2026-03-10T12:00:00",
            "has_video": False,
            "upload_date": "",
            "uploader": "TestChannel",
            "thumbnail": "",
            "lang_pairs": [("fr", "")],
            "sibling_paths": [],
        }
    ]

    html = _build_library_html(workspaces)
    assert "My Video" in html
    assert "TestChannel" in html
    assert "3:00" in html


def test_discover_tracks_no_files(tmp_path):
    """No subtitle files → empty track list."""
    tracks = _discover_tracks(tmp_path)
    assert tracks == []


def test_discover_tracks_bilingual_preferred(tmp_path):
    """Bilingual VTT appears first."""
    ws = tmp_path
    (ws / "bilingual.fr-en.vtt").touch()
    (ws / "transcription.fr.vtt").touch()
    (ws / "translation.en.vtt").touch()

    tracks = _discover_tracks(ws)
    assert len(tracks) == 3
    assert "Bilingual" in tracks[0]["label"]
