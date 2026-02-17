"""Tests for workspace directory management utilities."""

import json

from pgw.utils.paths import create_workspace, save_metadata, slugify, workspace_paths


class TestSlugify:
    def test_basic(self):
        assert slugify("Hello World") == "hello-world"

    def test_special_characters(self):
        assert slugify("RTS 19h30 — Édition spéciale!") == "rts-19h30-édition-spéciale"

    def test_collapses_dashes(self):
        assert slugify("a---b   c") == "a-b-c"

    def test_strips_leading_trailing(self):
        assert slugify("--hello--") == "hello"

    def test_truncates_long_strings(self):
        result = slugify("a" * 200)
        assert len(result) <= 80

    def test_empty_string(self):
        assert slugify("") == ""


class TestCreateWorkspace:
    def test_creates_directory(self, tmp_path):
        ws = create_workspace("My Video", base_dir=tmp_path)
        assert ws.is_dir()

    def test_nested_structure(self, tmp_path):
        """Workspace is <base>/<slug>/<timestamp>/."""
        ws = create_workspace("My Video", base_dir=tmp_path)
        # ws.parent is the slug dir, ws.parent.parent is base_dir
        assert ws.parent.name == "my-video"
        assert ws.parent.parent == tmp_path

    def test_has_timestamp(self, tmp_path):
        ws = create_workspace("test", base_dir=tmp_path)
        # Workspace name is the timestamp: YYYYMMDD_HHMMSS
        assert len(ws.name) == 15  # YYYYMMDD_HHMMSS
        assert ws.parent.name == "test"

    def test_untitled_fallback(self, tmp_path):
        ws = create_workspace("!!!", base_dir=tmp_path)
        assert ws.parent.name == "untitled"

    def test_multiple_runs_same_source(self, tmp_path):
        """Multiple runs of the same title share the slug parent dir."""
        ws1 = create_workspace("My Video", base_dir=tmp_path)
        ws2 = create_workspace("My Video", base_dir=tmp_path)
        assert ws1.parent == ws2.parent  # same slug dir
        # Both share the "my-video" parent
        assert ws1.parent.name == "my-video"


class TestWorkspacePaths:
    def test_without_translation(self, tmp_path):
        paths = workspace_paths(tmp_path, "fr")
        assert paths["video"] == tmp_path / "video.mp4"
        assert paths["audio"] == tmp_path / "audio.wav"
        assert paths["transcription_vtt"] == tmp_path / "transcription.fr.vtt"
        assert paths["transcription_txt"] == tmp_path / "transcription.fr.txt"
        assert paths["metadata"] == tmp_path / "metadata.json"
        assert "translation_vtt" not in paths

    def test_with_translation(self, tmp_path):
        paths = workspace_paths(tmp_path, "fr", target_lang="en")
        assert paths["translation_vtt"] == tmp_path / "translation.en.vtt"
        assert paths["translation_txt"] == tmp_path / "translation.en.txt"
        assert paths["bilingual_vtt"] == tmp_path / "bilingual.fr-en.vtt"


class TestSaveMetadata:
    def test_saves_json(self, tmp_path):
        meta_path = save_metadata(tmp_path, title="Test", language="fr")
        assert meta_path.is_file()
        data = json.loads(meta_path.read_text())
        assert data["title"] == "Test"
        assert data["language"] == "fr"
        assert "created_at" in data
        assert "files" in data

    def test_includes_file_inventory(self, tmp_path):
        # Create some workspace files
        (tmp_path / "video.mp4").write_bytes(b"fake video")
        (tmp_path / "audio.wav").write_bytes(b"fake audio")
        (tmp_path / "transcription.fr.vtt").write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello"
        )

        meta_path = save_metadata(tmp_path, title="Test")
        data = json.loads(meta_path.read_text())

        assert "video.mp4" in data["files"]
        assert data["files"]["video.mp4"]["type"] == "source_video"
        assert data["files"]["audio.wav"]["type"] == "extracted_audio"
        assert data["files"]["transcription.fr.vtt"]["type"] == "transcription_subtitle"
