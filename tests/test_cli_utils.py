"""Tests for CLI utility functions."""

import pytest

from pgw.cli.utils import build_config_overrides, expand_inputs


class TestExpandInputs:
    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/video.mp4",
            "http://example.com/video.mp4",
            "ftp://example.com/file.mp4",
        ],
        ids=["https", "http", "ftp"],
    )
    def test_url_passthrough(self, url):
        assert expand_inputs([url]) == [url]

    def test_txt_file_expansion(self, tmp_path):
        txt_file = tmp_path / "urls.txt"
        txt_file.write_text("https://a.com/1.mp4\n# comment\nhttps://b.com/2.mp4\n\n")
        result = expand_inputs([str(txt_file)])
        assert result == ["https://a.com/1.mp4", "https://b.com/2.mp4"]

    def test_txt_file_skips_blank_lines(self, tmp_path):
        txt_file = tmp_path / "list.txt"
        txt_file.write_text("\n\nhello\n\n")
        result = expand_inputs([str(txt_file)])
        assert result == ["hello"]

    def test_glob_expansion(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a.mp4").touch()
        (tmp_path / "b.mp4").touch()
        (tmp_path / "c.mkv").touch()
        result = expand_inputs(["*.mp4"])
        assert len(result) == 2
        assert any("a.mp4" in r for r in result)
        assert any("b.mp4" in r for r in result)

    def test_regular_path_passthrough(self):
        result = expand_inputs(["/some/path/video.mp4"])
        assert result == ["/some/path/video.mp4"]

    def test_mixed_inputs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "video.mp4").touch()
        txt_file = tmp_path / "urls.txt"
        txt_file.write_text("https://example.com/v.mp4\n")

        result = expand_inputs(["https://a.com/x.mp4", str(txt_file), "*.mp4"])
        assert "https://a.com/x.mp4" in result
        assert "https://example.com/v.mp4" in result
        assert any("video.mp4" in r for r in result)

    def test_empty_input(self):
        assert expand_inputs([]) == []

    def test_nonexistent_glob_returns_literal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # No matching files — glob pattern with no matches returns the literal
        result = expand_inputs(["*.nonexistent"])
        assert result == ["*.nonexistent"]


class TestBuildConfigOverrides:
    def test_refine_true_sets_refine_enabled(self):
        overrides = build_config_overrides(language="fr", device="auto", refine=True)
        assert overrides["llm.refine_enabled"] is True

    def test_refine_false_does_not_set_refine_enabled(self):
        overrides = build_config_overrides(language="fr", device="auto")
        assert "llm.refine_enabled" not in overrides

    def test_translate_sets_target_language(self):
        overrides = build_config_overrides(language="fr", device="auto", translate="en")
        assert overrides["llm.target_language"] == "en"

    def test_subs_true_sets_download_subtitles(self):
        overrides = build_config_overrides(language="fr", device="auto", subs=True)
        assert overrides["download.subtitles"] is True

    def test_backend_sets_whisper_backend(self):
        overrides = build_config_overrides(language="fr", device="auto", backend="api")
        assert overrides["whisper.backend"] == "api"

    def test_llm_backend_sets_llm_backend(self):
        overrides = build_config_overrides(language="fr", device="auto", llm_backend="api")
        assert overrides["llm.backend"] == "api"

    def test_whisper_model_api_backend(self):
        overrides = build_config_overrides(
            language="fr", device="auto", whisper_model="whisper-1", backend="api"
        )
        assert overrides["whisper.api_model"] == "whisper-1"

    def test_whisper_model_local_backend(self):
        overrides = build_config_overrides(
            language="fr", device="auto", whisper_model="large-v3", backend="local"
        )
        assert overrides["whisper.local_model"] == "large-v3"

    def test_llm_model_api_backend(self):
        overrides = build_config_overrides(
            language="fr", device="auto", llm_model="gpt-4o", llm_backend="api"
        )
        assert overrides["llm.api_model"] == "gpt-4o"

    def test_llm_model_local_backend(self):
        overrides = build_config_overrides(
            language="fr", device="auto", llm_model="gemma3:12b", llm_backend="local"
        )
        assert overrides["llm.local_model"] == "gemma3:12b"

    def test_refine_and_translate_together(self):
        """refine + translate should both be reflected in overrides."""
        overrides = build_config_overrides(
            language="fr", device="auto", refine=True, translate="en"
        )
        assert overrides["llm.refine_enabled"] is True
        assert overrides["llm.target_language"] == "en"

    def test_defaults_no_optional_overrides(self):
        overrides = build_config_overrides(language="fr", device="auto")
        assert "llm.target_language" not in overrides
        assert "download.subtitles" not in overrides
        assert "llm.refine_enabled" not in overrides
