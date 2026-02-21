"""Tests for CLI utility functions — expand_inputs."""

import pytest

from pgw.cli.utils import expand_inputs


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
