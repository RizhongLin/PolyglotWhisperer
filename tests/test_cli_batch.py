"""Tests for CLI batch utilities and config override logic."""

from pgw.cli.utils import build_config_overrides, expand_inputs


def test_expand_inputs_urls():
    result = expand_inputs(["https://example.com/video", "http://foo.bar/baz"])
    assert result == ["https://example.com/video", "http://foo.bar/baz"]


def test_expand_inputs_local_files():
    result = expand_inputs(["file.mp4", "another.mkv"])
    assert result == ["file.mp4", "another.mkv"]


def test_expand_inputs_txt_file(tmp_path):
    list_file = tmp_path / "urls.txt"
    list_file.write_text("https://a.com/v1\n# comment\nhttps://b.com/v2\n  \nfile.mp4\n")
    result = expand_inputs([str(list_file)])
    assert "https://a.com/v1" in result
    assert "https://b.com/v2" in result
    assert "file.mp4" in result
    assert "# comment" not in result
    assert "" not in result


def test_expand_inputs_glob(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "vid1.mp4").touch()
    (tmp_path / "vid2.mp4").touch()
    (tmp_path / "other.mkv").touch()

    result = expand_inputs(["*.mp4"])
    assert len(result) == 2
    assert "vid1.mp4" in result
    assert "vid2.mp4" in result


def test_expand_inputs_mixed():
    result = expand_inputs(["https://youtube.com/watch", "local.mp4"])
    assert len(result) == 2
    assert result[0] == "https://youtube.com/watch"
    assert result[1] == "local.mp4"


def test_build_config_overrides_language():
    overrides = build_config_overrides(language="fr", device="cpu")
    assert overrides["whisper.language"] == "fr"
    assert overrides["whisper.device"] == "cpu"


def test_build_config_overrides_whisper_model_local():
    overrides = build_config_overrides(
        language="en", device="cpu", whisper_model="large-v3", backend=None
    )
    assert overrides["whisper.local_model"] == "large-v3"


def test_build_config_overrides_whisper_model_api():
    overrides = build_config_overrides(
        language="en", device="cpu", whisper_model="groq/whisper-large", backend="api"
    )
    assert overrides["whisper.api_model"] == "groq/whisper-large"


def test_build_config_overrides_llm_model_api():
    overrides = build_config_overrides(
        language="en", device="cpu", llm_model="groq/gpt-oss", llm_backend="api"
    )
    assert overrides["llm.api_model"] == "groq/gpt-oss"


def test_build_config_overrides_llm_model_local():
    overrides = build_config_overrides(
        language="en", device="cpu", llm_model="ollama/qwen", llm_backend=None
    )
    assert overrides["llm.local_model"] == "ollama/qwen"


def test_build_config_overrides_backend():
    overrides = build_config_overrides(
        language="en", device="cpu", backend="api", llm_backend="api"
    )
    assert overrides["whisper.backend"] == "api"
    assert overrides["llm.backend"] == "api"


def test_build_config_overrides_translate():
    overrides = build_config_overrides(language="en", device="cpu", translate="zh")
    assert overrides["llm.target_language"] == "zh"


def test_build_config_overrides_subs():
    overrides = build_config_overrides(language="en", device="cpu", subs=True)
    assert overrides["download.subtitles"] is True
