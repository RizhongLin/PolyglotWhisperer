"""Tests for configuration system."""

from pgw.core.config import _deep_merge, load_config


def test_cli_overrides():
    """CLI overrides take precedence over defaults."""
    config = load_config(**{"whisper.model_size": "medium", "whisper.language": "de"})
    assert config.whisper.model_size == "medium"
    assert config.whisper.language == "de"


def test_cli_override_none_ignored():
    """None values in CLI overrides are ignored, defaults preserved."""
    default = load_config()
    overridden = load_config(**{"whisper.model_size": None})
    assert overridden.whisper.model_size == default.whisper.model_size


def test_deep_merge():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10, "e": 5}, "f": 6}
    result = _deep_merge(base, override)
    assert result == {"a": {"b": 10, "c": 2, "e": 5}, "d": 3, "f": 6}


def test_deep_merge_no_mutation():
    """Deep merge does not mutate the base dict."""
    base = {"a": {"b": 1}}
    override = {"a": {"c": 2}}
    _deep_merge(base, override)
    assert "c" not in base["a"]
