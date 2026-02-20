"""Tests for configuration system."""

from pgw.core.config import _deep_merge, load_config


def test_cli_overrides():
    """CLI overrides take precedence over defaults."""
    config = load_config(**{"whisper.local_model": "medium", "whisper.language": "de"})
    assert config.whisper.local_model == "medium"
    assert config.whisper.model == "medium"  # property returns local_model
    assert config.whisper.language == "de"


def test_cli_override_none_ignored():
    """None values in CLI overrides are ignored, defaults preserved."""
    default = load_config()
    overridden = load_config(**{"whisper.local_model": None})
    assert overridden.whisper.local_model == default.whisper.local_model


def test_model_property_selects_backend():
    """model property returns api_model when backend is api."""
    local = load_config(**{"whisper.backend": "local"})
    assert local.whisper.model == local.whisper.local_model

    api = load_config(**{"whisper.backend": "api"})
    assert api.whisper.model == api.whisper.api_model


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
