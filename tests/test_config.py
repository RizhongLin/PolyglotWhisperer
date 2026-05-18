"""Tests for configuration system."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pgw.core.config import load_config


def test_cli_overrides():
    """CLI overrides take precedence over defaults."""
    config = load_config(
        **{"whisper.backend": "local", "whisper.local_model": "medium", "whisper.language": "de"}
    )
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


def test_llm_model_property_selects_backend():
    """LLM model property returns api_model when backend is api."""
    local = load_config(**{"llm.backend": "local"})
    assert local.llm.model == local.llm.local_model

    api = load_config(**{"llm.backend": "api"})
    assert api.llm.model == api.llm.api_model


@pytest.fixture
def env_isolation(monkeypatch):
    """Strip any PGW_* env vars so test cases start from a clean slate."""
    for key in list(os.environ):
        if key.startswith("PGW_"):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_env_var_populates_nested_field(env_isolation):
    """PGW_LLM__API_KEY reaches LLMConfig.api_key (regression: was silently dropped)."""
    env_isolation.setenv("PGW_LLM__API_KEY", "sk-from-env")
    env_isolation.setenv("PGW_LLM__API_BASE", "https://from-env.example/v1")
    config = load_config()
    assert config.llm.api_key == "sk-from-env"
    assert config.llm.api_base == "https://from-env.example/v1"


def test_env_var_overrides_toml_default(env_isolation):
    """Env var beats default.toml's api_base (Ollama URL → user's provider)."""
    env_isolation.setenv("PGW_LLM__API_BASE", "https://api.deepseek.com/v1")
    config = load_config()
    assert config.llm.api_base == "https://api.deepseek.com/v1"
    # Sibling fields still come from default.toml — env shouldn't shadow them.
    assert config.llm.local_model == "qwen3:8b"
    assert config.llm.temperature == 0.3


def test_cli_beats_env(env_isolation):
    """CLI flags take precedence over environment variables."""
    env_isolation.setenv("PGW_LLM__BACKEND", "local")
    config = load_config(**{"llm.backend": "api"})
    assert config.llm.backend == "api"


def test_cli_does_not_clobber_env_siblings(env_isolation):
    """A CLI override on one nested field must not drop env-supplied siblings."""
    env_isolation.setenv("PGW_LLM__API_KEY", "sk-from-env")
    env_isolation.setenv("PGW_LLM__API_BASE", "https://from-env.example/v1")
    config = load_config(**{"llm.backend": "api"})
    assert config.llm.backend == "api"
    assert config.llm.api_key == "sk-from-env"
    assert config.llm.api_base == "https://from-env.example/v1"


def test_project_toml_partial_override(tmp_path, env_isolation, monkeypatch):
    """Partial [llm] override in pgw.toml keeps default.toml's other fields."""
    monkeypatch.chdir(tmp_path)
    Path("pgw.toml").write_text("[llm]\ntemperature = 0.9\n", encoding="utf-8")
    config = load_config()
    assert config.llm.temperature == 0.9  # from project pgw.toml
    # Other [llm] fields fall through to packaged default.toml.
    assert config.llm.local_model == "qwen3:8b"
    assert config.llm.api_model == "deepseek-chat"
    assert config.llm.api_base == "http://localhost:11434/v1"


def test_chunk_size_precedence(tmp_path, env_isolation, monkeypatch):
    """CLI > env > TOML > auto for the new ``llm.chunk_size`` field."""
    from pgw.llm.translator import _chunk_params

    # Default (no override): None — caller picks via auto-detect.
    assert load_config().llm.chunk_size is None

    # TOML beats default.
    monkeypatch.chdir(tmp_path)
    Path("pgw.toml").write_text("[llm]\nchunk_size = 80\n", encoding="utf-8")
    cfg = load_config()
    assert cfg.llm.chunk_size == 80
    assert _chunk_params(cfg.llm)[0] == 80

    # Env beats TOML.
    env_isolation.setenv("PGW_LLM__CHUNK_SIZE", "120")
    cfg = load_config()
    assert cfg.llm.chunk_size == 120
    assert _chunk_params(cfg.llm)[0] == 120

    # Explicit argument beats config (simulates CLI ``--chunk-size``).
    assert _chunk_params(cfg.llm, chunk_size=200)[0] == 200


def test_cli_refine_overrides_refine_enabled():
    """--refine flag sets llm.refine_enabled via build_config_overrides → load_config."""
    from pgw.cli.utils import build_config_overrides

    overrides = build_config_overrides(language="fr", device="auto", refine=True)
    config = load_config(**overrides)
    assert config.llm.refine_enabled is True


def test_cli_refine_false_leaves_default():
    """Without --refine, refine_enabled stays at default (False)."""
    config = load_config()
    assert config.llm.refine_enabled is False
