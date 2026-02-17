"""Configuration system for PolyglotWhisperer.

Layered config loading (lowest to highest priority):
1. config/default.toml (shipped with package)
2. ~/.config/pgw/config.toml (user-level)
3. ./pgw.toml (project-level)
4. Environment variables (PGW_WHISPER__MODEL_SIZE, etc.)
5. CLI flags
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_CONFIG = _PACKAGE_ROOT / "config" / "default.toml"
_USER_CONFIG = Path.home() / ".config" / "pgw" / "config.toml"
_PROJECT_CONFIG = Path("pgw.toml")


class WhisperConfig(BaseModel):
    model_size: str = "large-v3"
    language: str = "fr"
    device: str = "auto"
    compute_type: str = "float16"
    batch_size: int = 16
    word_timestamps: bool = True


class LLMConfig(BaseModel):
    provider: str = "ollama_chat/qwen3:8b"
    api_base: str = "http://localhost:11434"
    temperature: float = 0.3
    max_tokens: int = 4096
    cleanup_enabled: bool = True
    translation_enabled: bool = True
    target_language: str = "en"


class DownloadConfig(BaseModel):
    output_dir: Path = Path("./downloads")
    format: str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]"


class PlayerConfig(BaseModel):
    backend: str = "mpv"
    sub_font_size: int = 40
    secondary_sub_font_size: int = 32


class PGWConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PGW_",
        env_nested_delimiter="__",
    )

    whisper: WhisperConfig = WhisperConfig()
    llm: LLMConfig = LLMConfig()
    download: DownloadConfig = DownloadConfig()
    player: PlayerConfig = PlayerConfig()
    workspace_dir: Path = Path("./pgw_workspace")


def _load_toml(path: Path) -> dict:
    """Load a TOML file if it exists, return empty dict otherwise."""
    if path.is_file():
        with open(path, "rb") as f:
            return tomllib.load(f)
    return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(**cli_overrides: object) -> PGWConfig:
    """Load configuration from all layers and merge.

    Args:
        **cli_overrides: Direct overrides from CLI flags. Keys can be
            dot-separated (e.g. whisper.model_size="medium").
    """
    # Layer 1-3: TOML files
    config_data: dict = {}
    for path in (_DEFAULT_CONFIG, _USER_CONFIG, _PROJECT_CONFIG):
        layer = _load_toml(path)
        config_data = _deep_merge(config_data, layer)

    # Flatten 'general' section into top-level
    if "general" in config_data:
        general = config_data.pop("general")
        config_data = _deep_merge(config_data, general)

    # Apply CLI overrides (dot-separated keys)
    for key, value in cli_overrides.items():
        if value is None:
            continue
        parts = key.split(".")
        target = config_data
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value

    # Layer 4: env vars are handled by Pydantic BaseSettings
    return PGWConfig(**config_data)
