"""Configuration system for PolyglotWhisperer.

Layered config loading (lowest to highest priority):
1. config/default.toml (shipped with package)
2. ~/.config/pgw/config.toml (user-level)
3. ./pgw.toml (project-level)
4. Environment variables (PGW_WHISPER__MODEL_SIZE, etc.) and .env file
5. CLI flags (passed as nested dict to PGWConfig)

Each source is a separate ``PydanticBaseSettingsSource`` so pydantic-settings
deep-merges them field-by-field. This is critical for nested models: e.g.
``PGW_LLM__API_KEY`` must override ``[llm]`` defaults from TOML even though
the TOML layer fills in other ``[llm]`` fields. Passing the merged TOML dict
as a single ``__init__`` kwarg would silently shadow env-var loading for
every nested field.
"""

from __future__ import annotations

import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

_USER_CONFIG = Path.home() / ".config" / "pgw" / "config.toml"
_PROJECT_CONFIG = Path("pgw.toml")


class BackendConfig(BaseModel):
    """Base for configs with local/api backend selection."""

    backend: str = "local"  # "local" or "api"
    local_model: str = ""
    api_model: str = ""
    api_base: str = ""  # Custom API endpoint (OpenAI-compatible server, etc.)
    api_key: str = ""  # API key for the custom endpoint (env vars preferred)

    @property
    def model(self) -> str:
        """Return the model for the active backend."""
        return self.api_model if self.backend == "api" else self.local_model


class WhisperConfig(BackendConfig):
    local_model: str = "large-v3-turbo"
    api_model: str = "whisper-large-v3-turbo"
    api_base: str = ""
    language: str = "fr"
    device: str = "auto"
    word_timestamps: bool = True


class LLMConfig(BackendConfig):
    local_model: str = "qwen3:8b"
    api_model: str = "deepseek-chat"
    api_base: str = "http://localhost:11434/v1"
    temperature: float = 0.3
    max_tokens: int = 32768
    timeout: int = 600
    num_retries: int = 2
    refine_enabled: bool = False
    translation_enabled: bool = True
    target_language: str = "en"
    # Segments per LLM call. ``None`` lets the caller auto-pick from backend
    # and (for local) model size. Override via ``--chunk-size`` (CLI),
    # ``PGW_LLM__CHUNK_SIZE`` (env), or ``[llm] chunk_size = N`` in pgw.toml.
    chunk_size: int | None = None


class DownloadConfig(BaseModel):
    format: str = "bv*[ext=mp4]+ba[ext=m4a]/bv*+ba/b"
    subtitles: bool = False  # Download existing subtitles from video pages


class PlayerConfig(BaseModel):
    backend: str = "mpv"
    sub_font_size: int = 40


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file if it exists, return empty dict otherwise."""
    if path.is_file():
        with open(path, "rb") as f:
            return tomllib.load(f)
    return {}


def _load_packaged_default() -> dict[str, Any]:
    """Load the packaged default.toml shipped with the wheel."""
    return tomllib.loads((files("pgw.config") / "default.toml").read_text())


def _flatten_general(data: dict[str, Any]) -> dict[str, Any]:
    """Promote keys from a [general] section to top-level."""
    if "general" in data:
        general = data["general"]
        merged = {k: v for k, v in data.items() if k != "general"}
        for key, value in general.items():
            merged.setdefault(key, value)
        return merged
    return data


class _TomlSource(PydanticBaseSettingsSource):
    """Settings source backed by a single TOML dict already loaded in memory."""

    def __init__(self, settings_cls: type[BaseSettings], data: dict[str, Any]) -> None:
        super().__init__(settings_cls)
        self._data = _flatten_general(data)

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        value = self._data.get(field_name)
        return value, field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data


class PGWConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PGW_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    whisper: WhisperConfig = WhisperConfig()
    llm: LLMConfig = LLMConfig()
    download: DownloadConfig = DownloadConfig()
    player: PlayerConfig = PlayerConfig()
    workspace_dir: Path = Path("./pgw_workspace")

    @property
    def download_dir(self) -> Path:
        """Download cache directory, under the shared workspace cache."""
        return self.workspace_dir / ".cache" / "downloads"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Sources are consulted in order; earlier wins per-field.
        # Precedence (highest first): CLI kwargs > env > .env > project.toml
        # > user.toml > packaged default.toml.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _TomlSource(settings_cls, _load_toml(_PROJECT_CONFIG)),
            _TomlSource(settings_cls, _load_toml(_USER_CONFIG)),
            _TomlSource(settings_cls, _load_packaged_default()),
        )


def _expand_dot_paths(overrides: dict[str, object]) -> dict[str, Any]:
    """Convert {'llm.backend': 'api'} → {'llm': {'backend': 'api'}}, dropping None."""
    nested: dict[str, Any] = {}
    for key, value in overrides.items():
        if value is None:
            continue
        parts = key.split(".")
        target = nested
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return nested


def load_config(**cli_overrides: object) -> PGWConfig:
    """Load configuration from all layers and merge.

    Args:
        **cli_overrides: Direct overrides from CLI flags. Keys can be
            dot-separated (e.g. whisper.local_model="medium") to target
            nested models. ``None`` values are dropped so absent flags do
            not override env/TOML.
    """
    return PGWConfig(**_expand_dot_paths(cli_overrides))
