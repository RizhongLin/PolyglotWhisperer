"""Tests for response_format fallback chains in LLM and transcription clients."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIConnectionError, BadRequestError

from pgw.core.config import LLMConfig, WhisperConfig
from pgw.llm import client as llm_client
from pgw.transcriber import api as transcribe_api


def _bad_request(message: str) -> BadRequestError:
    request = httpx.Request("POST", "https://x.example/v1/x")
    response = httpx.Response(400, json={"error": {"message": message}}, request=request)
    return BadRequestError(message=message, response=response, body={"error": {"message": message}})


def _connection_error() -> APIConnectionError:
    request = httpx.Request("POST", "https://x.example/v1/x")
    return APIConnectionError(request=request)


def _success_response(content: str = '{"1": "ok"}') -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


@pytest.fixture(autouse=True)
def reset_caches():
    llm_client._RESPONSE_FORMAT_TIER.clear()
    transcribe_api._TRANSCRIPTION_TIER.clear()
    yield
    llm_client._RESPONSE_FORMAT_TIER.clear()
    transcribe_api._TRANSCRIPTION_TIER.clear()


class TestLLMFallback:
    def _config(self, model: str = "deepseek-chat") -> LLMConfig:
        return LLMConfig(
            backend="api",
            api_base="https://api.deepseek.com/v1",
            api_key="x",
            api_model=model,
        )

    def test_schema_to_object_to_none(self):
        """Schema rejected, object rejected, none succeeds — cache locks at 'none'."""
        attempts: list[dict | None] = []

        def fake_create(**kwargs):
            attempts.append(kwargs.get("response_format"))
            fmt = kwargs.get("response_format")
            if fmt is not None:
                raise _bad_request("This response_format type is unavailable now")
            return _success_response()

        client = MagicMock()
        client.chat.completions.create = fake_create

        cfg = self._config()
        schema = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
        llm_client._create_with_format_fallback(client, {"model": cfg.model}, schema, cfg)

        assert [a["type"] if a else None for a in attempts] == [
            "json_schema",
            "json_object",
            None,
        ]
        assert (
            llm_client._RESPONSE_FORMAT_TIER[("https://api.deepseek.com/v1", "deepseek-chat")]
            == "none"
        )

    def test_cache_hit_skips_discovery(self):
        """After discovery, subsequent calls go straight to the cached tier."""
        client = MagicMock()
        client.chat.completions.create = MagicMock(return_value=_success_response())

        cfg = self._config()
        llm_client._RESPONSE_FORMAT_TIER[("https://api.deepseek.com/v1", "deepseek-chat")] = (
            "object"
        )

        llm_client._create_with_format_fallback(
            client, {"model": cfg.model}, json_schema={"any": "schema"}, config=cfg
        )

        client.chat.completions.create.assert_called_once()
        sent = client.chat.completions.create.call_args.kwargs.get("response_format")
        assert sent == {"type": "json_object"}

    def test_non_format_error_reraises_without_downgrade(self):
        """Context-length / connection errors must NOT trigger fallback."""
        attempts: list[dict | None] = []

        def fake_create(**kwargs):
            attempts.append(kwargs.get("response_format"))
            raise _bad_request("This model's maximum context length is 16384 tokens")

        client = MagicMock()
        client.chat.completions.create = fake_create
        cfg = self._config()

        with pytest.raises(BadRequestError):
            llm_client._create_with_format_fallback(
                client,
                {"model": cfg.model},
                json_schema={"type": "json_schema"},
                config=cfg,
            )

        # Should have only attempted the first tier, then bailed out.
        assert len(attempts) == 1
        # Cache must NOT be poisoned by an unrelated error.
        assert (
            "https://api.deepseek.com/v1",
            "deepseek-chat",
        ) not in llm_client._RESPONSE_FORMAT_TIER

    def test_transient_network_error_reraises(self):
        """APIConnectionError is not a 4xx — propagate immediately."""
        client = MagicMock()
        client.chat.completions.create = MagicMock(side_effect=_connection_error())
        cfg = self._config()

        with pytest.raises(APIConnectionError):
            llm_client._create_with_format_fallback(
                client, {"model": cfg.model}, json_schema={"x": 1}, config=cfg
            )

    def test_apply_tier_schema_without_payload_degrades_to_object(self):
        """Cached tier='schema' but no schema (retry path) → still send JSON mode."""
        params: dict = {}
        llm_client._apply_tier(params, "schema", json_schema=None)
        assert params["response_format"] == {"type": "json_object"}


class TestTranscriptionFallback:
    def _config(self) -> WhisperConfig:
        return WhisperConfig(
            backend="api",
            api_base="https://api.groq.com/openai/v1",
            api_key="x",
            api_model="whisper-large-v3-turbo",
            language="en",
        )

    def test_verbose_words_to_verbose_to_json(self, tmp_path: Path):
        """Each tier rejected in turn until plain json succeeds."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00\x00")

        attempts: list[tuple[str | None, list[str] | None]] = []

        def fake_create(**kwargs):
            attempts.append((kwargs.get("response_format"), kwargs.get("timestamp_granularities")))
            fmt = kwargs.get("response_format")
            granularities = kwargs.get("timestamp_granularities")
            if fmt == "verbose_json" and granularities == ["word"]:
                raise _bad_request("timestamp_granularities is not supported")
            if fmt == "verbose_json":
                raise _bad_request("response_format verbose_json is unavailable")
            response = MagicMock()
            response.text = "hello world"
            return response

        client = MagicMock()
        client.audio.transcriptions.create = fake_create

        cfg = self._config()
        transcribe_api._transcribe_with_format_fallback(client, audio, cfg)

        assert attempts == [
            ("verbose_json", ["word"]),
            ("verbose_json", None),
            ("json", None),
        ]
        cache_key = ("https://api.groq.com/openai/v1", "whisper-large-v3-turbo")
        assert transcribe_api._TRANSCRIPTION_TIER[cache_key] == "json"

    def test_cache_hit_skips_discovery(self, tmp_path: Path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00\x00")
        client = MagicMock()
        client.audio.transcriptions.create = MagicMock(return_value=MagicMock())

        cfg = self._config()
        cache_key = ("https://api.groq.com/openai/v1", "whisper-large-v3-turbo")
        transcribe_api._TRANSCRIPTION_TIER[cache_key] = "verbose_json"

        transcribe_api._transcribe_with_format_fallback(client, audio, cfg)

        client.audio.transcriptions.create.assert_called_once()
        kwargs = client.audio.transcriptions.create.call_args.kwargs
        assert kwargs["response_format"] == "verbose_json"
        assert "timestamp_granularities" not in kwargs

    def test_non_format_error_reraises(self, tmp_path: Path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00\x00")
        client = MagicMock()
        client.audio.transcriptions.create = MagicMock(side_effect=_bad_request("File too large"))

        cfg = self._config()
        with pytest.raises(BadRequestError):
            transcribe_api._transcribe_with_format_fallback(client, audio, cfg)

        cache_key = ("https://api.groq.com/openai/v1", "whisper-large-v3-turbo")
        assert cache_key not in transcribe_api._TRANSCRIPTION_TIER
