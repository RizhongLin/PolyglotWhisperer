"""Unified LLM client via OpenAI SDK with structured output support."""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

from pgw.core.config import LLMConfig
from pgw.utils.console import debug, stage, warning

# Discovered response_format support per (api_base, model). Set on the first
# call that exercises a response_format and reused for all subsequent calls so
# we don't pay the fallback cost on every request.
# Values: "schema" (strict JSON schema), "object" (json_object), "none".
_RESPONSE_FORMAT_TIER: dict[tuple[str, str], str] = {}


def _ensure_ollama_model(api_base: str, model: str) -> None:
    """Pull the Ollama model if not already available locally.

    Uses subprocess since there is no Ollama Python dependency.
    No-op if api_base does not point to a local Ollama instance.
    """
    if not api_base or "11434" not in api_base:
        return

    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return

    if model in result.stdout:
        return

    stage("Pulling Ollama model", model)
    try:
        subprocess.run(["ollama", "pull", model], capture_output=True, timeout=300)
        stage("Model ready", model)
    except subprocess.TimeoutExpired:
        warning(f"Timed out pulling model {model}")


def _make_client(config: LLMConfig):
    """Create an OpenAI client configured from LLMConfig."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai is not installed. Install with: uv sync --extra llm")

    return OpenAI(
        base_url=config.api_base or None,
        api_key=config.api_key or None,
    )


# Substrings that identify "the server rejected response_format itself"
# rather than another 4xx (context length, bad model, etc.). Match is
# case-insensitive. Conservative: false negatives just mean the original
# error propagates with full traceback intact.
_FORMAT_REJECTION_SUBSTRINGS = (
    "response_format",
    "json_schema",
    "json_object",
    "structured output",
    "unavailable",
    "not supported",
    "unsupported",
)


def _is_format_unsupported_error(exc: Exception) -> bool:
    """True if a 4xx error message indicates the server rejected response_format.

    Narrowed by substring match so we don't mask context-length-exceeded,
    invalid-model, or content-policy errors that also raise BadRequestError.
    """
    try:
        from openai import BadRequestError
    except ImportError:  # pragma: no cover - openai always installed when calling
        return False
    if not isinstance(exc, BadRequestError):
        return False
    msg = str(exc).lower()
    return any(s in msg for s in _FORMAT_REJECTION_SUBSTRINGS)


def _apply_tier(params: dict, tier: str, json_schema: dict | None) -> None:
    """Set ``params['response_format']`` based on the resolved tier.

    If tier is ``"schema"`` but no schema is provided (e.g. retry/correction
    path), degrade to ``json_object`` so the call still requests JSON mode.
    """
    if tier == "schema":
        params["response_format"] = json_schema if json_schema else {"type": "json_object"}
    elif tier == "object":
        params["response_format"] = {"type": "json_object"}
    # "none" → leave unset


def _create_with_format_fallback(
    client: Any,
    params: dict,
    json_schema: dict | None,
    config: LLMConfig,
) -> Any:
    """Call chat.completions.create, discovering the best response_format once.

    Per-(api_base, model) cache. First call tries strict JSON schema, falls
    back to ``{"type": "json_object"}``, then no ``response_format``. The
    discovered tier is recorded and reused for every subsequent call so the
    fallback cost is paid at most once per provider+model.
    """
    cache_key = (config.api_base or "", config.model)
    cached = _RESPONSE_FORMAT_TIER.get(cache_key)

    if cached is not None:
        _apply_tier(params, cached, json_schema)
        return client.chat.completions.create(**params)

    # Discovery: try schema → object → none, recording whichever succeeds.
    candidates: list[str] = []
    if json_schema:
        candidates.append("schema")
    candidates.extend(["object", "none"])

    last_exc: Exception | None = None
    for tier in candidates:
        attempt = dict(params)
        _apply_tier(attempt, tier, json_schema)
        try:
            response = client.chat.completions.create(**attempt)
        except Exception as exc:
            if not _is_format_unsupported_error(exc):
                raise
            last_exc = exc
            debug(f"response_format tier {tier!r} rejected by {config.model}: {exc}")
            continue
        _RESPONSE_FORMAT_TIER[cache_key] = tier
        if tier != candidates[0]:
            warning(
                f"{config.model} does not support {candidates[0]!r} response_format; "
                f"using {tier!r} for the rest of this run."
            )
        return response

    raise (
        last_exc
        if last_exc is not None
        else RuntimeError("All response_format fallbacks exhausted")
    )


def complete(
    messages: list[dict[str, str]],
    config: LLMConfig,
    json_schema: dict | None = None,
    expected_count: int = 0,
    **kwargs: object,
) -> str:
    """Send a chat completion request via the OpenAI SDK.

    Works with any OpenAI-compatible endpoint (Ollama, Groq, DeepSeek,
    OpenRouter, OpenAI).  Set ``api_base`` and ``api_key`` in config.

    When *json_schema* is provided, uses strict JSON schema output.
    Falls back to ``{"type": "json_object"}`` if the model doesn't
    support it.

    When *expected_count* > 0, validates the response contains exactly
    that many items, and retries once on mismatch.
    """
    _ensure_ollama_model(config.api_base, config.model)

    client = _make_client(config)

    params: dict = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        **kwargs,
    }

    response = _create_with_format_fallback(client, params, json_schema, config)

    if not response.choices:
        raise RuntimeError("LLM returned empty response (no choices)")
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned empty content")

    # Strip thinking traces from reasoning models
    if "<think>" in content:
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
        if not content:
            raise RuntimeError("LLM response contained only thinking trace, no content")

    if expected_count > 0:
        content = _validate_item_count(content, expected_count, messages, config)

    return content


def _validate_item_count(
    content: str,
    expected_count: int,
    messages: list[dict[str, str]],
    config: LLMConfig,
) -> str:
    """Validate that the LLM response contains exactly expected_count items."""
    try:
        data = _try_parse_json(content)
    except (ValueError, json.JSONDecodeError):
        return content

    if not isinstance(data, dict):
        return content

    actual_count = _count_items(data)
    if actual_count == expected_count:
        return content

    warning(
        f"Item count mismatch ({actual_count} vs {expected_count} expected), "
        f"requesting correction..."
    )

    reask_msg = (
        f"You returned {actual_count} items but I need exactly "
        f"{expected_count}. Return the correct number of items "
        f"and nothing else. No explanations, no apologies."
    )
    retry_messages = messages + [
        {"role": "assistant", "content": content},
        {"role": "user", "content": reask_msg},
    ]

    try:
        client = _make_client(config)
        params = {
            "model": config.model,
            "messages": retry_messages,
            "temperature": 0.0,
            "max_tokens": config.max_tokens,
        }
        # Re-use the discovered response_format tier so the retry doesn't
        # silently send plain text when the original call used JSON mode.
        response2 = _create_with_format_fallback(client, params, json_schema=None, config=config)
        corrected = response2.choices[0].message.content
        if corrected:
            data2 = _try_parse_json(corrected)
            if isinstance(data2, dict) and _count_items(data2) == expected_count:
                return corrected
    except Exception as exc:
        debug(f"Item-count correction failed: {exc}")

    raise RuntimeError(f"LLM failed to return exactly {expected_count} items after correction.")


def _try_parse_json(text: str) -> dict | list | None:
    """Try to parse JSON, stripping markdown fences if present."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        t = "\n".join(lines)
    return json.loads(t)


def _count_items(data: dict) -> int:
    """Count items in a JSON response.

    Preferred: numbered-key format (``{"1": ..., "2": ...}``). Falls back
    to array-wrapped format for older/legacy responses.
    """
    numeric = 0
    for k in data:
        try:
            int(k)
            numeric += 1
        except (ValueError, TypeError):
            pass
    if numeric > 0:
        return numeric
    for key in ("translations", "refined", "translated", "results", "items"):
        if key in data and isinstance(data[key], list):
            return len(data[key])
    return 0
