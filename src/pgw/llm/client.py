"""Unified LLM client via OpenAI SDK with structured output support."""

from __future__ import annotations

import json
import re
import subprocess

from pgw.core.config import LLMConfig
from pgw.utils.console import stage, warning


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

    if json_schema:
        try:
            params["response_format"] = json_schema
        except Exception:
            params["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**params)

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
        response2 = client.chat.completions.create(
            model=config.model,
            messages=retry_messages,
            temperature=0.0,
            max_tokens=config.max_tokens,
        )
        corrected = response2.choices[0].message.content
        if corrected:
            data2 = _try_parse_json(corrected)
            if isinstance(data2, dict) and _count_items(data2) == expected_count:
                return corrected
    except Exception:
        pass

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
    """Count items in a JSON response, supporting both array and keyed formats."""
    for key in ("translations", "refined", "translated", "results", "items"):
        if key in data and isinstance(data[key], list):
            return len(data[key])
    numeric = 0
    for k in data:
        try:
            int(k)
            numeric += 1
        except (ValueError, TypeError):
            pass
    return numeric
