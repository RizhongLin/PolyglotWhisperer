"""Unified LLM client via LiteLLM with Ollama auto-pull support."""

from __future__ import annotations

import asyncio

from rich.console import Console

from pgw.core.config import LLMConfig

console = Console()


def _extract_ollama_model(provider: str) -> str | None:
    """Extract the Ollama model name from a LiteLLM provider string.

    Returns None if the provider is not an Ollama model.
    E.g. "ollama_chat/qwen3:8b" -> "qwen3:8b"
    """
    for prefix in ("ollama_chat/", "ollama/"):
        if provider.startswith(prefix):
            return provider[len(prefix) :]
    return None


def ensure_ollama_model(provider: str) -> None:
    """Pull the Ollama model if not already available locally.

    No-op if the provider is not an Ollama model or if ollama package
    is not installed.
    """
    model_name = _extract_ollama_model(provider)
    if model_name is None:
        return

    try:
        import ollama
    except ImportError:
        return

    # Check if model is already available
    try:
        available = {m.model for m in ollama.list().models}
    except Exception:
        return

    if model_name in available:
        return

    # Also check without tag (ollama sometimes stores as "model:latest")
    base_name = model_name.split(":")[0]
    if any(m.startswith(base_name) for m in available):
        return

    console.print(f"[bold]Pulling Ollama model:[/bold] {model_name}")
    try:
        ollama.pull(model_name)
        console.print(f"[green]Model ready:[/green] {model_name}")
    except Exception as e:
        console.print(f"[yellow]Failed to pull model {model_name}:[/yellow] {e}")


async def acomplete(
    messages: list[dict[str, str]],
    config: LLMConfig,
    **kwargs: object,
) -> str:
    """Send an async chat completion request via LiteLLM.

    Auto-pulls Ollama models if not available locally.

    Args:
        messages: Chat messages in OpenAI format.
        config: LLM configuration.
        **kwargs: Additional kwargs passed to litellm.acompletion.

    Returns:
        The assistant's response text.
    """
    try:
        from litellm import acompletion
    except ImportError:
        raise ImportError("LiteLLM is not installed. Install with: uv sync --extra llm")

    ensure_ollama_model(config.provider)

    response = await acompletion(
        model=config.provider,
        messages=messages,
        api_base=config.api_base,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        **kwargs,
    )
    return response.choices[0].message.content


def complete(
    messages: list[dict[str, str]],
    config: LLMConfig,
    **kwargs: object,
) -> str:
    """Synchronous wrapper around acomplete."""
    return asyncio.run(acomplete(messages, config, **kwargs))
