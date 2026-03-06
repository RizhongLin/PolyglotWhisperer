"""Unified LLM client via LiteLLM with Ollama auto-pull support."""

from __future__ import annotations

from pgw.core.config import LLMConfig
from pgw.utils.console import console


def _extract_ollama_model(model_id: str) -> str | None:
    """Extract the Ollama model name from a LiteLLM model string.

    Returns None if the model is not an Ollama model.
    E.g. "ollama_chat/qwen3:8b" -> "qwen3:8b"
    """
    for prefix in ("ollama_chat/", "ollama/"):
        if model_id.startswith(prefix):
            return model_id[len(prefix) :]
    return None


def ensure_ollama_model(model_id: str) -> None:
    """Pull the Ollama model if not already available locally.

    No-op if the model is not an Ollama model or if ollama package
    is not installed.
    """
    model_name = _extract_ollama_model(model_id)
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

    # Ollama stores models as "name:tag" — check if the exact base matches with :latest
    if ":" not in model_name and f"{model_name}:latest" in available:
        return

    console.print(f"[bold]Pulling Ollama model:[/bold] {model_name}")
    try:
        ollama.pull(model_name)
        console.print(f"[green]Model ready:[/green] {model_name}")
    except Exception as e:
        console.print(f"[yellow]Failed to pull model {model_name}:[/yellow] {e}")


def complete(
    messages: list[dict[str, str]],
    config: LLMConfig,
    **kwargs: object,
) -> str:
    """Send a chat completion request via LiteLLM.

    Auto-pulls Ollama models if not available locally.

    Args:
        messages: Chat messages in OpenAI format.
        config: LLM configuration.
        **kwargs: Additional kwargs passed to litellm.completion.

    Returns:
        The assistant's response text.
    """
    try:
        import litellm
        from litellm import completion
    except ImportError:
        raise ImportError("LiteLLM is not installed. Install with: uv sync --extra llm")

    litellm.drop_params = True  # silently drop unsupported params per provider
    ensure_ollama_model(config.model)

    call_kwargs: dict = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "timeout": config.timeout,
        "num_retries": config.num_retries,
        **kwargs,
    }
    if _extract_ollama_model(config.model) is not None:
        if config.api_base:
            call_kwargs["api_base"] = config.api_base
        # Disable thinking mode for reasoning models (qwen3.5, etc.)
        # Thinking consumes the token budget and leaves content empty
        call_kwargs.setdefault("extra_body", {})["think"] = False
        # No max_tokens for local models — no cost concern, use full context window
    else:
        call_kwargs["max_tokens"] = config.max_tokens

    response = completion(**call_kwargs)

    if not response.choices:
        raise RuntimeError("LLM returned empty response (no choices)")
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned empty content")

    # Strip thinking traces from reasoning models (e.g. qwen3.5)
    # Some Ollama/LiteLLM versions ignore think=False and include <think> tags in content
    if "<think>" in content:
        import re

        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
        if not content:
            raise RuntimeError("LLM response contained only thinking trace, no content")

    return content


def unload_ollama_model(model_id: str) -> None:
    """Unload an Ollama model from GPU memory immediately.

    Sends a generate request with keep_alive=0, which tells Ollama
    to unload the model right away instead of keeping it for 5 minutes.
    No-op if the model is not an Ollama model.
    """
    model_name = _extract_ollama_model(model_id)
    if model_name is None:
        return

    try:
        import ollama
    except ImportError:
        return

    try:
        ollama.generate(model=model_name, keep_alive=0)
    except Exception:
        pass
