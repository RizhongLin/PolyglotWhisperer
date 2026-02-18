"""Unified LLM client via LiteLLM with Ollama auto-pull support."""

from __future__ import annotations

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
        from litellm import completion
    except ImportError:
        raise ImportError("LiteLLM is not installed. Install with: uv sync --extra llm")

    ensure_ollama_model(config.provider)

    response = completion(
        model=config.provider,
        messages=messages,
        api_base=config.api_base,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        **kwargs,
    )
    return response.choices[0].message.content


def unload_ollama_model(provider: str) -> None:
    """Unload an Ollama model from GPU memory immediately.

    Sends a generate request with keep_alive=0, which tells Ollama
    to unload the model right away instead of keeping it for 5 minutes.
    No-op if the provider is not an Ollama model.
    """
    model_name = _extract_ollama_model(provider)
    if model_name is None:
        return

    try:
        import ollama
    except ImportError:
        return

    try:
        ollama.generate(model=model_name, keep_alive=0)
        console.print(f"[dim]Unloaded Ollama model:[/dim] {model_name}")
    except Exception:
        pass
