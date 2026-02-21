"""Pipeline event system for streaming progress to external consumers.

Provides a lightweight callback mechanism that the pipeline emits events through.
Consumers (CLI progress bars, Gradio, FastAPI SSE) register a callback to receive
real-time updates without modifying pipeline logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class PipelineEvent:
    """A progress event emitted during pipeline execution.

    Attributes:
        stage: Pipeline stage name (download, audio, transcribe, translate, vocab, save).
        progress: Progress within this stage, 0.0 to 1.0.
        message: Human-readable status message.
        data: Optional payload (e.g. partial segments, file paths).
    """

    stage: str
    progress: float
    message: str
    data: dict | None = field(default=None)


EventCallback = Callable[[PipelineEvent], None]
