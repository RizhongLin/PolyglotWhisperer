"""Tests for the pipeline event system."""

from pgw.core.events import EventCallback, PipelineEvent


def test_pipeline_event_creation():
    """PipelineEvent stores stage, progress, message, and optional data."""
    event = PipelineEvent(stage="transcribe", progress=0.5, message="Halfway done")
    assert event.stage == "transcribe"
    assert event.progress == 0.5
    assert event.message == "Halfway done"
    assert event.data is None


def test_pipeline_event_with_data():
    """PipelineEvent accepts optional data payload."""
    event = PipelineEvent(
        stage="save",
        progress=1.0,
        message="Done",
        data={"workspace": "/tmp/test"},
    )
    assert event.data == {"workspace": "/tmp/test"}


def test_event_callback_type():
    """EventCallback is a callable type alias accepting PipelineEvent."""
    collected: list[PipelineEvent] = []

    def handler(event: PipelineEvent) -> None:
        collected.append(event)

    # Type check — handler satisfies EventCallback
    cb: EventCallback = handler
    cb(PipelineEvent(stage="download", progress=0.0, message="Starting"))
    assert len(collected) == 1
    assert collected[0].stage == "download"
