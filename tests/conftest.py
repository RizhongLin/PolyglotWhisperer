"""Shared test fixtures."""

from pathlib import Path

import pytest

from pgw.core.models import SubtitleSegment

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def make_segments(texts: list[str], duration: float = 1.0) -> list[SubtitleSegment]:
    """Create test segments with sequential timestamps."""
    return [
        SubtitleSegment(text=t, start=i * duration, end=(i + 1) * duration)
        for i, t in enumerate(texts)
    ]


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def sample_vtt(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample.vtt"
