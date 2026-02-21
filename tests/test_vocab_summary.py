"""Tests for vocabulary summary CEFR estimation."""

import pytest

from pgw.vocab.summary import zipf_to_cefr


@pytest.mark.parametrize(
    "zipf, expected",
    [
        (7.0, "A1"),
        (5.1, "A1"),
        (5.0, "A2"),  # boundary: > 5.0, not >=
        (4.1, "A2"),
        (4.0, "B1"),
        (3.5, "B1"),
        (3.0, "B2"),
        (2.5, "B2"),
        (2.0, "C1"),
        (1.5, "C1"),
        (1.0, "C2"),  # boundary: > 1.0, not >=
        (0.5, "C2"),
        (0.0, "C2"),
        (-1.0, "C2"),
        (-100.0, "C2"),
    ],
    ids=[
        "very-high→A1",
        "above-5→A1",
        "exactly-5→A2",
        "above-4→A2",
        "exactly-4→B1",
        "mid-B1",
        "exactly-3→B2",
        "mid-B2",
        "exactly-2→C1",
        "mid-C1",
        "exactly-1→C2",
        "below-1→C2",
        "zero→C2",
        "negative→C2",
        "very-negative→C2",
    ],
)
def test_zipf_to_cefr(zipf, expected):
    assert zipf_to_cefr(zipf) == expected
