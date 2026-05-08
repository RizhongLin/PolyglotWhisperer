"""Shared test fixtures."""

from pathlib import Path

import pytest

from pgw.core.models import SubtitleSegment

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolated_db(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
):
    """Each test gets its own SQLite file + a fresh engine.

    Without this, the first ``create_library_app`` call would lock in
    the default ``./pgw_workspace/pgw.db`` location for every subsequent
    test in the run, causing cross-test state pollution (and on
    multi-test-run CI, leaking real workspace dirs into the repo root).
    """
    from pgw.auth import deps as auth_deps
    from pgw.db.engine import reset_engine
    from pgw.server.worker_registry import GLOBAL_WORKERS

    db_dir = tmp_path_factory.mktemp("pgw_db")
    monkeypatch.setenv("PGW_DATABASE_URL", f"sqlite:///{db_dir / 'pgw.db'}")
    # Strip dev-shell admin env so app bootstrap doesn't auto-create a user
    # under tests that expect a pristine DB.
    monkeypatch.delenv("PGW_ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("PGW_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("PGW_DISABLE_BOOTSTRAP", raising=False)
    # Sticky flags / process-global registries — reset so tests don't
    # see one another's worker connections or bootstrap-ended state.
    auth_deps._bootstrap_ended.clear()
    with GLOBAL_WORKERS._lock:  # noqa: SLF001
        GLOBAL_WORKERS._workers.clear()  # noqa: SLF001
    reset_engine()
    yield
    reset_engine()


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
