"""``pgw maintenance`` Typer subcommand: backfill, migrate, future GC tasks."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

maintenance_app = typer.Typer(
    name="maintenance",
    help="Out-of-band maintenance tasks (backfill, migrate, cleanup).",
    no_args_is_help=True,
)
console = Console()


@maintenance_app.command("backfill-flashcards")
def backfill_flashcards(
    owner_email: str = typer.Option(
        ...,
        "--owner",
        help="Email of the user whose vocab to materialise as flashcards.",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        help="Optional ISO code filter (e.g. 'fr'). Default: all languages.",
    ),
    require_translation: bool = typer.Option(
        True,
        "--require-translation/--allow-empty-back",
        help="Skip vocab entries whose first occurrence has no translation.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Optional cap on cards created in this run (spot-testing).",
    ),
) -> None:
    """Bulk-create flashcards from existing vocab entries.

    Idempotent — vocab entries already carded for the user are skipped
    via ``flashcards.vocab_entry_id``. Re-run safely after each new
    pipeline run to materialise newly-discovered words.
    """
    from sqlalchemy import select

    import pgw.db.models  # noqa: F401  ensure ORM tables registered
    from pgw.db import Base, SessionLocal, get_engine
    from pgw.db.models.user import User
    from pgw.maintenance.flashcards_backfill import run as run_backfill

    Base.metadata.create_all(get_engine())

    with SessionLocal() as db:
        owner = db.scalar(select(User).where(User.email == owner_email.lower()))
        if owner is None:
            console.print(
                f"[red]no such user:[/] {owner_email}. "
                f"Create one first with [bold]pgw admin create-user --admin {owner_email}[/]."
            )
            raise typer.Exit(code=1)

        report = run_backfill(
            db,
            owner=owner,
            language=language,
            require_translation=require_translation,
            limit=limit,
        )
        console.print(
            f"[green]flashcard backfill complete[/]\n"
            f"  cards created:           {report.cards_created}\n"
            f"  skipped (already done):  {report.skipped_already_exists}\n"
            f"  skipped (no translation): {report.skipped_no_translation}\n"
            f"  skipped (no occurrence): {report.skipped_no_occurrence}"
        )


@maintenance_app.command("migrate")
def migrate() -> None:
    """Run ``alembic upgrade head`` against the configured database.

    Used by the Docker entrypoint and by operators upgrading between
    pgw versions. Handles three cases:

    1. **Empty DB** — runs every migration in order.
    2. **Already-migrated DB** — no-op (alembic compares revisions).
    3. **Legacy ``create_all`` DB** (tables exist, no
       ``alembic_version`` table) — stamps the schema as ``head`` so
       future migrations run incrementally instead of trying to
       recreate existing tables.

    SQLite is allowed but the test/dev path uses
    ``Base.metadata.create_all`` directly; calling this against SQLite
    is mostly useful for dev parity checks.
    """
    from pathlib import Path as _Path

    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect

    from pgw.db import get_engine

    repo_root = _Path(__file__).resolve().parents[3]
    cfg_path = repo_root / "alembic.ini"
    if not cfg_path.is_file():
        console.print(f"[red]alembic.ini not found at[/] {cfg_path}")
        raise typer.Exit(code=1)

    cfg = Config(str(cfg_path))
    cfg.set_main_option("script_location", str(repo_root / "alembic"))

    engine = get_engine()
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    legacy = "users" in tables and "alembic_version" not in tables
    if legacy:
        console.print(
            "[yellow]legacy create_all schema detected — "
            "stamping as head before applying further migrations[/]"
        )
        command.stamp(cfg, "head")
        console.print("[green]stamped[/]")
        return

    command.upgrade(cfg, "head")
    console.print("[green]migrate complete[/]")


@maintenance_app.command("backfill")
def backfill(
    base_dir: Path = typer.Option(
        Path("./pgw_workspace"),
        "--base-dir",
        help="Workspace root to scan.",
    ),
    owner_email: str = typer.Option(
        ...,
        "--owner",
        help="Email of the user to attribute imported workspaces + vocab to.",
    ),
) -> None:
    """Import existing on-disk workspaces + vocab JSON into the DB.

    Idempotent — re-runnable. Honours ``(owner_id, slug, timestamp)``
    uniqueness on workspaces and the natural key on vocab entries.
    """
    from sqlalchemy import select

    import pgw.db.models  # noqa: F401
    from pgw.db import Base, SessionLocal, get_engine
    from pgw.db.models.user import User
    from pgw.maintenance.backfill import run as run_backfill

    Base.metadata.create_all(get_engine())

    if not base_dir.is_dir():
        console.print(f"[red]base_dir not found:[/] {base_dir}")
        raise typer.Exit(code=1)

    with SessionLocal() as db:
        owner = db.scalar(select(User).where(User.email == owner_email.lower()))
        if owner is None:
            console.print(
                f"[red]no such user:[/] {owner_email}. "
                f"Create one first with [bold]pgw admin create-user --admin {owner_email}[/]."
            )
            raise typer.Exit(code=1)

        report = run_backfill(db, owner=owner, base_dir=base_dir)
        console.print(
            f"[green]backfill complete[/]\n"
            f"  workspaces imported: {report.workspaces_imported}\n"
            f"  workspaces skipped:  {report.workspaces_skipped}\n"
            f"  vocab entries:       {report.vocab_entries_imported}\n"
            f"  vocab occurrences:   {report.vocab_occurrences_imported}"
        )
