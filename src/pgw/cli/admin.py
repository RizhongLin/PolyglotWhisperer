"""``pgw admin`` Typer subcommand: user CRUD from the terminal."""

from __future__ import annotations

import getpass

import typer
from rich.console import Console

admin_app = typer.Typer(
    name="admin",
    help="User management (create, reset password, list).",
    no_args_is_help=True,
)
console = Console()


def _require_db():
    """Initialise the schema lazily — same path as the web app uses."""
    import pgw.db.models  # noqa: F401  ensure ORM tables registered
    from pgw.db import Base, SessionLocal, get_engine

    Base.metadata.create_all(get_engine())
    return SessionLocal


def _prompt_password(prompt: str = "Password: ") -> str:
    pw = getpass.getpass(prompt)
    if len(pw) < 8:
        raise typer.BadParameter("password must be at least 8 characters")
    confirm = getpass.getpass("Confirm: ")
    if pw != confirm:
        raise typer.BadParameter("passwords do not match")
    return pw


@admin_app.command("create-user")
def create_user(
    email: str = typer.Argument(..., help="Email address (used for login)."),
    admin: bool = typer.Option(False, "--admin", help="Grant admin privileges."),
    password: str | None = typer.Option(
        None,
        "--password",
        help="Password. Prompted interactively if omitted (recommended).",
    ),
) -> None:
    """Create a new user. Admins can manage other users + workers."""
    from sqlalchemy import select

    from pgw.auth.bootstrap import create_user as _create
    from pgw.db.models.user import User

    pw = password or _prompt_password()
    SessionLocal = _require_db()
    with SessionLocal() as db:
        existing = db.scalar(select(User).where(User.email == email.lower()))
        if existing is not None:
            console.print(f"[red]user already exists:[/] {email}")
            raise typer.Exit(code=1)
        user = _create(db, email=email, password=pw, is_admin=admin)
        kind = "admin" if user.is_admin else "user"
        console.print(f"[green]created {kind}[/] {user.email} (id={user.id})")


@admin_app.command("reset-password")
def reset_password(
    email: str = typer.Argument(...),
    password: str | None = typer.Option(None, "--password"),
) -> None:
    """Set a new password for an existing user."""
    from sqlalchemy import select

    from pgw.auth.passwords import hash_password
    from pgw.db.models.user import User

    pw = password or _prompt_password("New password: ")
    SessionLocal = _require_db()
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.email == email.lower()))
        if user is None:
            console.print(f"[red]no such user:[/] {email}")
            raise typer.Exit(code=1)
        user.password_hash = hash_password(pw)
        db.commit()
        console.print(f"[green]password reset[/] for {user.email}")


@admin_app.command("list")
def list_users() -> None:
    """List all users."""
    from sqlalchemy import select

    from pgw.db.models.user import User

    SessionLocal = _require_db()
    with SessionLocal() as db:
        rows = list(db.scalars(select(User).order_by(User.created_at)))
        if not rows:
            console.print("(no users)")
            return
        for u in rows:
            mark = "[bold]admin[/]" if u.is_admin else "user"
            console.print(f"  {u.id}\t{u.email}\t{mark}\t{u.created_at:%Y-%m-%d}")
