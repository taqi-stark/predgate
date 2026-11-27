"""`predgate users` subcommands: `add` and `list`."""

from __future__ import annotations

import typer

from predgate.cli.common import console

users_app = typer.Typer(
    name="users",
    help="Manage API users and roles (engineer, reviewer, admin).",
    no_args_is_help=True,
)


@users_app.command("add")
def users_add(
    name: str = typer.Argument(..., help="Unique username"),
    role: str = typer.Option("engineer", "--role", "-r", help="Role: engineer, reviewer, or admin"),
) -> None:
    """Provision an API user and print the bearer token once."""
    from sqlmodel import Session, SQLModel

    from predgate.api.main import engine
    from predgate.auth import VALID_ROLES, create_user, record_audit

    if role not in VALID_ROLES:
        console.print(f"[red]Invalid role '{role}'. Must be one of {list(VALID_ROLES)}[/red]")
        raise typer.Exit(1)

    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        user, token = create_user(session, name=name, role=role)
        record_audit(
            session,
            actor="cli",
            role="admin",
            action="user.create",
            target=f"user:{user.name}",
            outcome="allowed",
            detail={"role": user.role},
        )
        console.print(
            f"[green]✓ User '{user.name}' ({user.role}) provisioned.[/green]\n"
            f"Bearer token (save now; only SHA-256 hash is stored):\n{token}"
        )


@users_app.command("list")
def users_list() -> None:
    """List provisioned API users and roles."""
    from sqlmodel import Session, SQLModel, col, select

    from predgate.api.main import engine
    from predgate.api.models import User

    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        rows = session.exec(select(User).order_by(col(User.id).asc())).all()
        if not rows:
            console.print("[dim]No users provisioned.[/dim]")
            return
        for u in rows:
            console.print(f"  {u.id}: [bold]{u.name}[/bold] ({u.role}) — created {u.created_at}")
