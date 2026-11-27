"""Shared Typer application and Rich console for PredGate CLI commands."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="predgate",
    help="CI/CD quality gate for machine learning models",
    add_completion=False,
)
console = Console()
