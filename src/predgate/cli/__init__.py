"""PredGate CLI package."""

from __future__ import annotations

from predgate.cli.baseline_cmd import baseline_app
from predgate.cli.common import app, console
from predgate.cli.init_cmd import init
from predgate.cli.lifecycle_cmd import compare, promote, reproduce
from predgate.cli.ops_cmd import bench, doctor, serve
from predgate.cli.run_cmd import run
from predgate.cli.users_cmd import users_app

app.add_typer(users_app, name="users")
app.add_typer(baseline_app, name="baseline")

__all__ = [
    "app",
    "baseline_app",
    "bench",
    "compare",
    "console",
    "doctor",
    "init",
    "promote",
    "reproduce",
    "run",
    "serve",
    "users_app",
]
