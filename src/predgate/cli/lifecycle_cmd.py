"""Run lifecycle commands: `compare`, `promote`, and `reproduce`."""

from __future__ import annotations

import contextlib
from pathlib import Path

import typer
from rich.table import Table

from predgate.cli.common import app, console


@app.command()
def compare(
    run_a: str = typer.Argument(..., help="First run ID"),
    run_b: str = typer.Argument(..., help="Second run ID"),
) -> None:
    """Compare two runs side-by-side."""
    console.print(f"Comparing runs [bold]{run_a}[/bold] vs [bold]{run_b}[/bold]")

    try:
        import httpx

        resp = httpx.get(f"http://localhost:8000/compare?a={run_a}&b={run_b}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            table = Table(title="Run Comparison")
            table.add_column("Metric")
            table.add_column(f"Run {run_a}")
            table.add_column(f"Run {run_b}")
            table.add_column("Delta")

            for m in data.get("metrics", []):
                delta = m.get("delta", 0)
                delta_str = f"{delta:+.4f}" if isinstance(delta, (int, float)) else str(delta)
                table.add_row(m["name"], str(m["value_a"]), str(m["value_b"]), delta_str)

            console.print(table)
        else:
            console.print(f"[red]API error:[/red] {resp.status_code}")
    except Exception:
        console.print(
            "[yellow]API not available.[/yellow] Use JSON files in predgate_output/ to compare."
        )


@app.command()
def promote(
    run_id: str = typer.Argument(..., help="Run ID to promote"),
    config: Path = typer.Option(
        Path("predgate.yaml"), "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Promote a run's model to champion."""
    from predgate.config import load_config
    from predgate.registry import promote_to_champion

    console.print(f"Promoting run [bold]{run_id}[/bold] to champion...")

    project = "predgate"
    tracking_uri: str | None = None
    if config.exists():
        try:
            cfg = load_config(config)
            project = cfg.project
            tracking_uri = cfg.mlflow.tracking_uri
        except Exception as exc:
            console.print(f"[dim]Failed to load config for project name: {exc}[/dim]")

    with contextlib.suppress(Exception):
        import httpx

        resp = httpx.get(f"http://localhost:8000/runs/{run_id}", timeout=10)
        if resp.status_code == 200:
            run_data = resp.json()
            if run_data.get("verdict", {}).get("status") == "fail":
                console.print("[red]Cannot promote a FAIL run without override approval.[/red]")
                raise typer.Exit(1)

    success = promote_to_champion(project, run_id, tracking_uri=tracking_uri)
    if success:
        console.print("[green]✓[/green] Model promoted to champion")
    else:
        console.print("[red]Failed to promote model.[/red]")
        raise typer.Exit(1)


@app.command()
def reproduce(
    run_id: str = typer.Argument(..., help="Run ID to reproduce"),
    config: Path = typer.Option(
        Path("predgate.yaml"), "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Rebuild a model from lineage and verify it matches."""
    import os
    import subprocess

    from predgate.config import load_config

    console.print(f"Reproducing run [bold]{run_id}[/bold]...")

    tracking_uri: str = "sqlite:///mlflow.db"
    if config.exists():
        with contextlib.suppress(Exception):
            cfg = load_config(config)
            tracking_uri = cfg.mlflow.tracking_uri
    if "MLFLOW_TRACKING_URI" in os.environ:
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri)
        run = client.get_run(run_id)

        seed = int(run.data.params.get("seed", 42))
        git_sha = run.data.params.get("git_sha", "")
        dvc_hash = run.data.params.get("dvc_hash") or run.data.tags.get("dvc_hash", "")

        console.print(f"  Seed: {seed}")
        console.print(f"  Git SHA: {git_sha}")
        if dvc_hash:
            console.print(f"  DVC Data Hash: {dvc_hash}")
            console.print("  [cyan]Checking out / pulling data from DVC by hash...[/cyan]")
            try:
                import shutil
                import sys

                dvc_bin = shutil.which("dvc") or str(Path(sys.executable).parent / "dvc")
                chk_res = subprocess.run([dvc_bin, "checkout"], capture_output=True, text=True)
                if chk_res.returncode == 0:
                    console.print(
                        f"  [green]✓[/green] Data checked out via DVC ({dvc_hash[:8]}...)"
                    )
                else:
                    pull_res = subprocess.run([dvc_bin, "pull"], capture_output=True, text=True)
                    if pull_res.returncode == 0:
                        console.print(f"  [green]✓[/green] Data pulled via DVC ({dvc_hash[:8]}...)")
                    else:
                        console.print(
                            f"  [dim]DVC notice: {chk_res.stderr.strip() or pull_res.stderr.strip()}[/dim]"
                        )
            except FileNotFoundError:
                console.print("  [yellow]DVC CLI not found.[/yellow]")

        original_metrics = run.data.metrics
        console.print(f"  Original metrics: {original_metrics}")

        console.print("\n[yellow]To fully reproduce, check out the commit and re-run:[/yellow]")
        if git_sha:
            console.print(f"  git checkout {git_sha}")
        if dvc_hash:
            console.print("  dvc checkout")
        console.print("  predgate run --config predgate.yaml")

    except ImportError as exc:
        console.print("[red]MLflow not available.[/red]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
