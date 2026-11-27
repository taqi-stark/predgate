"""Operational commands: `doctor`, `bench`, and `serve`."""

from __future__ import annotations

import json

import typer
from rich.table import Table

from predgate.cli.common import app, console


@app.command()
def doctor(
    config: str = typer.Option("predgate.yaml", "--config", "-c", help="Config file path"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON output"),
) -> None:
    """Check Python, git, DVC, MLflow, config, schema import, and data paths."""
    from predgate.doctor import run_doctor

    report = run_doctor(config_path=config)

    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2))
        if not report.ok:
            raise typer.Exit(1)
        return

    table = Table(title="PredGate Doctor")
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail")
    table.add_column("Suggested Fix")

    status_styles = {
        "ok": "[green]OK[/green]",
        "warn": "[yellow]WARN[/yellow]",
        "fail": "[red]FAIL[/red]",
    }
    for chk in report.checks:
        table.add_row(
            chk.name,
            status_styles.get(chk.status, chk.status),
            chk.detail,
            chk.fix or "—",
        )

    console.print(table)
    if report.ok:
        if report.n_warn:
            console.print(f"\n[yellow]Doctor passed with {report.n_warn} warning(s).[/yellow]")
        else:
            console.print("\n[bold green]All doctor checks passed.[/bold green]")
    else:
        console.print(
            f"\n[bold red]Doctor found {report.n_fail} error(s) and {report.n_warn} warning(s).[/bold red]"
        )
        raise typer.Exit(1)


@app.command()
def bench() -> None:
    """Run the fault-injection benchmark."""
    console.print("[bold]PredGate Benchmark[/bold] — Fault Injection Suite\n")

    try:
        from predgate.bench import run_benchmark

        run_benchmark(console)
    except ImportError as exc:
        console.print("[yellow]Benchmark module not found.[/yellow]")
        raise typer.Exit(1) from exc


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="API host"),
    port: int = typer.Option(8000, "--port", help="API port"),
) -> None:
    """Start the API server."""
    import uvicorn

    console.print(f"[bold]PredGate[/bold] — Starting server on {host}:{port}")
    console.print(f"  API: http://localhost:{port}")
    console.print(f"  Docs: http://localhost:{port}/docs")

    uvicorn.run(
        "predgate.api.main:app",
        host=host,
        port=port,
        reload=True,
    )
