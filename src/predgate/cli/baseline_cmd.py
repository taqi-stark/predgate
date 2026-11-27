"""`predgate baseline` subcommands: `export`, `import`, and `verify`."""

from __future__ import annotations

import typer

from predgate.cli.common import console

baseline_app = typer.Typer(
    name="baseline",
    help="Export or import persisted champion baseline (champion/ folder + metadata + metrics).",
    no_args_is_help=True,
)


@baseline_app.command("export")
def baseline_export(
    config: str = typer.Option("predgate.yaml", "--config", "-c", help="Config file path"),
    dir_path: str = typer.Option("champion", "--dir", "-d", help="Champion baseline directory"),
) -> None:
    """Export champion model, metadata, and metrics to a baseline directory."""
    from predgate.config import load_config
    from predgate.registry import export_champion_baseline

    try:
        cfg = load_config(config)
        res = export_champion_baseline(cfg, baseline_dir=dir_path)
        console.print(
            f"[green]✓ Exported champion baseline to {res['baseline_dir']}/[/green]\n"
            f"  Source: {res['metadata']['source']}  SHA-256: {res['metadata']['model_sha256'][:12]}\n"
            f"  Metrics: accuracy={res['metrics']['accuracy']:.4f}, f1={res['metrics']['f1']:.4f}"
        )
    except Exception as exc:
        console.print(f"[red]Failed to export champion baseline:[/red] {exc}")
        raise typer.Exit(1) from exc


@baseline_app.command("import")
def baseline_import(
    config: str = typer.Option("predgate.yaml", "--config", "-c", help="Config file path"),
    dir_path: str = typer.Option("champion", "--dir", "-d", help="Champion baseline directory"),
    pinned_sha256: str = typer.Option(
        "",
        "--pinned-sha256",
        help="Optional base-branch SHA-256 to verify champion/model.pkl against",
    ),
) -> None:
    """Import a persisted champion baseline into the MLflow registry."""
    from predgate.config import load_config
    from predgate.registry import import_champion_baseline

    try:
        cfg = load_config(config)
        if pinned_sha256:
            res = import_champion_baseline(cfg, baseline_dir=dir_path, pinned_sha256=pinned_sha256)
        else:
            res = import_champion_baseline(cfg, baseline_dir=dir_path)
        console.print(
            f"[green]✓ Imported champion baseline from {res['baseline_dir']}/[/green]\n"
            f"  MLflow run: {res['run_id']}  Promoted to @{cfg.champion.mlflow_alias}: {res['promoted']}\n"
            f"  Metrics: accuracy={res['metrics'].get('accuracy', 0.0):.4f}, f1={res['metrics'].get('f1', 0.0):.4f}"
        )
    except Exception as exc:
        console.print(f"[red]Failed to import champion baseline:[/red] {exc}")
        raise typer.Exit(1) from exc


@baseline_app.command("verify")
def baseline_verify(
    dir_path: str = typer.Option("champion", "--dir", "-d", help="Champion baseline directory"),
    dvc_path: str | None = typer.Option(
        None, "--dvc", help="Optional path to champion.dvc manifest"
    ),
    pinned_sha256: str = typer.Option(
        "",
        "--pinned-sha256",
        help="Pinned SHA-256 hash from the base branch to verify against",
    ),
    changed_file: list[str] | None = typer.Option(
        None,
        "--changed-file",
        help="File paths modified in the PR (repeatable); fails closed if champion/ is modified",
    ),
) -> None:
    """Verify champion baseline SHA-256 integrity (fails closed on mismatch or missing manifest)."""
    from predgate.registry import verify_champion_integrity

    try:
        res = verify_champion_integrity(
            baseline_dir=dir_path,
            dvc_path=dvc_path,
            pinned_sha256=pinned_sha256,
            changed_files=changed_file or None,
        )
        console.print(
            f"[green]✓ Champion baseline verified ({res['baseline_dir']}/)[/green]\n"
            f"  SHA-256: {res['sha256']}"
        )
    except Exception as exc:
        console.print(f"[red]Champion integrity verification failed:[/red] {exc}")
        raise typer.Exit(1) from exc
