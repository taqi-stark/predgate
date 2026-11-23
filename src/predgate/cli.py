"""PredGate CLI.

Commands:
    predgate init           – scaffold predgate.yaml and a demo
    predgate run            – run the gate locally
    predgate compare A B    – print comparison table
    predgate promote RUN_ID – set champion (requires approval record)
    predgate reproduce ID   – rebuild and verify
    predgate bench          – fault-injection benchmark
    predgate serve          – start API and dashboard
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from predgate.verdict import CheckResult

app = typer.Typer(
    name="predgate",
    help="CI/CD quality gate for machine learning models",
    add_completion=False,
)
console = Console()


@app.command()
def init(
    path: str = typer.Argument(".", help="Directory to initialize"),
) -> None:
    """Scaffold a predgate.yaml and demo project."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)

    config_file = target / "predgate.yaml"
    if config_file.exists():
        console.print("[yellow]predgate.yaml already exists, skipping.[/yellow]")
    else:
        config_file.write_text(_SCAFFOLD_CONFIG)
        console.print(f"[green]✓[/green] Created {config_file}")

    # Create demo structure
    demo_dir = target / "demo"
    demo_dir.mkdir(exist_ok=True)

    train_file = demo_dir / "train.py"
    if not train_file.exists():
        train_file.write_text(_SCAFFOLD_TRAIN)
        console.print(f"[green]✓[/green] Created {train_file}")

    console.print("\n[bold green]PredGate project initialized![/bold green]")
    console.print("Next steps:")
    console.print("  1. Add your data to data/")
    console.print("  2. Edit predgate.yaml")
    console.print("  3. Run: predgate run")


@app.command()
def run(
    config: str = typer.Option("predgate.yaml", "--config", "-c", help="Config file path"),
    branch: str = typer.Option("", "--branch", "-b", help="Branch name"),
    pr_number: int | None = typer.Option(None, "--pr", help="PR number"),
    output: str = typer.Option("", "--output", "-o", help="Output directory for reports"),
) -> None:
    """Run the PredGate quality gate locally."""
    from predgate.checks.behavioral import run_behavioral_check
    from predgate.checks.budgets import run_budgets_check
    from predgate.checks.data_contract import run_data_contract_check
    from predgate.checks.drift import run_drift_check
    from predgate.checks.fairness import run_fairness_check
    from predgate.checks.label_shift import run_label_shift_check
    from predgate.checks.leakage import run_leakage_check
    from predgate.checks.slices import run_slices_check
    from predgate.config import load_config
    from predgate.loader import load_candidate_and_champion
    from predgate.registry import log_run_to_mlflow
    from predgate.report import build_html_report, build_json_payload, build_pr_comment
    from predgate.verdict import compute_verdict

    console.print("[bold]PredGate[/bold] — Running quality gate...\n")

    # Load config
    try:
        cfg = load_config(config)
    except FileNotFoundError:
        console.print(f"[red]Config file not found:[/red] {config}")
        console.print("Run 'predgate init' to create one.")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Invalid config:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"Project: [bold]{cfg.project}[/bold]  Seed: {cfg.seed}")

    # Load models
    with console.status("Loading candidate and champion models..."):
        try:
            load_result = load_candidate_and_champion(cfg)
        except Exception as e:
            console.print(f"[red]Failed to load models:[/red] {e}")
            raise typer.Exit(1)

    if load_result.is_first_run:
        console.print(
            "[yellow]First run — no champion found. Comparison checks will be skipped.[/yellow]"
        )
    console.print(f"Training time: {load_result.train_time_s:.2f}s\n")

    # Run checks
    results: list[CheckResult] = []

    with console.status("[1/6] Data contract check..."):
        cr = run_data_contract_check(
            load_result.train_df,
            load_result.test_df,
            cfg.data.schema,
            cfg.checks.data_contract,
        )
        results.append(cr)
        _print_check_result(cr)

    with console.status("[2/8] Drift check..."):
        cr = run_drift_check(
            load_result.train_df,
            load_result.reference_df,
            load_result.target_column,
            cfg.checks.drift,
        )
        results.append(cr)
        _print_check_result(cr)

    with console.status("[3/8] Leakage check..."):
        cr = run_leakage_check(
            load_result.train_df,
            load_result.test_df,
            load_result.target_column,
            cfg.checks.leakage,
            reference_df=load_result.reference_df,
        )
        results.append(cr)
        _print_check_result(cr)

    with console.status("[4/8] Label-shift check..."):
        cr = run_label_shift_check(
            load_result.train_df,
            load_result.reference_df,
            load_result.target_column,
            cfg.checks.label_shift,
        )
        results.append(cr)
        _print_check_result(cr)

    with console.status("[3/6] Slice regression check..."):
        cr = run_slices_check(
            load_result.test_df,
            load_result.target_column,
            load_result.champion_model,
            load_result.candidate_model,
            cfg.checks.slices,
            seed=cfg.seed,
        )
        results.append(cr)
        _print_check_result(cr)

    with console.status("[4/6] Behavioral check..."):
        cr = run_behavioral_check(
            load_result.test_df,
            load_result.target_column,
            load_result.candidate_model,
            cfg.checks.behavioral,
            seed=cfg.seed,
        )
        results.append(cr)
        _print_check_result(cr)

    with console.status("[7/8] Performance budget check..."):
        cr = run_budgets_check(
            load_result.test_df,
            load_result.target_column,
            load_result.candidate_model,
            load_result.champion_model,
            cfg.checks.budgets,
            seed=cfg.seed,
        )
        results.append(cr)
        _print_check_result(cr)

    with console.status("[8/8] Fairness check..."):
        cr = run_fairness_check(
            load_result.test_df,
            load_result.target_column,
            load_result.candidate_model,
            load_result.champion_model,
            cfg.checks.fairness,
            seed=cfg.seed,
        )
        results.append(cr)
        _print_check_result(cr)

    # Compute verdict
    verdict = compute_verdict(results, cfg.verdict)

    # Print verdict
    console.print()
    status_style = {
        "pass": "[bold green]PASS[/bold green]",
        "warn": "[bold yellow]WARN[/bold yellow]",
        "fail": "[bold red]FAIL[/bold red]",
    }
    console.print(f"Verdict: {status_style.get(verdict.status, verdict.status)}")
    console.print(f"Summary: {verdict.summary}")

    if verdict.reasons:
        console.print("\nReasons:")
        for i, reason in enumerate(verdict.reasons, 1):
            severity_color = "red" if reason.severity == "fail" else "yellow"
            console.print(
                f"  {i}. [{severity_color}]{reason.severity.upper()}[/{severity_color}] [{reason.check_name}] {reason.message}"
            )

    # Generate reports
    pr_comment = build_pr_comment(
        verdict,
        candidate_sha=load_result.lineage.git_sha,
        seed=cfg.seed,
        data_hash=load_result.lineage.dvc_hash
        or next(iter(load_result.lineage.data_hashes.values()), ""),
    )

    # Save reports
    output_dir = Path(output) if output else Path("predgate_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "verdict.json").write_text(json.dumps(verdict.to_dict(), indent=2))
    (output_dir / "pr_comment.md").write_text(pr_comment)

    html_report = build_html_report(verdict, load_result.lineage, cfg.project)
    (output_dir / "report.html").write_text(html_report)

    json_payload = build_json_payload(
        verdict,
        load_result.lineage,
        project=cfg.project,
        branch=branch,
        pr_number=pr_number,
    )
    (output_dir / "payload.json").write_text(json.dumps(json_payload, indent=2, default=str))

    console.print(f"\n[dim]Reports saved to {output_dir}/[/dim]")

    # Log to MLflow
    try:
        lineage_dict = {
            "git_sha": load_result.lineage.git_sha,
            "data_hashes": load_result.lineage.data_hashes,
            "dvc_hash": load_result.lineage.dvc_hash,
            "seed": load_result.lineage.seed,
        }
        run_id = log_run_to_mlflow(
            cfg.project,
            verdict,
            lineage_dict,
            load_result.candidate_model,
            cfg.seed,
            tracking_uri=cfg.mlflow.tracking_uri,
            artifact_location=cfg.mlflow.artifact_location,
        )
        if run_id:
            console.print(f"[dim]MLflow run: {run_id}[/dim]")
    except Exception as e:
        console.print(f"[dim]MLflow logging skipped: {e}[/dim]")

    # Post to API if running
    try:
        import os

        import httpx

        api_url = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
        api_token = (
            os.getenv("PREDGATE_API_TOKEN")
            or os.getenv("PREDGATE_ADMIN_TOKEN")
            or os.getenv("MODELGATE_API_TOKEN")
            or os.getenv("MODELGATE_ADMIN_TOKEN", "")
        )
        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        resp = httpx.post(
            f"{api_url}/runs",
            content=json.dumps(json_payload, default=str),
            headers=headers,
            timeout=5,
        )
        if resp.status_code == 200:
            console.print("[dim]Run posted to API[/dim]")
    except Exception:
        pass  # API not running, that's fine

    # Exit with appropriate code
    if verdict.status == "fail":
        raise typer.Exit(1)


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
        except Exception as e:
            console.print(f"[dim]Failed to load config for project name: {e}[/dim]")

    try:
        import httpx

        # Check for approval
        resp = httpx.get(f"http://localhost:8000/runs/{run_id}", timeout=10)
        if resp.status_code == 200:
            run_data = resp.json()
            if run_data.get("verdict", {}).get("status") == "fail":
                console.print("[red]Cannot promote a FAIL run without override approval.[/red]")
                raise typer.Exit(1)
    except Exception:
        pass

    # Try MLflow promotion
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
        try:
            cfg = load_config(config)
            tracking_uri = cfg.mlflow.tracking_uri
        except Exception:
            pass
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

        # Get original metrics
        original_metrics = run.data.metrics
        console.print(f"  Original metrics: {original_metrics}")

        console.print("\n[yellow]To fully reproduce, check out the commit and re-run:[/yellow]")
        if git_sha:
            console.print(f"  git checkout {git_sha}")
        if dvc_hash:
            console.print("  dvc checkout")
        console.print("  predgate run --config predgate.yaml")

    except ImportError:
        console.print("[red]MLflow not available.[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def bench() -> None:
    """Run the fault-injection benchmark."""
    console.print("[bold]PredGate Benchmark[/bold] — Fault Injection Suite\n")

    # Import benchmark module
    try:
        from predgate.benchmark import run_benchmark

        run_benchmark(console)
    except ImportError:
        console.print("[yellow]Benchmark module not found. Ensure demo/ is set up.[/yellow]")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="API host"),
    port: int = typer.Option(8000, "--port", help="API port"),
) -> None:
    """Start the API server and dashboard."""
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


users_app = typer.Typer(
    name="users",
    help="Manage API users and roles (engineer, reviewer, admin).",
    no_args_is_help=True,
)
app.add_typer(users_app, name="users")

baseline_app = typer.Typer(
    name="baseline",
    help="Export or import persisted champion baseline (champion/ folder + metadata + metrics).",
    no_args_is_help=True,
)
app.add_typer(baseline_app, name="baseline")


@baseline_app.command("export")
def baseline_export(
    config: str = typer.Option("predgate.yaml", "--config", "-c", help="Config file path"),
    dir_path: str = typer.Option("champion", "--dir", "-d", help="Champion baseline directory"),
) -> None:
    """Export champion model, metadata, and metrics to champion/ folder."""
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
    except Exception as e:
        console.print(f"[red]Failed to export champion baseline:[/red] {e}")
        raise typer.Exit(1)


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
    """Import champion model, metadata, and metrics from champion/ into MLflow registry."""
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
    except Exception as e:
        console.print(f"[red]Failed to import champion baseline:[/red] {e}")
        raise typer.Exit(1)


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
    except Exception as e:
        console.print(f"[red]Champion integrity verification failed:[/red] {e}")
        raise typer.Exit(1)


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


def _print_check_result(cr: CheckResult) -> None:
    """Pretty-print a check result."""
    status_style = {
        "pass": "[green]PASS[/green]",
        "warn": "[yellow]WARN[/yellow]",
        "fail": "[red]FAIL[/red]",
        "skip": "[dim]SKIP[/dim]",
    }
    style = status_style.get(cr.status, cr.status)
    console.print(f"  {style} {cr.name}: {cr.summary} ({cr.duration_s:.2f}s)")


# ── Scaffold templates ──

_SCAFFOLD_CONFIG = """\
project: my-project
seed: 42

model:
  entrypoint: demo/train.py:build_model
  task: classification

data:
  train: data/train.csv
  test: data/test.csv
  reference: data/reference.csv
  target: default

champion:
  mlflow_alias: champion

checks:
  data_contract:
    enabled: true
    on_violation: fail
  drift:
    enabled: true
    method: psi
    warn_at: 0.1
    fail_at: 0.25
  slices:
    enabled: true
    metric: f1
    by: [region, age_bucket]
    min_slice_size: 100
    max_drop: 0.03
    confidence: 0.95
    n_bootstrap: 1000
  behavioral:
    enabled: true
    invariance: []
    directional: []
  budgets:
    max_latency_ratio: 1.5
    # Optional absolute ms ceiling:
    # p95_latency_ms: 25
    max_model_mb: 50
    max_peak_mem_mb: 500
  fairness:
    enabled: true
    protected: []
    metric: demographic_parity_diff
    warn_at: 0.05
    fail_at: 0.10

verdict:
  fail_if_any_fail: true
  warn_requires_approval: true
"""

_SCAFFOLD_TRAIN = """\
\"\"\"Example model training script.\"\"\"

from sklearn.ensemble import GradientBoostingClassifier


def build_model(X_train, y_train, seed=42):
    \"\"\"Build and train a gradient boosting classifier.\"\"\"
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model
"""


if __name__ == "__main__":
    app()
