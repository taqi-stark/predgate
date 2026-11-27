"""`predgate run` command."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import typer

from predgate.cli.common import app, console
from predgate.verdict import CheckResult


def _print_check_result(cr: CheckResult) -> None:
    """Pretty-print a check result row."""
    status_style = {
        "pass": "[green]PASS[/green]",
        "warn": "[yellow]WARN[/yellow]",
        "fail": "[red]FAIL[/red]",
        "skip": "[dim]SKIP[/dim]",
    }
    style = status_style.get(cr.status, cr.status)
    console.print(f"  {style} {cr.name}: {cr.summary} ({cr.duration_s:.2f}s)")


@app.command()
def run(
    config: str = typer.Option("predgate.yaml", "--config", "-c", help="Config file path"),
    preset: str | None = typer.Option(
        None, "--preset", help="Optional threshold preset override: strict, balanced, or quiet"
    ),
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
    from predgate.loader import detect_model_family_change, load_candidate_and_champion
    from predgate.registry import log_run_to_mlflow
    from predgate.report import build_html_report, build_json_payload, build_pr_comment
    from predgate.verdict import compute_verdict

    console.print("[bold]PredGate[/bold] — Running quality gate...\n")

    try:
        cfg = load_config(config, preset_override=preset)  # type: ignore[arg-type]
    except FileNotFoundError as exc:
        console.print(f"[red]Config file not found:[/red] {config}")
        console.print("Run 'predgate init' to create one.")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Invalid config:[/red] {exc}")
        raise typer.Exit(1) from exc

    preset_label = f"  Preset: {cfg.preset}" if cfg.preset else ""
    console.print(f"Project: [bold]{cfg.project}[/bold]  Seed: {cfg.seed}{preset_label}")

    with console.status("Loading candidate and champion models..."):
        try:
            load_result = load_candidate_and_champion(cfg)
        except Exception as exc:
            console.print(f"[red]Failed to load models:[/red] {exc}")
            raise typer.Exit(1) from exc

    if load_result.is_first_run:
        console.print(
            "[yellow]First run — no champion found. Comparison checks will be skipped.[/yellow]"
        )
    family_change = detect_model_family_change(
        load_result.champion_model, load_result.candidate_model
    )
    if family_change and family_change.get("changed"):
        console.print(f"[cyan]Note:[/cyan] {family_change['note']}")
    console.print(f"Training time: {load_result.train_time_s:.2f}s\n")

    results: list[CheckResult] = []

    with console.status("[1/8] Data contract check..."):
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

    with console.status("[5/8] Slice regression check..."):
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

    with console.status("[6/8] Behavioral check..."):
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

    verdict = compute_verdict(results, cfg.verdict, model_family_change=family_change)

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

    pr_comment = build_pr_comment(
        verdict,
        candidate_sha=load_result.lineage.git_sha,
        seed=cfg.seed,
        data_hash=load_result.lineage.dvc_hash
        or next(iter(load_result.lineage.data_hashes.values()), ""),
    )

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
    except Exception as exc:
        console.print(f"[dim]MLflow logging skipped: {exc}[/dim]")

    # Best-effort post to local API server; silent when running standalone in CLI/CI.
    with contextlib.suppress(Exception):
        import os

        import httpx

        api_url = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
        api_token = os.getenv("PREDGATE_API_TOKEN") or os.getenv("PREDGATE_ADMIN_TOKEN", "")
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

    if verdict.status == "fail":
        raise typer.Exit(1)
