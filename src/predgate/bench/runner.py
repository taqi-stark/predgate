"""Fault-injection benchmark runner and Wilson CI helper."""

from __future__ import annotations

import contextlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score

from predgate.bench.faults import (
    FAULT_DESCRIPTIONS,
    GroupThresholdModel,
    _generate_data,
    _train_latency_model,
    _train_model,
    fixed_dp_limit_baseline,
    inject_fault,
    per_group_accuracy_drop_baseline,
)
from predgate.bench.render import generate_benchmark_markdown_from_json
from predgate.checks.budgets import run_budgets_check
from predgate.checks.drift import run_drift_check
from predgate.checks.fairness import run_fairness_check
from predgate.checks.label_shift import run_label_shift_check
from predgate.checks.leakage import run_leakage_check
from predgate.checks.slices import run_slices_check
from predgate.config import (
    BudgetConfig,
    DriftConfig,
    FairnessConfig,
    LabelShiftConfig,
    LeakageConfig,
    SliceConfig,
)
from predgate.verdict import CheckResult, compute_verdict


def wilson_ci(k: int, n: int, z: float = 1.95996) -> tuple[float, float]:
    """Compute the Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + (z**2 / n)
    center = p + (z**2 / (2 * n))
    margin = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    lower = max(0.0, (center - margin) / denom)
    upper = min(1.0, (center + margin) / denom)
    return float(lower), float(upper)


def _get_git_commit() -> str:
    """Return short git commit hash, or `'unknown'` if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def run_single_trial(fault_type: str, seed: int) -> dict[str, Any]:
    """Run one synthetic fault-injection trial for PredGate and baseline comparators."""
    clean_train, test_df = _generate_data(seed=seed)

    X_clean = clean_train.drop(columns=["default"])
    y_clean = clean_train["default"]
    champion = _train_model(X_clean, y_clean, seed=seed)

    faulty_train, model_override = inject_fault(clean_train, fault_type, seed)

    X_faulty = faulty_train.drop(columns=["default"])
    y_faulty = faulty_train["default"]

    cand_seed = seed + 500 if fault_type == "noop" else seed
    if model_override == "large_model":
        candidate: Any = _train_latency_model(X_faulty, y_faulty, seed=cand_seed)
    elif model_override == "group_threshold_shift":
        base_cand = _train_model(X_faulty, y_faulty, seed=cand_seed)
        candidate = GroupThresholdModel(
            base_cand,
            protected_col="gender",
            target_group="female",
            group_threshold=0.20,
            default_threshold=0.50,
        )
    elif model_override == "proxy_blind_model":
        # Drop explicit protected attribute so the model discriminates only via proxy_score.
        X_proxy = X_faulty.drop(columns=["gender"])
        candidate = _train_model(X_proxy, y_faulty, seed=cand_seed)
    else:
        candidate = _train_model(X_faulty, y_faulty, seed=cand_seed)

    test_eval = test_df.copy()
    if fault_type == "feature_leakage":
        test_rng = np.random.RandomState(seed + 1000)
        test_eval["leaked"] = test_eval["default"] * 0.9 + test_rng.normal(0, 0.1, len(test_eval))
    elif fault_type == "fairness_proxy_feature":
        test_rng = np.random.RandomState(seed + 1000)
        is_female_t = (test_eval["gender"] == "female").astype(float)
        test_eval["proxy_score"] = is_female_t * 0.85 + test_rng.normal(0.0, 0.25, len(test_eval))

    X_test = test_eval.drop(columns=["default"])
    y_test = test_eval["default"]

    champ_preds = champion.predict(X_test)
    cand_preds = candidate.predict(X_test)

    champ_acc = accuracy_score(y_test, champ_preds)
    cand_acc = accuracy_score(y_test, cand_preds)

    baseline_detected = abs(champ_acc - cand_acc) > 0.01
    per_group_acc_detected = per_group_accuracy_drop_baseline(
        y_test.values,
        champ_preds,
        cand_preds,
        test_eval,
        group_cols=["region", "gender"],
        max_drop=0.02,
        min_group_size=30,
    )
    fixed_dp_detected = fixed_dp_limit_baseline(
        cand_preds, test_eval["gender"].values, dp_limit=0.10
    )

    predgate_results: list[CheckResult] = [
        run_drift_check(
            faulty_train,
            clean_train,
            "default",
            DriftConfig(enabled=True, warn_at=0.1, fail_at=0.25),
        ),
        run_leakage_check(
            faulty_train,
            test_eval,
            "default",
            LeakageConfig(enabled=True),
            reference_df=clean_train,
        ),
        run_label_shift_check(
            faulty_train,
            clean_train,
            "default",
            LabelShiftConfig(enabled=True, by=["region", "gender", "age_bucket"]),
        ),
        run_slices_check(
            test_eval,
            "default",
            champion,
            candidate,
            SliceConfig(
                enabled=True,
                metric="f1",
                by=["region", "gender"],
                min_slice_size=20,
                max_drop=0.03,
                confidence=0.95,
                n_bootstrap=200,
            ),
            seed=seed,
        ),
        run_fairness_check(
            test_eval,
            "default",
            candidate,
            champion,
            FairnessConfig(
                enabled=True,
                protected=["gender"],
                metric="demographic_parity_diff",
                warn_at=0.05,
                fail_at=0.10,
            ),
            seed=seed,
        ),
    ]

    if fault_type in ("latency_regression", "noop"):
        predgate_results.append(
            run_budgets_check(
                test_eval,
                "default",
                candidate,
                champion,
                BudgetConfig(enabled=True, max_latency_ratio=1.5, p95_latency_ms=None),
                seed=seed,
            )
        )

    verdict = compute_verdict(predgate_results)
    predgate_fail = verdict.status == "fail"
    predgate_detected = verdict.status in ("warn", "fail")

    return {
        "fault_type": fault_type,
        "seed": seed,
        "predgate_fail": predgate_fail,
        "predgate_detected": predgate_detected,
        "predgate_status": verdict.status,
        "check_statuses": {cr.name: cr.status for cr in predgate_results},
        "baseline_detected": baseline_detected,
        "per_group_acc_detected": per_group_acc_detected,
        "fixed_dp_detected": fixed_dp_detected,
        "champion_acc": round(champ_acc, 4),
        "candidate_acc": round(cand_acc, 4),
    }


def run_benchmark(
    console: Console,
    n_seeds: int = 30,
    start_seed: int = 1000,
    command_str: str = "predgate bench",
) -> dict[str, Any]:
    """Run the synthetic 7-fault benchmark across contiguous seeds and write JSON/Markdown outputs."""
    fault_types = [
        "feature_leakage",
        "label_shift",
        "degraded_slice",
        "schema_drift",
        "latency_regression",
        "fairness_regression",
        "noop",
    ]

    seeds = [start_seed + i for i in range(n_seeds)]
    commit_hash = _get_git_commit()
    all_results: list[dict] = []
    summary: dict[str, dict] = {}

    for fault in fault_types:
        console.print(f"  Running [bold]{fault}[/bold] ({n_seeds} seeds)...", end=" ")
        start = time.time()

        pg_fail = 0
        pg_detected = 0
        bl_detected = 0
        pg_acc_detected = 0
        fixed_dp_detected = 0

        for seed in seeds:
            try:
                result = run_single_trial(fault, seed)
                all_results.append(result)
                if result.get("predgate_fail", result.get("predgate_status") == "fail"):
                    pg_fail += 1
                if result["predgate_detected"]:
                    pg_detected += 1
                if result["baseline_detected"]:
                    bl_detected += 1
                if result.get("per_group_acc_detected"):
                    pg_acc_detected += 1
                if result.get("fixed_dp_detected"):
                    fixed_dp_detected += 1
            except Exception as exc:
                console.print(f"[red]Error seed {seed}: {exc}[/red]")

        elapsed = time.time() - start
        pg_fail_rate = pg_fail / n_seeds if n_seeds > 0 else 0
        pg_rate = pg_detected / n_seeds if n_seeds > 0 else 0
        bl_rate = bl_detected / n_seeds if n_seeds > 0 else 0
        pg_acc_rate = pg_acc_detected / n_seeds if n_seeds > 0 else 0
        fixed_dp_rate = fixed_dp_detected / n_seeds if n_seeds > 0 else 0
        pg_fail_ci = wilson_ci(pg_fail, n_seeds)
        pg_ci = wilson_ci(pg_detected, n_seeds)
        bl_ci = wilson_ci(bl_detected, n_seeds)
        pg_acc_ci = wilson_ci(pg_acc_detected, n_seeds)
        fixed_dp_ci = wilson_ci(fixed_dp_detected, n_seeds)
        ci_overlap = (pg_ci[0] <= bl_ci[1]) and (bl_ci[0] <= pg_ci[1])

        summary[fault] = {
            "description": FAULT_DESCRIPTIONS[fault],
            "n_trials": n_seeds,
            "pg_fail": pg_fail,
            "pg_warn": pg_detected - pg_fail,
            "pg_detected": pg_detected,
            "bl_detected": bl_detected,
            "per_group_acc_detected": pg_acc_detected,
            "fixed_dp_detected": fixed_dp_detected,
            "predgate_fail_rate": round(pg_fail_rate, 3),
            "predgate_fail_ci_95": [round(pg_fail_ci[0], 3), round(pg_fail_ci[1], 3)],
            "predgate_detection_rate": round(pg_rate, 3),
            "baseline_detection_rate": round(bl_rate, 3),
            "per_group_acc_rate": round(pg_acc_rate, 3),
            "fixed_dp_rate": round(fixed_dp_rate, 3),
            "predgate_ci_95": [round(pg_ci[0], 3), round(pg_ci[1], 3)],
            "baseline_ci_95": [round(bl_ci[0], 3), round(bl_ci[1], 3)],
            "per_group_acc_ci_95": [round(pg_acc_ci[0], 3), round(pg_acc_ci[1], 3)],
            "fixed_dp_ci_95": [round(fixed_dp_ci[0], 3), round(fixed_dp_ci[1], 3)],
            "ci_95_overlap": ci_overlap,
            "predgate_wins": (pg_ci[0] > bl_ci[1]) if fault != "noop" else (pg_ci[1] < bl_ci[0]),
        }

        console.print(
            f"PG: {pg_fail_rate:.0%} FAIL / {pg_rate:.0%} flagged  BL: {bl_rate:.0%}  ({elapsed:.1f}s)"
        )

    console.print()
    table = Table(title="PredGate Benchmark Results")
    table.add_column("Fault", style="bold")
    table.add_column("Description")
    table.add_column("PredGate (FAIL / Flagged)", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Winner", justify="center")

    for fault, s in summary.items():
        pg_str = (
            f"{s['pg_fail']}/{s['n_trials']} ({s['predgate_fail_rate']:.0%}) / "
            f"{s['pg_detected']}/{s['n_trials']} ({s['predgate_detection_rate']:.0%})"
        )
        bl_str = f"{s['bl_detected']}/{s['n_trials']} ({s['baseline_detection_rate']:.0%})"
        winner = (
            "✅ PG"
            if s["predgate_wins"]
            else (
                "🔵 TIE"
                if s["predgate_detection_rate"] == s["baseline_detection_rate"]
                else "❌ BL"
            )
        )

        if fault == "noop":
            pg_str = f"{s['pg_fail']}/{s['n_trials']}F / {s['pg_detected']}/{s['n_trials']} ({s['predgate_detection_rate']:.0%} FP)"
            bl_str = f"{s['bl_detected']}/{s['n_trials']} ({s['baseline_detection_rate']:.0%} FP)"
            winner = (
                "✅ PG" if s["predgate_detection_rate"] <= s["baseline_detection_rate"] else "❌ BL"
            )

        table.add_row(fault, s["description"], pg_str, bl_str, winner)

    console.print(table)

    # Keep aggregate summary in git; write large per-seed records to release_artifacts/.
    per_seed_artifact = f"release_artifacts/benchmark_per_seed_{commit_hash}.json"
    results_data: dict[str, Any] = {
        "command": command_str,
        "commit_hash": commit_hash,
        "n_seeds": n_seeds,
        "seed_range": f"{seeds[0]}..{seeds[-1]}" if seeds else "",
        "seeds": seeds,
        "per_seed_artifact": per_seed_artifact,
        "summary": summary,
    }

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    json_path = docs_dir / "benchmark_results.json"
    if json_path.exists():
        with contextlib.suppress(Exception):
            existing_data = json.loads(json_path.read_text())
            for preserve_key in (
                "public_fairness_benchmark",
                "benign_change_control",
                "held_out_validation",
                "comparator_config",
            ):
                if preserve_key in existing_data:
                    results_data[preserve_key] = existing_data[preserve_key]
    json_path.write_text(json.dumps(results_data, indent=2) + "\n")

    artifact_path = Path(per_seed_artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "command": command_str,
                "commit_hash": commit_hash,
                "synthetic_details": all_results,
            },
            indent=2,
        )
        + "\n"
    )

    generate_benchmark_markdown_from_json(json_path, docs_dir / "benchmark.md")

    console.print(
        f"\n[green]Aggregate saved to docs/benchmark_results.json, per-seed records to {per_seed_artifact}, and report to docs/benchmark.md[/green]"
    )

    return results_data
