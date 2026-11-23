"""PredGate fault-injection benchmark.

Automatically creates faulty candidate models and reports whether
PredGate and an accuracy-only baseline catch each fault type.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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


def _generate_data(seed: int = 42, n: int = 3000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate train/test data."""
    rng = np.random.RandomState(seed)

    def _make_df(n: int, rng: np.random.RandomState) -> pd.DataFrame:
        income = rng.lognormal(10.5, 0.5, n)
        age = rng.randint(18, 70, n)
        credit = rng.normal(680, 80, n).clip(300, 850).astype(int)
        debt = rng.beta(2, 5, n)
        emp = rng.exponential(5, n).clip(0, 40).astype(int)
        region = rng.choice(["North", "South", "East", "West"], n)
        gender = rng.choice(["male", "female"], n, p=[0.55, 0.45])
        age_bucket = np.where(age < 30, "young", np.where(age < 50, "middle", "senior"))

        logit = -2 + 0.3 * (debt - 0.3) - 0.001 * (income - 30000) / 10000 - 0.005 * (credit - 600)
        logit += np.where(region == "North", 0.1, 0)
        prob = 1 / (1 + np.exp(-logit))
        default = (rng.random(n) < prob).astype(int)

        return pd.DataFrame(
            {
                "income": income,
                "age": age,
                "credit_score": credit,
                "debt_ratio": debt,
                "employment_years": emp,
                "region": region,
                "age_bucket": age_bucket,
                "gender": gender,
                "default": default,
            }
        )

    train = _make_df(n, rng)
    test_rng = np.random.RandomState(seed + 1000)
    test = _make_df(n // 2, test_rng)
    return train, test


def _train_model(X: pd.DataFrame, y: pd.Series, seed: int) -> Pipeline:
    """Train a gradient boosting pipeline."""
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=seed),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def _train_latency_model(X: pd.DataFrame, y: pd.Series, seed: int) -> Pipeline:
    """Train a large random forest pipeline with high latency."""
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=300, max_depth=15, random_state=seed),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


class GroupThresholdModel:
    """Wrapper that shifts the binary classification threshold for one protected group while keeping predict_proba unchanged."""

    def __init__(
        self,
        base_model: Any,
        protected_col: str,
        target_group: str,
        group_threshold: float = 0.25,
        default_threshold: float = 0.50,
    ) -> None:
        self.base_model = base_model
        self.protected_col = protected_col
        self.target_group = target_group
        self.group_threshold = group_threshold
        self.default_threshold = default_threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.base_model.predict_proba(X)[:, 1]
        if self.protected_col in X.columns:
            thresh = np.where(
                X[self.protected_col].astype(str).values == self.target_group,
                self.group_threshold,
                self.default_threshold,
            )
        else:
            thresh = np.full(len(X), self.default_threshold)
        return (probs >= thresh).astype(int)


FAULT_DESCRIPTIONS = {
    "feature_leakage": "Add a column derived from the target",
    "label_shift": "Corrupt labels in one subgroup of the train set",
    "degraded_slice": "Remove most of one subgroup from training data",
    "schema_drift": "Change a column's scale or unit",
    "latency_regression": "Swap in a much larger model",
    "fairness_regression": "Bias sampling against one protected group",
    "fairness_threshold_shift": "Shift decision threshold for one protected group (probabilities unchanged)",
    "fairness_proxy_feature": "Drop protected attribute and bias training via a correlated proxy feature",
    "noop": "Retrain with a different seed (should PASS)",
}


def inject_fault(
    train_df: pd.DataFrame,
    fault_type: str,
    seed: int,
) -> tuple[pd.DataFrame, Any]:
    """Inject a specific fault and return modified data and optional model override.

    Returns:
        Tuple of (modified_train_df, model_override_or_None).
    """
    rng = np.random.RandomState(seed)
    df = train_df.copy()
    model_override = None

    if fault_type == "feature_leakage":
        df["leaked"] = df["default"] * 0.9 + rng.normal(0, 0.1, len(df))

    elif fault_type == "label_shift":
        mask = df["region"] == "North"
        flip_idx = df[mask].sample(frac=0.4, random_state=seed).index
        df.loc[flip_idx, "default"] = 1 - df.loc[flip_idx, "default"]

    elif fault_type == "degraded_slice":
        north = df[df["region"] == "North"]
        drop_idx = north.sample(frac=0.9, random_state=seed).index
        df = df.drop(drop_idx)

    elif fault_type == "schema_drift":
        df["income"] = df["income"] / 1000

    elif fault_type == "latency_regression":
        # Use a much larger model
        model_override = "large_model"

    elif fault_type == "fairness_regression":
        female_non_default = df[(df["gender"] == "female") & (df["default"] == 0)]
        flip_idx = female_non_default.sample(frac=0.3, random_state=seed).index
        df.loc[flip_idx, "default"] = 1

    elif fault_type == "fairness_threshold_shift":
        model_override = "group_threshold_shift"

    elif fault_type == "fairness_proxy_feature":
        is_female = (df["gender"] == "female").astype(float)
        df["proxy_score"] = is_female * 0.85 + rng.normal(0.0, 0.25, len(df))
        high_proxy_neg = df[(df["proxy_score"] > 0.6) & (df["default"] == 0)]
        flip_idx = high_proxy_neg.sample(frac=0.35, random_state=seed).index
        df.loc[flip_idx, "default"] = 1
        model_override = "proxy_blind_model"

    elif fault_type == "noop":
        pass  # No change, just different seed

    return df, model_override


def run_single_trial(
    fault_type: str,
    seed: int,
) -> dict[str, Any]:
    """Run a single trial of a fault injection.

    Returns:
        Dict with fault_type, seed, predgate_detected, baseline_detected.
    """
    # Generate clean data
    clean_train, test_df = _generate_data(seed=seed)

    # Train champion on clean data
    X_clean = clean_train.drop(columns=["default"])
    y_clean = clean_train["default"]
    champion = _train_model(X_clean, y_clean, seed=seed)

    # Inject fault
    faulty_train, model_override = inject_fault(clean_train, fault_type, seed)

    # Train candidate on faulty data
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
        # Drop explicit protected attribute so the model discriminates only through proxy_score
        X_proxy = X_faulty.drop(columns=["gender"])
        candidate = _train_model(X_proxy, y_faulty, seed=cand_seed)
    else:
        candidate = _train_model(X_faulty, y_faulty, seed=cand_seed)

    # Prepare test data
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

    # Predict
    champ_preds = champion.predict(X_test)
    cand_preds = candidate.predict(X_test)

    champ_acc = accuracy_score(y_test, champ_preds)
    cand_acc = accuracy_score(y_test, cand_preds)

    # ── Baseline: accuracy-only check ──
    baseline_detected = abs(champ_acc - cand_acc) > 0.01

    # ── PredGate: run multiple checks ──
    predgate_results: list[CheckResult] = []

    # Drift check
    drift_result = run_drift_check(
        faulty_train,
        clean_train,
        "default",
        DriftConfig(enabled=True, warn_at=0.1, fail_at=0.25),
    )
    predgate_results.append(drift_result)

    # Leakage check
    leakage_result = run_leakage_check(
        faulty_train,
        test_eval,
        "default",
        LeakageConfig(enabled=True),
        reference_df=clean_train,
    )
    predgate_results.append(leakage_result)

    # Label-shift check
    label_shift_result = run_label_shift_check(
        faulty_train,
        clean_train,
        "default",
        LabelShiftConfig(enabled=True, by=["region", "gender", "age_bucket"]),
    )
    predgate_results.append(label_shift_result)

    # Slice check
    slice_result = run_slices_check(
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
    )
    predgate_results.append(slice_result)

    # Fairness check
    fairness_result = run_fairness_check(
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
    )
    predgate_results.append(fairness_result)

    # Budgets check (interleaved batched median latency ratio gate with paired bootstrap CI)
    if fault_type in ("latency_regression", "noop"):
        budget_result = run_budgets_check(
            test_eval,
            "default",
            candidate,
            champion,
            BudgetConfig(enabled=True, max_latency_ratio=1.5, p95_latency_ms=None),
            seed=seed,
        )
        predgate_results.append(budget_result)

    # Compute PredGate verdict
    verdict = compute_verdict(predgate_results)
    predgate_detected = verdict.status in ("warn", "fail")

    return {
        "fault_type": fault_type,
        "seed": seed,
        "predgate_detected": predgate_detected,
        "predgate_status": verdict.status,
        "baseline_detected": baseline_detected,
        "champion_acc": round(champ_acc, 4),
        "candidate_acc": round(cand_acc, 4),
    }


def wilson_ci(k: int, n: int, z: float = 1.95996) -> tuple[float, float]:
    """Compute Wilson score 95% confidence interval for a binomial proportion."""
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
    """Return short git commit hash (or 'unknown' if unavailable)."""
    import subprocess

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


def run_benchmark(
    console: Console,
    n_seeds: int = 30,
    start_seed: int = 1000,
    command_str: str = "predgate bench",
) -> dict[str, Any]:
    """Run the full fault injection benchmark.

    Args:
        console: Rich console for output.
        n_seeds: Number of seeds per fault type.
        start_seed: First random seed in the contiguous range [start_seed, start_seed + n_seeds - 1].
        command_str: Command used to invoke the benchmark.

    Returns:
        Benchmark results dict.
    """
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

        mg_detected = 0
        bl_detected = 0

        for seed in seeds:
            try:
                result = run_single_trial(fault, seed)
                all_results.append(result)
                if result["predgate_detected"]:
                    mg_detected += 1
                if result["baseline_detected"]:
                    bl_detected += 1
            except Exception as e:
                console.print(f"[red]Error seed {seed}: {e}[/red]")

        elapsed = time.time() - start
        mg_rate = mg_detected / n_seeds if n_seeds > 0 else 0
        bl_rate = bl_detected / n_seeds if n_seeds > 0 else 0
        mg_ci = wilson_ci(mg_detected, n_seeds)
        bl_ci = wilson_ci(bl_detected, n_seeds)
        ci_overlap = (mg_ci[0] <= bl_ci[1]) and (bl_ci[0] <= mg_ci[1])

        summary[fault] = {
            "description": FAULT_DESCRIPTIONS[fault],
            "n_trials": n_seeds,
            "mg_detected": mg_detected,
            "bl_detected": bl_detected,
            "predgate_detection_rate": round(mg_rate, 3),
            "baseline_detection_rate": round(bl_rate, 3),
            "predgate_ci_95": [round(mg_ci[0], 3), round(mg_ci[1], 3)],
            "baseline_ci_95": [round(bl_ci[0], 3), round(bl_ci[1], 3)],
            "ci_95_overlap": ci_overlap,
            "predgate_wins": (mg_ci[0] > bl_ci[1]) if fault != "noop" else (mg_ci[1] < bl_ci[0]),
        }

        console.print(f"MG: {mg_rate:.0%}  BL: {bl_rate:.0%}  ({elapsed:.1f}s)")

    # Print summary table
    console.print()
    table = Table(title="PredGate Benchmark Results")
    table.add_column("Fault", style="bold")
    table.add_column("Description")
    table.add_column("PredGate", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Winner", justify="center")

    for fault, s in summary.items():
        mg_str = f"{s['mg_detected']}/{s['n_trials']} ({s['predgate_detection_rate']:.0%})"
        bl_str = f"{s['bl_detected']}/{s['n_trials']} ({s['baseline_detection_rate']:.0%})"
        winner = (
            "✅ MG"
            if s["predgate_wins"]
            else (
                "🔵 TIE"
                if s["predgate_detection_rate"] == s["baseline_detection_rate"]
                else "❌ BL"
            )
        )

        if fault == "noop":
            # For no-op, lower detection = better (false positives)
            mg_str = f"{s['mg_detected']}/{s['n_trials']} ({s['predgate_detection_rate']:.0%} FP)"
            bl_str = f"{s['bl_detected']}/{s['n_trials']} ({s['baseline_detection_rate']:.0%} FP)"
            winner = (
                "✅ MG" if s["predgate_detection_rate"] <= s["baseline_detection_rate"] else "❌ BL"
            )

        table.add_row(fault, s["description"], mg_str, bl_str, winner)

    console.print(table)

    # Save results
    results_data = {
        "command": command_str,
        "commit_hash": commit_hash,
        "n_seeds": n_seeds,
        "seed_range": f"{seeds[0]}..{seeds[-1]}" if seeds else "",
        "seeds": seeds,
        "summary": summary,
        "details": all_results,
    }

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    json_path = docs_dir / "benchmark_results.json"
    # Preserve existing public_fairness_benchmark section if already populated by scripts/bench_public.py
    if json_path.exists():
        try:
            existing_data = json.loads(json_path.read_text())
            if "public_fairness_benchmark" in existing_data:
                results_data["public_fairness_benchmark"] = existing_data[
                    "public_fairness_benchmark"
                ]
        except Exception:
            pass
    json_path.write_text(json.dumps(results_data, indent=2) + "\n")

    # Generate markdown strictly from the JSON file
    generate_benchmark_markdown_from_json(json_path, docs_dir / "benchmark.md")

    console.print(
        "\n[green]Results saved to docs/benchmark_results.json and docs/benchmark.md[/green]"
    )

    return results_data


def generate_benchmark_markdown_from_json(
    json_path: Path,
    md_path: Path,
) -> str:
    """Generate docs/benchmark.md strictly from a machine-readable benchmark_results.json file.

    No benchmark numbers are hardcoded in Python; every table row and metric is rendered
    directly from the JSON artifact.
    """
    data: dict[str, Any] = json.loads(json_path.read_text())
    summary: dict[str, dict[str, Any]] = data.get("summary", {})
    commit_hash = str(data.get("commit_hash", "unknown"))
    command_str = str(data.get("command", "predgate bench"))
    seeds: list[int] = list(data.get("seeds", []))
    if not seeds:
        first_entry: dict[str, Any] = next(iter(summary.values()), {})
        n_default = int(first_entry.get("n_trials", 30))
        seeds = [1000 + i for i in range(n_default)]

    seed_range_str = f"`{seeds[0]}..{seeds[-1]}` (N={len(seeds)}: `{', '.join(map(str, seeds))}`)"
    noop = summary.get("noop", {})
    noop_mg_k = noop.get("mg_detected", 0)
    noop_n = noop.get("n_trials", len(seeds))
    noop_mg_rate = noop.get("predgate_detection_rate", 0.0)
    noop_mg_ci = noop.get("predgate_ci_95", [0.0, 0.114])
    noop_bl_k = noop.get("bl_detected", 0)
    noop_bl_rate = noop.get("baseline_detection_rate", 0.0)
    noop_bl_ci = noop.get("baseline_ci_95", [0.0, 0.114])

    lines = [
        "# PredGate Benchmark Results",
        "",
        "Single source of truth for the PredGate fault-injection benchmark, generated strictly from [`docs/benchmark_results.json`](benchmark_results.json).",
        "",
        "## Reproducibility Metadata",
        "",
        f"- **Command**: `{command_str}`",
        f"- **Commit Hash**: `{commit_hash}`",
        f"- **Evaluated Seeds**: {seed_range_str}",
        f"- **No-op False-Positive Rate (PredGate)**: **{noop_mg_k}/{noop_n} ({noop_mg_rate:.1%} observed FP; Wilson 95% CI [{noop_mg_ci[0]:.1%}, {noop_mg_ci[1]:.1%}], i.e. an {noop_mg_ci[1]:.1%} 95% upper bound at N={noop_n})**",
        f"- **No-op False-Positive Rate (Baseline)**: **{noop_bl_k}/{noop_n} ({noop_bl_rate:.1%} observed FP; Wilson 95% CI [{noop_bl_ci[0]:.1%}, {noop_bl_ci[1]:.1%}], i.e. an {noop_bl_ci[1]:.1%} 95% upper bound at N={noop_n})**",
        "",
        f"## Detection Rates (Synthetic Credit Benchmark, Seeds `{seeds[0]}..{seeds[-1]}`)",
        "",
        "A winner is declared only when the 95% Wilson confidence intervals do not overlap; overlapping 95% CIs are reported as a statistical **Tie (overlapping 95% CIs)**.",
        "",
        "| Fault | Injected Defect | PredGate (Count, Rate, Wilson 95% CI) | Baseline (Count, Rate, Wilson 95% CI) | Winner |",
        "|-------|-----------------|----------------------------------------|---------------------------------------|--------|",
    ]

    for fault, s in summary.items():
        n = s["n_trials"]
        mg_k = s["mg_detected"]
        bl_k = s["bl_detected"]
        mg_ci = s["predgate_ci_95"]
        bl_ci = s["baseline_ci_95"]
        overlaps = (mg_ci[0] <= bl_ci[1]) and (bl_ci[0] <= mg_ci[1])

        if fault == "noop":
            mg_cell = f"{mg_k}/{n} ({s['predgate_detection_rate']:.1%} FP, [{mg_ci[0]:.1%}, {mg_ci[1]:.1%}])"
            bl_cell = f"{bl_k}/{n} ({s['baseline_detection_rate']:.1%} FP, [{bl_ci[0]:.1%}, {bl_ci[1]:.1%}])"
            if overlaps:
                winner = f"Tie ({mg_ci[1]:.1%} 95% CI upper bound)"
            else:
                winner = "PredGate" if mg_ci[1] < bl_ci[0] else "Baseline"
        else:
            mg_cell = (
                f"{mg_k}/{n} ({s['predgate_detection_rate']:.1%}, [{mg_ci[0]:.1%}, {mg_ci[1]:.1%}])"
            )
            bl_cell = (
                f"{bl_k}/{n} ({s['baseline_detection_rate']:.1%}, [{bl_ci[0]:.1%}, {bl_ci[1]:.1%}])"
            )
            if overlaps:
                winner = "Tie (overlapping 95% CIs)"
            else:
                winner = "PredGate" if mg_ci[0] > bl_ci[1] else "Baseline"

        lines.append(f"| `{fault}` | {s['description']} | {mg_cell} | {bl_cell} | **{winner}** |")

    lines.extend(
        [
            "",
            "## Baseline Definitions",
            "",
            "1. **Two-Sided Accuracy-Change Baseline (`|acc_champ - acc_cand| > 0.01`, shown in the table above)**: Flags any trial where overall test accuracy changes by more than 1 percentage point in either direction (drop or increase).",
            "2. **One-Sided Accuracy-Drop Baseline (`acc_champ - acc_cand > 0.01`)**: Flags only when overall test accuracy *drops* by more than 1 percentage point (misses `feature_leakage`, which artificially increases accuracy, while PredGate's `run_leakage_check` catches it directly).",
            "",
            "## No-op False-Positive Rate & Upper Bound",
            "",
            f"For the `noop` control (retraining the identical pipeline on clean data with candidate seed `s + 500`), PredGate produced **{noop_mg_k}/{noop_n} ({noop_mg_rate:.1%})** and the baseline produced **{noop_bl_k}/{noop_n} ({noop_bl_rate:.1%})** observed false positives across seeds `{seeds[0]}..{seeds[-1]}` (Wilson 95% CI `[{noop_mg_ci[0]:.1%}, {noop_mg_ci[1]:.1%}]`).",
        ]
    )

    pub = data.get("public_fairness_benchmark")
    if isinstance(pub, dict) and pub.get("rows"):
        pub_seeds: list[int] = list(pub.get("seeds", []))
        pub_range = (
            f"{pub_seeds[0]}..{pub_seeds[-1]}"
            if pub_seeds
            else str(pub.get("seed_range", "unknown"))
        )
        pub_cmd = str(pub.get("command", "python scripts/bench_public.py"))
        pub_commit = str(pub.get("commit_hash", commit_hash))
        frozen = pub.get("frozen_settings", {})
        frozen_str = ", ".join(f"`{k}={v}`" for k, v in frozen.items())

        lines.extend(
            [
                "",
                f"## Independent Fairness Validation on Public Datasets ({len(pub_seeds) or 30} Fresh Seeds `{pub_range}`, Frozen Defaults)",
                "",
                f"- **Command**: `{pub_cmd}`",
                f"- **Commit Hash**: `{pub_commit}`",
                f"- **Frozen `FairnessConfig` Settings**: {frozen_str}",
                "",
                "Evaluated using the **full `run_fairness_check` gate** (`status in ('warn', 'fail')`: paired bootstrap gap CI across demographic parity, equal opportunity, equalized odds, and per-group calibration + two-sided paired bootstrap per-group selection-rate shift at `WARN` level; static absolute DP opt-in via `enforce_absolute=True` relative to champion):",
                "",
                f"| Dataset (`{pub_range}`) | Fault | Full Fairness Check (Count, Rate, Wilson 95% CI; Fail/Warn) | Accuracy Baseline (`\\|Δacc\\| > 1%`) | Mean DP (`champ -> cand`) | Power / MDE Note (`80% power`) |",
                "|------------------------|-------|------------------------------------------------------------|--------------------------------------|---------------------------|--------------------------------|",
            ]
        )

        for row in pub["rows"]:
            ds = row["dataset"]
            flt = row["fault_label"]
            n_t = int(row["n_trials"])
            det = int(row["detected_total"])
            det_fail = int(row["detected_fail"])
            det_warn = int(row["detected_warn"])
            rate = float(row["detection_rate"])
            ci = row["ci_95"]
            bl_det = int(row["baseline_detected"])
            bl_rate = float(row["baseline_rate"])
            bl_ci = row["baseline_ci_95"]
            dp_str = f"`{row['mean_champ_dp']:.3f} -> {row['mean_cand_dp']:.3f}`"
            note_cnt = int(row["power_note_count"])
            mde_str = (
                f"`min_n={row['mean_min_n']:.0f}, MDE_DP={row['mean_mde_dp']:.3f}, "
                f"MDE_EO={row['mean_mde_eo']:.3f}` (note emitted {note_cnt}/{n_t})"
            )
            if row["fault"] == "noop":
                mg_cell = f"**{det}/{n_t} ({rate:.1%} FP, [{ci[0]:.1%}, {ci[1]:.1%}]; {det_fail}F/{det_warn}W)**"
                bl_cell = f"**{bl_det}/{n_t} ({bl_rate:.1%} FP, [{bl_ci[0]:.1%}, {bl_ci[1]:.1%}])**"
            else:
                mg_cell = (
                    f"{det}/{n_t} ({rate:.1%}, [{ci[0]:.1%}, {ci[1]:.1%}]; {det_fail}F/{det_warn}W)"
                )
                bl_cell = f"{bl_det}/{n_t} ({bl_rate:.1%}, [{bl_ci[0]:.1%}, {bl_ci[1]:.1%}])"
            lines.append(f"| **{ds}** | {flt} | {mg_cell} | {bl_cell} | {dp_str} | {mde_str} |")

    lines.extend(
        [
            "",
            "## Methodology",
            "",
            f"Each synthetic fault is injected across {len(seeds)} independent random seeds (`{seeds[0]}..{seeds[-1]}`). "
            "For each seed `s`, training data (`N=3000`) is generated with `RandomState(s)` and held-out test data (`N=1500`) with `RandomState(s + 1000)`. "
            "Public fairness datasets (UCI Adult `adult` v2 and German Credit `credit-g` v1) are fetched directly from OpenML via `scripts/bench_public.py`. "
            "Confidence intervals are calculated using the Wilson score interval at the 95% confidence level (`z = 1.95996`).",
        ]
    )

    content = "\n".join(lines) + "\n"
    md_path.write_text(content)
    return content


def _generate_benchmark_markdown(
    summary: dict[str, dict[str, Any]],
    path: Path,
    commit_hash: str = "unknown",
    seeds: list[int] | None = None,
    command_str: str = "predgate bench",
) -> None:
    """Compatibility wrapper that renders benchmark markdown via the JSON-backed renderer."""
    import tempfile

    if seeds is None:
        first_entry: dict[str, Any] = next(iter(summary.values()), {})
        n_default = int(first_entry.get("n_trials", 30))
        seeds = [1000 + i for i in range(n_default)]
    payload = {
        "command": command_str,
        "commit_hash": commit_hash,
        "n_seeds": len(seeds),
        "seed_range": f"{seeds[0]}..{seeds[-1]}" if seeds else "",
        "seeds": seeds,
        "summary": summary,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(payload, tf, indent=2)
        tmp_json = Path(tf.name)
    try:
        generate_benchmark_markdown_from_json(tmp_json, path)
    finally:
        tmp_json.unlink(missing_ok=True)
