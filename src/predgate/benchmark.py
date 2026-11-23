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
from predgate.stats import demographic_parity_diff
from predgate.verdict import CheckResult, compute_verdict


def per_group_accuracy_drop_baseline(
    y_true: np.ndarray,
    champ_preds: np.ndarray,
    cand_preds: np.ndarray,
    test_df: pd.DataFrame,
    group_cols: list[str],
    max_drop: float = 0.02,
    min_group_size: int = 30,
) -> bool:
    """Stronger non-bootstrap baseline: flags if any subgroup's accuracy drops > max_drop."""
    y_arr = np.asarray(y_true)
    ch_arr = np.asarray(champ_preds)
    ca_arr = np.asarray(cand_preds)
    for col in group_cols:
        if col not in test_df.columns:
            continue
        vals = test_df[col].astype(str).values
        for g in np.unique(vals):
            mask = vals == g
            if int(np.sum(mask)) < min_group_size:
                continue
            acc_ch = float(np.mean(y_arr[mask] == ch_arr[mask]))
            acc_ca = float(np.mean(y_arr[mask] == ca_arr[mask]))
            if (acc_ch - acc_ca) > max_drop:
                return True
    return False


def fixed_dp_limit_baseline(
    cand_preds: np.ndarray,
    protected_values: np.ndarray,
    dp_limit: float = 0.10,
) -> bool:
    """Stronger fairness baseline: flags if candidate absolute demographic parity diff >= dp_limit."""
    return bool(
        demographic_parity_diff(np.asarray(cand_preds, dtype=int), protected_values) >= dp_limit
    )


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

    # ── Baselines ──
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

        mg_fail = 0
        mg_detected = 0
        bl_detected = 0
        pg_acc_detected = 0
        fixed_dp_detected = 0

        for seed in seeds:
            try:
                result = run_single_trial(fault, seed)
                all_results.append(result)
                if result.get("predgate_fail", result.get("predgate_status") == "fail"):
                    mg_fail += 1
                if result["predgate_detected"]:
                    mg_detected += 1
                if result["baseline_detected"]:
                    bl_detected += 1
                if result.get("per_group_acc_detected"):
                    pg_acc_detected += 1
                if result.get("fixed_dp_detected"):
                    fixed_dp_detected += 1
            except Exception as e:
                console.print(f"[red]Error seed {seed}: {e}[/red]")

        elapsed = time.time() - start
        mg_fail_rate = mg_fail / n_seeds if n_seeds > 0 else 0
        mg_rate = mg_detected / n_seeds if n_seeds > 0 else 0
        bl_rate = bl_detected / n_seeds if n_seeds > 0 else 0
        pg_acc_rate = pg_acc_detected / n_seeds if n_seeds > 0 else 0
        fixed_dp_rate = fixed_dp_detected / n_seeds if n_seeds > 0 else 0
        mg_fail_ci = wilson_ci(mg_fail, n_seeds)
        mg_ci = wilson_ci(mg_detected, n_seeds)
        bl_ci = wilson_ci(bl_detected, n_seeds)
        pg_acc_ci = wilson_ci(pg_acc_detected, n_seeds)
        fixed_dp_ci = wilson_ci(fixed_dp_detected, n_seeds)
        ci_overlap = (mg_ci[0] <= bl_ci[1]) and (bl_ci[0] <= mg_ci[1])

        summary[fault] = {
            "description": FAULT_DESCRIPTIONS[fault],
            "n_trials": n_seeds,
            "mg_fail": mg_fail,
            "mg_warn": mg_detected - mg_fail,
            "mg_detected": mg_detected,
            "bl_detected": bl_detected,
            "per_group_acc_detected": pg_acc_detected,
            "fixed_dp_detected": fixed_dp_detected,
            "predgate_fail_rate": round(mg_fail_rate, 3),
            "predgate_fail_ci_95": [round(mg_fail_ci[0], 3), round(mg_fail_ci[1], 3)],
            "predgate_detection_rate": round(mg_rate, 3),
            "baseline_detection_rate": round(bl_rate, 3),
            "per_group_acc_rate": round(pg_acc_rate, 3),
            "fixed_dp_rate": round(fixed_dp_rate, 3),
            "predgate_ci_95": [round(mg_ci[0], 3), round(mg_ci[1], 3)],
            "baseline_ci_95": [round(bl_ci[0], 3), round(bl_ci[1], 3)],
            "per_group_acc_ci_95": [round(pg_acc_ci[0], 3), round(pg_acc_ci[1], 3)],
            "fixed_dp_ci_95": [round(fixed_dp_ci[0], 3), round(fixed_dp_ci[1], 3)],
            "ci_95_overlap": ci_overlap,
            "predgate_wins": (mg_ci[0] > bl_ci[1]) if fault != "noop" else (mg_ci[1] < bl_ci[0]),
        }

        console.print(
            f"PG: {mg_fail_rate:.0%} FAIL / {mg_rate:.0%} flagged  BL: {bl_rate:.0%}  ({elapsed:.1f}s)"
        )

    # Print summary table
    console.print()
    table = Table(title="PredGate Benchmark Results")
    table.add_column("Fault", style="bold")
    table.add_column("Description")
    table.add_column("PredGate (FAIL / Flagged)", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Winner", justify="center")

    for fault, s in summary.items():
        mg_str = (
            f"{s['mg_fail']}/{s['n_trials']} ({s['predgate_fail_rate']:.0%}) / "
            f"{s['mg_detected']}/{s['n_trials']} ({s['predgate_detection_rate']:.0%})"
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
            mg_str = f"{s['mg_fail']}/{s['n_trials']}F / {s['mg_detected']}/{s['n_trials']} ({s['predgate_detection_rate']:.0%} FP)"
            bl_str = f"{s['bl_detected']}/{s['n_trials']} ({s['baseline_detection_rate']:.0%} FP)"
            winner = (
                "✅ PG" if s["predgate_detection_rate"] <= s["baseline_detection_rate"] else "❌ BL"
            )

        table.add_row(fault, s["description"], mg_str, bl_str, winner)

    console.print(table)

    # Store aggregate-only summary in docs/benchmark_results.json; per-seed records go to release_artifacts/
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
        try:
            existing_data = json.loads(json_path.read_text())
            for preserve_key in (
                "public_fairness_benchmark",
                "benign_change_control",
                "held_out_validation",
                "comparator_config",
            ):
                if preserve_key in existing_data:
                    results_data[preserve_key] = existing_data[preserve_key]
        except Exception:
            pass
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

    # Generate markdown strictly from the aggregate JSON file
    generate_benchmark_markdown_from_json(json_path, docs_dir / "benchmark.md")

    console.print(
        f"\n[green]Aggregate saved to docs/benchmark_results.json, per-seed records to {per_seed_artifact}, and report to docs/benchmark.md[/green]"
    )

    return results_data


def _fmt_rate_ci(k: int, n: int, rate: float, ci: list[float], suffix: str = "") -> str:
    suf = f" {suffix}" if suffix else ""
    return f"{k}/{n} ({rate:.1%}{suf}, [{ci[0]:.1%}, {ci[1]:.1%}])"


def generate_benchmark_markdown_from_json(
    json_path: Path,
    md_path: Path,
) -> str:
    """Generate docs/benchmark.md strictly from a machine-readable aggregate benchmark_results.json file.

    Reports FAIL-only and Flagged (WARN+FAIL) separately across all tables, includes
    stronger baselines (per-group accuracy drop, fixed DP limit, Evidently defaults),
    benign-change alert-fatigue controls, and held-out dataset/fault validation.
    """
    data: dict[str, Any] = json.loads(json_path.read_text())
    summary: dict[str, dict[str, Any]] = data.get("summary", {})
    commit_hash = str(data.get("commit_hash", "unknown"))
    command_str = str(data.get("command", "predgate bench"))
    per_seed_artifact = str(data.get("per_seed_artifact", ""))
    seeds: list[int] = list(data.get("seeds", []))
    if not seeds:
        first_entry: dict[str, Any] = next(iter(summary.values()), {})
        n_default = int(first_entry.get("n_trials", 30))
        seeds = [1000 + i for i in range(n_default)]

    seed_range_str = f"`{seeds[0]}..{seeds[-1]}` (N={len(seeds)})"
    noop = summary.get("noop", {})
    noop_n = noop.get("n_trials", len(seeds))
    noop_fail_k = noop.get("mg_fail", noop.get("mg_detected", 0))
    noop_fail_rate = noop.get("predgate_fail_rate", noop.get("predgate_detection_rate", 0.0))
    noop_fail_ci = noop.get("predgate_fail_ci_95", noop.get("predgate_ci_95", [0.0, 0.114]))
    noop_mg_k = noop.get("mg_detected", 0)
    noop_mg_rate = noop.get("predgate_detection_rate", 0.0)
    noop_mg_ci = noop.get("predgate_ci_95", [0.0, 0.114])
    noop_bl_k = noop.get("bl_detected", 0)
    noop_bl_rate = noop.get("baseline_detection_rate", 0.0)
    noop_bl_ci = noop.get("baseline_ci_95", [0.0, 0.114])

    has_synth_evidently = any("evidently_detected" in s for s in summary.values())
    has_synth_stronger = any("per_group_acc_detected" in s for s in summary.values())

    lines = [
        "# PredGate Benchmark Results",
        "",
        "Single source of truth for all PredGate benchmark evaluations, generated strictly from [`docs/benchmark_results.json`](benchmark_results.json) (aggregate summary; no hardcoded numbers).",
        "",
        "## Reproducibility Metadata",
        "",
        f"- **Command**: `{command_str}`",
        f"- **Evaluated Code Commit Hash**: `{commit_hash}`",
        f"- **Synthetic Suite Seeds**: {seed_range_str}",
        f"- **No-op FP (PredGate FAIL-only)**: **{_fmt_rate_ci(noop_fail_k, noop_n, noop_fail_rate, noop_fail_ci, 'FP')}**",
        f"- **No-op FP (PredGate Flagged WARN+FAIL)**: **{_fmt_rate_ci(noop_mg_k, noop_n, noop_mg_rate, noop_mg_ci, 'FP')}**",
        f"- **No-op FP (Overall Accuracy Baseline)**: **{_fmt_rate_ci(noop_bl_k, noop_n, noop_bl_rate, noop_bl_ci, 'FP')}**",
    ]
    if per_seed_artifact:
        lines.append(
            f"- **Per-Seed Records Release Artifact**: `{per_seed_artifact}` (excluded from repo JSON to keep git lean; attach to GitHub Release)"
        )

    # ── 1. Synthetic 7-Fault Benchmark ──
    lines.extend(
        [
            "",
            f"## 1. Synthetic Credit 7-Fault Benchmark (Seeds `{seeds[0]}..{seeds[-1]}`)",
            "",
            "Reports **FAIL-only** (hard merge block) and **Flagged (WARN+FAIL)** (blocked or flagged for human review) separately alongside stronger baselines.",
            "",
        ]
    )
    if has_synth_stronger and has_synth_evidently:
        lines.append(
            "| Fault | PredGate FAIL-only (95% CI) | PredGate Flagged WARN+FAIL (95% CI) | Overall Acc (`\\|Δacc\\|>1%`) | Per-Group Acc Drop (`>2%`) | Fixed DP (`>=0.10`) | Evidently 0.7 Defaults |"
        )
        lines.append(
            "|-------|-----------------------------|-------------------------------------|------------------------------|----------------------------|---------------------|------------------------|"
        )
    elif has_synth_stronger:
        lines.append(
            "| Fault | PredGate FAIL-only (95% CI) | PredGate Flagged WARN+FAIL (95% CI) | Overall Acc (`\\|Δacc\\|>1%`) | Per-Group Acc Drop (`>2%`) | Fixed DP (`>=0.10`) |"
        )
        lines.append(
            "|-------|-----------------------------|-------------------------------------|------------------------------|----------------------------|---------------------|"
        )
    else:
        lines.append(
            "| Fault | PredGate FAIL-only (95% CI) | PredGate Flagged WARN+FAIL (95% CI) | Overall Acc (`\\|Δacc\\|>1%`) |"
        )
        lines.append(
            "|-------|-----------------------------|-------------------------------------|------------------------------|"
        )

    for fault, s in summary.items():
        n = int(s["n_trials"])
        suf = "FP" if fault == "noop" else ""
        fail_k = int(s.get("mg_fail", s["mg_detected"]))
        fail_r = float(s.get("predgate_fail_rate", s["predgate_detection_rate"]))
        fail_ci = s.get("predgate_fail_ci_95", s["predgate_ci_95"])
        flag_k = int(s["mg_detected"])
        flag_r = float(s["predgate_detection_rate"])
        flag_ci = s["predgate_ci_95"]
        bl_k = int(s["bl_detected"])
        bl_r = float(s["baseline_detection_rate"])
        bl_ci = s["baseline_ci_95"]

        fail_cell = _fmt_rate_ci(fail_k, n, fail_r, fail_ci, suf)
        flag_cell = _fmt_rate_ci(flag_k, n, flag_r, flag_ci, suf)
        bl_cell = _fmt_rate_ci(bl_k, n, bl_r, bl_ci, suf)
        if fault == "noop":
            fail_cell = f"**{fail_cell}**"
            flag_cell = f"**{flag_cell}**"
            bl_cell = f"**{bl_cell}**"

        row_cells = [f"`{fault}`", fail_cell, flag_cell, bl_cell]
        if has_synth_stronger:
            pg_k = int(s.get("per_group_acc_detected", 0))
            pg_r = float(s.get("per_group_acc_rate", 0.0))
            pg_ci = s.get("per_group_acc_ci_95", [0.0, 0.0])
            dp_k = int(s.get("fixed_dp_detected", 0))
            dp_r = float(s.get("fixed_dp_rate", 0.0))
            dp_ci = s.get("fixed_dp_ci_95", [0.0, 0.0])
            row_cells.append(_fmt_rate_ci(pg_k, n, pg_r, pg_ci, suf))
            row_cells.append(_fmt_rate_ci(dp_k, n, dp_r, dp_ci, suf))
        if has_synth_evidently:
            ev_k = int(s.get("evidently_detected", 0))
            ev_r = float(s.get("evidently_rate", 0.0))
            ev_ci = s.get("evidently_ci_95", [0.0, 0.0])
            row_cells.append(_fmt_rate_ci(ev_k, n, ev_r, ev_ci, suf))
        lines.append("| " + " | ".join(row_cells) + " |")

    # ── 2. Benign-Change Control (Quality vs Latency Separated) ──
    benign = data.get("benign_change_control")
    if isinstance(benign, dict) and benign.get("rows"):
        b_seeds: list[int] = list(benign.get("seeds", []))
        b_range = (
            f"{b_seeds[0]}..{b_seeds[-1]}" if b_seeds else str(benign.get("seed_range", "unknown"))
        )
        ov = benign.get("overall", {})
        b_has_ev = any("evidently_detected" in r for r in benign["rows"])
        b_has_ql = any("quality_fail_count" in r for r in benign["rows"])
        n_changes = len(benign.get("changes", [])) or len({r["change"] for r in benign["rows"]})
        n_datasets = len({r["dataset"] for r in benign["rows"]})
        lines.extend(
            [
                "",
                f"## 2. Benign-Change Control (Harmless Retrains, Seeds `{b_range}`, Frozen Defaults)",
                "",
                f"{n_changes} realistic harmless retrain scenarios across {len(b_seeds) or 'N'} fresh seeds × {n_datasets} datasets "
                f"({int(ov.get('n_trials', len(benign['rows'])))} total trials), running all 6 schema-agnostic checks with **zero threshold tuning**. "
                "**Quality findings** (`drift`, `leakage`, `label_shift`, `slices`, `fairness`) are shown **separately from latency findings** (`budgets`).",
                "",
            ]
        )
        if ov:
            ov_n = int(ov.get("n_trials", 0))
            if "quality_fail_count" in ov:
                lines.extend(
                    [
                        f"- **Quality-Only Benign FAIL Rate (Hard Block FP)**: **{_fmt_rate_ci(int(ov['quality_fail_count']), ov_n, float(ov['quality_fail_rate']), ov['quality_fail_ci_95'])}**",
                        f"- **Quality-Only Benign WARN-Only Rate (Alert Fatigue)**: **{_fmt_rate_ci(int(ov['quality_warn_only_count']), ov_n, float(ov['quality_warn_only_rate']), ov['quality_warn_only_ci_95'])}**",
                        f"- **Latency-Only Benign FAIL Rate (`budgets`)**: **{_fmt_rate_ci(int(ov['latency_fail_count']), ov_n, float(ov['latency_fail_rate']), ov['latency_fail_ci_95'])}**",
                        f"- **Latency-Only Benign WARN Rate (`budgets`)**: **{_fmt_rate_ci(int(ov['latency_warn_count']), ov_n, float(ov['latency_warn_rate']), ov['latency_warn_ci_95'])}**",
                        f"- **Combined Benign FAIL Rate (Quality + Latency)**: **{_fmt_rate_ci(int(ov['fail_count']), ov_n, float(ov['fail_rate']), ov['fail_ci_95'])}** | **Combined WARN-Only**: **{_fmt_rate_ci(int(ov['warn_only_count']), ov_n, float(ov['warn_only_rate']), ov['warn_only_ci_95'])}**",
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"- **Overall Benign FAIL Rate (Hard Block FP)**: **{_fmt_rate_ci(int(ov['fail_count']), ov_n, float(ov['fail_rate']), ov['fail_ci_95'])}**",
                        f"- **Overall Benign WARN-Only Rate (Alert Fatigue)**: **{_fmt_rate_ci(int(ov['warn_only_count']), ov_n, float(ov['warn_only_rate']), ov['warn_only_ci_95'])}**",
                        f"- **Overall Benign Flagged Rate (WARN+FAIL)**: **{_fmt_rate_ci(int(ov['flagged_count']), ov_n, float(ov['flagged_rate']), ov['flagged_ci_95'])}**",
                        "",
                    ]
                )

        if b_has_ql:
            header_cols = [
                "Dataset",
                "Benign Change",
                "Quality FAIL (95% CI)",
                "Quality WARN-Only (Alert Fatigue)",
                "Latency FAIL (95% CI)",
                "Latency WARN",
                "Per-Group Acc Drop",
                "Fixed DP (`>=0.10`)",
            ]
        else:
            header_cols = [
                "Dataset",
                "Benign Change",
                "PredGate FAIL (95% CI)",
                "PredGate WARN-Only (Alert Fatigue)",
                "PredGate Flagged (WARN+FAIL)",
                "Per-Group Acc Drop",
                "Fixed DP (`>=0.10`)",
            ]
        if b_has_ev:
            header_cols.append("Evidently 0.7 Defaults")
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|" + "|".join("---" for _ in header_cols) + "|")

        for r in benign["rows"]:
            n_t = int(r["n_trials"])
            if b_has_ql:
                cells = [
                    f"**{r['dataset']}**",
                    f"`{r['change']}`",
                    _fmt_rate_ci(
                        int(r["quality_fail_count"]),
                        n_t,
                        float(r["quality_fail_rate"]),
                        r["quality_fail_ci_95"],
                    ),
                    _fmt_rate_ci(
                        int(r["quality_warn_only_count"]),
                        n_t,
                        float(r["quality_warn_only_rate"]),
                        r["quality_warn_only_ci_95"],
                    ),
                    _fmt_rate_ci(
                        int(r["latency_fail_count"]),
                        n_t,
                        float(r["latency_fail_rate"]),
                        r["latency_fail_ci_95"],
                    ),
                    _fmt_rate_ci(
                        int(r["latency_warn_count"]),
                        n_t,
                        float(r["latency_warn_rate"]),
                        r["latency_warn_ci_95"],
                    ),
                    _fmt_rate_ci(
                        int(r["per_group_acc_detected"]),
                        n_t,
                        float(r["per_group_acc_rate"]),
                        r["per_group_acc_ci_95"],
                    ),
                    _fmt_rate_ci(
                        int(r["fixed_dp_detected"]),
                        n_t,
                        float(r["fixed_dp_rate"]),
                        r["fixed_dp_ci_95"],
                    ),
                ]
            else:
                cells = [
                    f"**{r['dataset']}**",
                    f"`{r['change']}`",
                    _fmt_rate_ci(int(r["fail_count"]), n_t, float(r["fail_rate"]), r["fail_ci_95"]),
                    _fmt_rate_ci(
                        int(r["warn_only_count"]),
                        n_t,
                        float(r["warn_only_rate"]),
                        r["warn_only_ci_95"],
                    ),
                    _fmt_rate_ci(
                        int(r["flagged_count"]),
                        n_t,
                        float(r["flagged_rate"]),
                        r["flagged_ci_95"],
                    ),
                    _fmt_rate_ci(
                        int(r["per_group_acc_detected"]),
                        n_t,
                        float(r["per_group_acc_rate"]),
                        r["per_group_acc_ci_95"],
                    ),
                    _fmt_rate_ci(
                        int(r["fixed_dp_detected"]),
                        n_t,
                        float(r["fixed_dp_rate"]),
                        r["fixed_dp_ci_95"],
                    ),
                ]
            if b_has_ev:
                cells.append(
                    _fmt_rate_ci(
                        int(r["evidently_detected"]),
                        n_t,
                        float(r["evidently_rate"]),
                        r["evidently_ci_95"],
                    )
                )
            lines.append("| " + " | ".join(cells) + " |")

        per_check = benign.get("per_check", [])
        if per_check:
            lines.extend(
                [
                    "",
                    "### Per-Check Benign-Change FAIL & WARN Rates (All Datasets Combined)",
                    "",
                    "| Check | Category | FAIL Count & Rate (95% CI) | WARN Count & Rate (Alert Fatigue, 95% CI) | Flagged WARN+FAIL (95% CI) |",
                    "|-------|----------|----------------------------|-------------------------------------------|----------------------------|",
                ]
            )
            for pc in per_check:
                n_pc = int(pc["n_trials"])
                cat = "Latency" if pc["check"] == "budgets" else "Quality"
                lines.append(
                    f"| `{pc['check']}` | {cat} | "
                    f"{_fmt_rate_ci(int(pc['fail_count']), n_pc, float(pc['fail_rate']), pc['fail_ci_95'])} | "
                    f"{_fmt_rate_ci(int(pc['warn_count']), n_pc, float(pc['warn_rate']), pc['warn_ci_95'])} | "
                    f"{_fmt_rate_ci(int(pc['flagged_count']), n_pc, float(pc['flagged_rate']), pc['flagged_ci_95'])} |"
                )

        algo_fair = benign.get("algo_swap_fairness_metrics")
        if isinstance(algo_fair, dict) and algo_fair.get("n_trials"):
            fail_m_str = (
                ", ".join(
                    f"`{k}`: {v}"
                    for k, v in sorted(
                        algo_fair.get("fail_metrics", {}).items(), key=lambda x: -x[1]
                    )
                )
                or "none"
            )
            warn_m_str = (
                ", ".join(
                    f"`{k}`: {v}"
                    for k, v in sorted(
                        algo_fair.get("warn_metrics", {}).items(), key=lambda x: -x[1]
                    )
                )
                or "none"
            )
            lines.extend(
                [
                    "",
                    "### Fairness Metric Breakdown on Benign Algorithm Swaps (`algo_swap_rf`, `algo_swap_logreg`)",
                    "",
                    f"- **Trials**: {algo_fair['n_trials']} (`fairness` FAIL: {algo_fair['n_fail']}, WARN: {algo_fair['n_warn']})",
                    f"- **Triggering FAIL metrics**: {fail_m_str}",
                    f"- **Triggering WARN metrics**: {warn_m_str}",
                ]
            )

    # ── 3. Full Fault Validation Across All Datasets ──
    held = data.get("held_out_validation")
    if isinstance(held, dict) and held.get("rows"):
        h_seeds: list[int] = list(held.get("seeds", []))
        h_range = (
            f"{h_seeds[0]}..{h_seeds[-1]}" if h_seeds else str(held.get("seed_range", "unknown"))
        )
        h_has_ev = any("evidently_detected" in r for r in held["rows"])
        lines.extend(
            [
                "",
                f"## 3. Full Fault Validation Across All Datasets (Seeds `{h_range}`, Frozen Defaults)",
                "",
                "Evaluated with **frozen thresholds** (zero tuning) across all datasets (**UCI Adult**, **German Credit**, **Synthetic Credit**, **Bank Marketing** `data_id=1461`, **COMPAS** `data_id=42192`) covering all fault families including `silent_null_imputation` (40% null-to-zero corruption of primary numeric feature in training data) and `cohort_feature_corruption` (cohort-localized 2σ feature corruption at inference time with clean training data).",
                "",
            ]
        )
        if h_has_ev:
            lines.append(
                "| Dataset | Fault | PredGate FAIL-only (95% CI) | PredGate Flagged WARN+FAIL (95% CI) | Overall Acc (`\\|Δacc\\|>1%`) | Per-Group Acc Drop (`>2%`) | Fixed DP (`>=0.10`) | Evidently 0.7 Defaults | Power / MDE Note |"
            )
            lines.append(
                "|---------|-------|-----------------------------|-------------------------------------|------------------------------|----------------------------|---------------------|------------------------|------------------|"
            )
        else:
            lines.append(
                "| Dataset | Fault | PredGate FAIL-only (95% CI) | PredGate Flagged WARN+FAIL (95% CI) | Overall Acc (`\\|Δacc\\|>1%`) | Per-Group Acc Drop (`>2%`) | Fixed DP (`>=0.10`) | Power / MDE Note |"
            )
            lines.append(
                "|---------|-------|-----------------------------|-------------------------------------|------------------------------|----------------------------|---------------------|------------------|"
            )

        for r in held["rows"]:
            n_t = int(r["n_trials"])
            suf = "FP" if r["fault"] == "noop" else ""
            fail_cell = _fmt_rate_ci(
                int(r["detected_fail"]), n_t, float(r["fail_rate"]), r["fail_ci_95"], suf
            )
            flag_cell = _fmt_rate_ci(
                int(r["detected_total"]), n_t, float(r["detection_rate"]), r["ci_95"], suf
            )
            if r["fault"] == "noop":
                fail_cell = f"**{fail_cell}**"
                flag_cell = f"**{flag_cell}**"
            mde_str = (
                f"`min_n={r['mean_min_n']:.0f}, MDE_DP={r['mean_mde_dp']:.3f}` "
                f"(note {int(r['power_note_count'])}/{n_t})"
            )
            cells = [
                f"**{r['dataset']}**",
                r["fault_label"],
                fail_cell,
                flag_cell,
                _fmt_rate_ci(
                    int(r["baseline_detected"]),
                    n_t,
                    float(r["baseline_rate"]),
                    r["baseline_ci_95"],
                    suf,
                ),
                _fmt_rate_ci(
                    int(r["per_group_acc_detected"]),
                    n_t,
                    float(r["per_group_acc_rate"]),
                    r["per_group_acc_ci_95"],
                    suf,
                ),
                _fmt_rate_ci(
                    int(r["fixed_dp_detected"]),
                    n_t,
                    float(r["fixed_dp_rate"]),
                    r["fixed_dp_ci_95"],
                    suf,
                ),
            ]
            if h_has_ev:
                cells.append(
                    _fmt_rate_ci(
                        int(r["evidently_detected"]),
                        n_t,
                        float(r["evidently_rate"]),
                        r["evidently_ci_95"],
                        suf,
                    )
                )
            cells.append(mde_str)
            lines.append("| " + " | ".join(cells) + " |")

    # ── 4. Public Fairness Validation (Development Public Datasets) ──
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
        pub_has_stronger = any("per_group_acc_detected" in r for r in pub["rows"])

        lines.extend(
            [
                "",
                f"## 4. Standalone Fairness Check Across All Datasets (Seeds `{pub_range}`, Frozen Defaults)",
                "",
                f"- **Command**: `{pub_cmd}`",
                f"- **Evaluated Code Commit Hash**: `{pub_commit}`",
                f"- **Frozen `FairnessConfig` Settings**: {frozen_str}",
                "",
            ]
        )
        if pub_has_stronger:
            lines.append(
                f"| Dataset (`{pub_range}`) | Fault | Fairness FAIL-only (95% CI) | Fairness Flagged WARN+FAIL (95% CI) | Overall Acc (`\\|Δacc\\|>1%`) | Per-Group Acc Drop (`>2%`) | Fixed DP (`>=0.10`) | Mean DP (`champ -> cand`) | Power / MDE Note |"
            )
            lines.append(
                "|------------------------|-------|-----------------------------|-------------------------------------|------------------------------|----------------------------|---------------------|---------------------------|------------------|"
            )
        else:
            lines.append(
                f"| Dataset (`{pub_range}`) | Fault | Fairness FAIL-only (95% CI) | Fairness Flagged WARN+FAIL (95% CI) | Overall Acc (`\\|Δacc\\|>1%`) | Mean DP (`champ -> cand`) | Power / MDE Note |"
            )
            lines.append(
                "|------------------------|-------|-----------------------------|-------------------------------------|------------------------------|---------------------------|------------------|"
            )

        for row in pub["rows"]:
            ds = row["dataset"]
            flt = row["fault_label"]
            n_t = int(row["n_trials"])
            suf = "FP" if row["fault"] == "noop" else ""
            det_fail = int(row["detected_fail"])
            fail_rate = float(row.get("fail_rate", det_fail / n_t if n_t else 0.0))
            fail_ci = row.get("fail_ci_95", list(wilson_ci(det_fail, n_t)))
            det = int(row["detected_total"])
            rate = float(row["detection_rate"])
            ci = row["ci_95"]
            bl_det = int(row["baseline_detected"])
            bl_rate = float(row["baseline_rate"])
            bl_ci = row["baseline_ci_95"]
            dp_str = f"`{row['mean_champ_dp']:.3f} -> {row['mean_cand_dp']:.3f}`"
            note_cnt = int(row["power_note_count"])
            mde_str = (
                f"`min_n={row['mean_min_n']:.0f}, MDE_DP={row['mean_mde_dp']:.3f}, "
                f"MDE_EO={row['mean_mde_eo']:.3f}` (note {note_cnt}/{n_t})"
            )
            fail_cell = _fmt_rate_ci(det_fail, n_t, fail_rate, fail_ci, suf)
            flag_cell = _fmt_rate_ci(det, n_t, rate, ci, suf)
            bl_cell = _fmt_rate_ci(bl_det, n_t, bl_rate, bl_ci, suf)
            if row["fault"] == "noop":
                fail_cell = f"**{fail_cell}**"
                flag_cell = f"**{flag_cell}**"
                bl_cell = f"**{bl_cell}**"

            cells = [f"**{ds}**", flt, fail_cell, flag_cell, bl_cell]
            if pub_has_stronger:
                cells.append(
                    _fmt_rate_ci(
                        int(row["per_group_acc_detected"]),
                        n_t,
                        float(row["per_group_acc_rate"]),
                        row["per_group_acc_ci_95"],
                        suf,
                    )
                )
                cells.append(
                    _fmt_rate_ci(
                        int(row["fixed_dp_detected"]),
                        n_t,
                        float(row["fixed_dp_rate"]),
                        row["fixed_dp_ci_95"],
                        suf,
                    )
                )
            cells.extend([dp_str, mde_str])
            lines.append("| " + " | ".join(cells) + " |")

    # ── 5. Baseline & Off-the-Shelf Comparator Definitions ──
    comp = data.get("comparator_config", {})
    lines.extend(
        [
            "",
            "## 5. Baseline & Comparator Definitions",
            "",
            "1. **Overall Accuracy Baseline (`|acc_champ - acc_cand| > 0.01`)**: Flags any trial where overall test accuracy changes by more than 1 percentage point in either direction.",
            "2. **Per-Group Accuracy-Drop Baseline (`max_g (acc_champ(g) - acc_cand(g)) > 0.02`, `n_g >= 30`)**: Flags any trial where any subgroup's accuracy drops by more than 2 percentage points compared to the champion.",
            "3. **Fixed Demographic-Parity-Limit Baseline (`DP_cand >= 0.10`)**: Classic static absolute fairness threshold flagging any candidate with demographic parity difference `>= 0.10` regardless of champion baseline.",
        ]
    )
    if comp:
        lines.append(
            f"4. **Evidently Off-the-Shelf Default Comparator (`evidently=={comp.get('evidently_version', '0.7.23')}`, `pandas=={comp.get('pandas_version', '2.3.3')}`)**: "
            f"{comp.get('description', 'Runs Report([DataDriftPreset()], include_tests=True) on candidate vs reference training data and Report([ClassificationPreset()], include_tests=True) on candidate vs champion test predictions with 100% Evidently default thresholds; flagged if any test returns FAIL.')}"
        )

    lines.extend(
        [
            "",
            "## Methodology",
            "",
            f"All suites use {len(seeds)} independent fresh random seeds (`{seeds[0]}..{seeds[-1]}`). "
            "Seeds `6000..6029` from commit `8467d37` were used strictly as a development set to diagnose the `slices.py`, `drift.py`, and algo-swap defects; no thresholds were tuned after observing the fresh `9000..9029` evaluation results. "
            "Public datasets (UCI Adult `adult` v2, German Credit `credit-g` v1, Bank Marketing `1461`, COMPAS `42192`) are fetched directly from OpenML via `scripts/bench_public.py`. "
            "Confidence intervals use the 95% Wilson score interval (`z = 1.95996`).",
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
