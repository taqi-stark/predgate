"""Reproducible end-to-end PredGate benchmark runner on fresh seeds (default: 9000..9029).

Runs all suites across ALL 5 datasets (UCI Adult, German Credit, Synthetic Credit,
Bank Marketing, COMPAS) with frozen defaults and ZERO post-hoc tuning:
1. Synthetic 7-fault benchmark (feature_leakage, label_shift, degraded_slice,
   schema_drift, latency_regression, fairness_regression, noop)
2. Benign-change control: 9 harmless retrains (original 6 + drop_irrelevant_feature,
   change_imputation, rescale_inputs) x 30 fresh seeds x 5 datasets = 1,350 trials,
   reporting Quality findings separately from Latency findings alongside per-check
   FAIL/WARN rates, algo-swap fairness metric triggers, stronger baselines, and
   Evidently 0.7.23 defaults
3. Full fault validation across all 5 datasets covering all fault families
   (silent_null_imputation, cohort_feature_corruption, fairness_threshold_shift,
   fairness_threshold_shift_hi, fairness_proxy_feature, fairness_regression, noop)
4. Standalone public fairness check summary (Adult, German Credit, Synthetic Credit)
   for apples-to-apples comparison against prior commits

Writes aggregate-only summary JSON to docs/benchmark_results.json, writes raw per-seed
records to release_artifacts/benchmark_per_seed_<commit>.json, and renders
docs/benchmark.md strictly from docs/benchmark_results.json.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from predgate.benchmark import (
    FAULT_DESCRIPTIONS,
    GroupThresholdModel,
    _generate_data,
    _get_git_commit,
    _train_latency_model,
    _train_model,
    fixed_dp_limit_baseline,
    generate_benchmark_markdown_from_json,
    inject_fault,
    per_group_accuracy_drop_baseline,
    wilson_ci,
)
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

ALL_DATASETS: list[str] = [
    "UCI Adult",
    "German Credit",
    "Synthetic Credit",
    "Bank Marketing",
    "COMPAS",
]

SYNTHETIC_FAULTS: list[str] = [
    "feature_leakage",
    "label_shift",
    "degraded_slice",
    "schema_drift",
    "latency_regression",
    "fairness_regression",
    "noop",
]

PUBLIC_FAULTS: list[tuple[str, str]] = [
    ("fairness_threshold_shift", "`fairness_threshold_shift` (`τ=0.25`)"),
    ("fairness_threshold_shift_hi", "`fairness_threshold_shift_hi` (`τ=0.75`)"),
    ("fairness_proxy_feature", "`fairness_proxy_feature`"),
    ("fairness_regression", "`fairness_regression`"),
    ("noop", "`noop` (FP rate)"),
]

BENIGN_CHANGES: list[str] = [
    "hyperparam_change",
    "algo_swap_rf",
    "algo_swap_logreg",
    "more_data_20pct",
    "irrelevant_feature",
    "preprocessing_change",
    "drop_irrelevant_feature",
    "change_imputation",
    "rescale_inputs",
]

ALL_DATASET_FAULTS: list[tuple[str, str]] = [
    ("silent_null_imputation", "`silent_null_imputation` (40% null->0 in train)"),
    ("cohort_feature_corruption", "`cohort_feature_corruption` (2σ cohort noise)"),
    ("fairness_threshold_shift", "`fairness_threshold_shift` (`τ=0.25`)"),
    ("fairness_threshold_shift_hi", "`fairness_threshold_shift_hi` (`τ=0.75`)"),
    ("fairness_proxy_feature", "`fairness_proxy_feature`"),
    ("fairness_regression", "`fairness_regression`"),
    ("noop", "`noop` (FP rate)"),
]

QUALITY_CHECKS: list[str] = ["drift", "leakage", "label_shift", "slices", "fairness"]
ALL_CHECKS: list[str] = ["drift", "leakage", "label_shift", "slices", "fairness", "budgets"]

# ── Per-worker persistent Evidently comparator client ──

_EVIDENTLY_PROC: subprocess.Popen[str] | None = None
_EVIDENTLY_TMPDIR: tempfile.TemporaryDirectory[str] | None = None


def _close_evidently_worker() -> None:
    global _EVIDENTLY_PROC, _EVIDENTLY_TMPDIR
    if _EVIDENTLY_PROC is not None:
        try:
            if _EVIDENTLY_PROC.stdin:
                _EVIDENTLY_PROC.stdin.close()
            _EVIDENTLY_PROC.wait(timeout=5)
        except Exception:
            _EVIDENTLY_PROC.kill()
        _EVIDENTLY_PROC = None
    if _EVIDENTLY_TMPDIR is not None:
        _EVIDENTLY_TMPDIR.cleanup()
        _EVIDENTLY_TMPDIR = None


def _init_worker(evidently_python: str | None) -> None:
    global _EVIDENTLY_PROC, _EVIDENTLY_TMPDIR
    import warnings

    warnings.filterwarnings("ignore", message="Found unknown categories")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    if not evidently_python:
        return
    comp_script = Path(__file__).resolve().parent / "evidently_comparator.py"
    _EVIDENTLY_TMPDIR = tempfile.TemporaryDirectory(prefix="evidently_cmp_")
    _EVIDENTLY_PROC = subprocess.Popen(
        [evidently_python, str(comp_script), "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    atexit.register(_close_evidently_worker)


def _run_evidently_eval(
    train_ref: pd.DataFrame,
    train_cur: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    champ_preds: np.ndarray,
    cand_preds: np.ndarray,
) -> bool | None:
    """Invoke the persistent Evidently 0.7.23 default-preset worker on this trial."""
    if _EVIDENTLY_PROC is None or _EVIDENTLY_TMPDIR is None:
        return None
    assert _EVIDENTLY_PROC.stdin is not None and _EVIDENTLY_PROC.stdout is not None
    tmp = Path(_EVIDENTLY_TMPDIR.name)
    tr_ref_p: str | None = None
    tr_cur_p: str | None = None
    # Skip redundant DataDriftPreset() when shared columns of train_cur are byte-identical to train_ref
    common_cols = [c for c in train_ref.columns if c in train_cur.columns]
    if not train_ref[common_cols].equals(train_cur[common_cols]):
        tr_ref_path = tmp / "tr_ref.csv"
        tr_cur_path = tmp / "tr_cur.csv"
        train_ref.to_csv(tr_ref_path, index=False)
        train_cur.to_csv(tr_cur_path, index=False)
        tr_ref_p = str(tr_ref_path)
        tr_cur_p = str(tr_cur_path)

    te_ref_p = tmp / "te_ref.csv"
    te_cur_p = tmp / "te_cur.csv"
    te_ref = test_df[[target_col]].copy()
    te_ref["__pred__"] = np.asarray(champ_preds, dtype=int)
    te_cur = test_df[[target_col]].copy()
    te_cur["__pred__"] = np.asarray(cand_preds, dtype=int)
    te_ref.to_csv(te_ref_p, index=False)
    te_cur.to_csv(te_cur_p, index=False)

    req = {
        "train_ref": tr_ref_p,
        "train_cur": tr_cur_p,
        "test_ref": str(te_ref_p),
        "test_cur": str(te_cur_p),
        "target": target_col,
        "pred_col": "__pred__",
    }
    _EVIDENTLY_PROC.stdin.write(json.dumps(req) + "\n")
    _EVIDENTLY_PROC.stdin.flush()
    resp_line = _EVIDENTLY_PROC.stdout.readline()
    if not resp_line:
        return None
    resp = json.loads(resp_line)
    return bool(resp.get("failed", False))


# ── Dataset loaders (all OpenML) ──


def _normalize_frame(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == target_col:
            continue
        if (
            isinstance(out[col].dtype, pd.CategoricalDtype)
            or out[col].dtype == object
            or str(out[col].dtype) in ("string", "str")
        ):
            out[col] = out[col].astype(str).fillna("missing").str.strip().astype(object)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)
    return out


def _load_adult_frame() -> pd.DataFrame:
    bunch = fetch_openml("adult", version=2, as_frame=True)
    df = bunch.frame.copy()
    target_col = bunch.target.name if hasattr(bunch.target, "name") else "class"
    df["target"] = (df[target_col].astype(str).str.strip() == ">50K").astype(int)
    df = df.drop(columns=[target_col])
    return _normalize_frame(df, "target")


def _load_german_frame() -> pd.DataFrame:
    bunch = fetch_openml("credit-g", version=1, as_frame=True)
    df = bunch.frame.copy()
    target_col = bunch.target.name if hasattr(bunch.target, "name") else "class"
    df["target"] = (df[target_col].astype(str).str.strip() == "bad").astype(int)
    df = df.drop(columns=[target_col])
    df["sex"] = np.where(
        df["personal_status"].astype(str).str.lower().str.startswith("female"),
        "female",
        "male",
    )
    df = df.drop(columns=["personal_status"])
    return _normalize_frame(df, "target")


def _load_bank_marketing_frame() -> pd.DataFrame:
    bunch = fetch_openml(data_id=1461, parser="liac-arff", as_frame=True)
    df = bunch.frame.copy()
    rename_map = {
        "V1": "age",
        "V2": "job",
        "V3": "marital",
        "V4": "education",
        "V5": "default_credit",
        "V6": "balance",
        "V7": "housing",
        "V8": "loan",
        "V9": "contact",
        "V10": "day",
        "V11": "month",
        "V12": "duration",
        "V13": "campaign",
        "V14": "pdays",
        "V15": "previous",
        "V16": "poutcome",
    }
    df = df.rename(columns=rename_map)
    target_col = bunch.target.name if hasattr(bunch.target, "name") else "Class"
    df["target"] = (df[target_col].astype(str).str.strip() == "2").astype(int)
    df = df.drop(columns=[target_col])
    return _normalize_frame(df, "target")


def _load_compas_frame() -> pd.DataFrame:
    bunch = fetch_openml(data_id=42192, parser="liac-arff", as_frame=True)
    df = bunch.frame.copy()
    target_col = bunch.target.name if hasattr(bunch.target, "name") else "two_year_recid"
    df["target"] = (df[target_col].astype(str).str.strip() == "1").astype(int)
    df = df.drop(columns=[target_col])
    df["race"] = np.where(
        df["race_African-American"].astype(str).str.strip() == "1",
        "African-American",
        "Caucasian",
    )
    df["sex"] = np.where(df["sex"].astype(str).str.strip() == "1", "Male", "Female")
    drop_cols = [c for c in ("race_African-American", "race_Caucasian") if c in df.columns]
    df = df.drop(columns=drop_cols)
    return _normalize_frame(df, "target")


# ── Model builders ──


def _build_preprocessor(X: pd.DataFrame, mode: str = "default") -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    if mode == "alt_preprocessing":
        num_pipe: Any = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        cat_enc: Any = OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)
    elif mode == "champ_imputation":
        num_pipe = SimpleImputer(strategy="median")
        cat_enc = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    elif mode == "change_imputation":
        num_pipe = SimpleImputer(strategy="mean", add_indicator=True)
        cat_enc = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    elif mode == "rescale_inputs":
        num_pipe = RobustScaler()
        cat_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        num_pipe = "passthrough"
        cat_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_enc, cat_cols)],
        remainder="drop",
    )


def _train_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    variant: str = "champion_gbdt",
) -> Pipeline:
    X_norm = X.copy()
    if variant not in ("champ_imputation", "change_imputation"):
        X_norm = _normalize_frame(X_norm.assign(_tmp=0), "_tmp").drop(columns=["_tmp"])
    if variant == "champion_gbdt":
        prep = _build_preprocessor(X_norm, mode="default")
        clf: Any = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=seed)
    elif variant == "hyperparam_change":
        prep = _build_preprocessor(X_norm, mode="default")
        clf = GradientBoostingClassifier(
            n_estimators=75, max_depth=4, learning_rate=0.08, random_state=seed
        )
    elif variant == "algo_swap_rf":
        prep = _build_preprocessor(X_norm, mode="default")
        clf = RandomForestClassifier(
            n_estimators=60, max_depth=8, min_samples_leaf=5, random_state=seed
        )
    elif variant == "algo_swap_logreg":
        prep = _build_preprocessor(X_norm, mode="alt_preprocessing")
        clf = LogisticRegression(max_iter=500, random_state=seed)
    elif variant == "preprocessing_change":
        prep = _build_preprocessor(X_norm, mode="alt_preprocessing")
        clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=seed)
    elif variant == "champ_imputation":
        prep = _build_preprocessor(X_norm, mode="champ_imputation")
        clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=seed)
    elif variant == "change_imputation":
        prep = _build_preprocessor(X_norm, mode="change_imputation")
        clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=seed)
    elif variant == "rescale_inputs":
        prep = _build_preprocessor(X_norm, mode="rescale_inputs")
        clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=seed)
    else:
        raise ValueError(f"Unknown model variant: {variant}")

    pipe = Pipeline([("preprocessor", prep), ("classifier", clf)])
    pipe.fit(X_norm, y)
    return pipe


class CohortCorruptedModel:
    """Wrapper simulating a serving/preprocessing bug that adds 2σ noise to numeric features for one cohort."""

    def __init__(
        self,
        base_model: Any,
        cohort_col: str,
        target_cohort: str,
        numeric_stds: dict[str, float],
        seed: int = 42,
    ) -> None:
        self.base_model = base_model
        self.cohort_col = cohort_col
        self.target_cohort = target_cohort
        self.numeric_stds = numeric_stds
        self.seed = seed

    def _corrupt(self, X: pd.DataFrame) -> pd.DataFrame:
        X_mod = X.copy()
        if self.cohort_col not in X_mod.columns:
            return X_mod
        mask = X_mod[self.cohort_col].astype(str).values == self.target_cohort
        n_m = int(np.sum(mask))
        if n_m == 0:
            return X_mod
        rng = np.random.RandomState(self.seed)
        for col, std in self.numeric_stds.items():
            if col in X_mod.columns and std > 0:
                X_mod[col] = X_mod[col].astype(float)
                vals = X_mod.loc[mask, col].values
                X_mod.loc[mask, col] = vals + rng.normal(0.0, 2.0 * std, size=n_m)
        return X_mod

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.base_model.predict(self._corrupt(X)), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.base_model.predict_proba(self._corrupt(X)), dtype=float)


def _run_all_frozen_checks(
    train_cand: pd.DataFrame,
    train_ref: pd.DataFrame,
    test_eval: pd.DataFrame,
    target_col: str,
    protected_col: str,
    slice_cols: list[str],
    champion: Any,
    candidate: Any,
    seed: int,
    run_budgets: bool = True,
) -> tuple[Any, list[CheckResult]]:
    """Run PredGate checks using 100% frozen default thresholds."""
    results: list[CheckResult] = [
        run_drift_check(train_cand, train_ref, target_col, DriftConfig(enabled=True)),
        run_leakage_check(
            train_cand, test_eval, target_col, LeakageConfig(enabled=True), reference_df=train_ref
        ),
        run_label_shift_check(
            train_cand,
            train_ref,
            target_col,
            LabelShiftConfig(enabled=True, by=slice_cols),
        ),
        run_slices_check(
            test_eval,
            target_col,
            champion,
            candidate,
            SliceConfig(enabled=True, by=slice_cols, n_bootstrap=200),
            seed=seed,
        ),
        run_fairness_check(
            test_eval,
            target_col,
            candidate,
            champion,
            FairnessConfig(enabled=True, protected=[protected_col]),
            seed=seed,
        ),
    ]
    if run_budgets:
        results.append(
            run_budgets_check(
                test_eval,
                target_col,
                candidate,
                champion,
                BudgetConfig(enabled=True),
                seed=seed,
            )
        )
    return compute_verdict(results), results


def _split_dataset(
    ds_name: str,
    full_df: pd.DataFrame | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str, str, list[str], str]:
    """Return (train_base, train_more_20pct, test_df, target_col, protected_col, minority_val, slice_cols, null_col)."""
    rng = np.random.RandomState(seed)
    if ds_name == "UCI Adult":
        assert full_df is not None
        idx = rng.choice(len(full_df), size=5100, replace=False)
        return (
            full_df.iloc[idx[:3000]].reset_index(drop=True),
            full_df.iloc[idx[:3600]].reset_index(drop=True),
            full_df.iloc[idx[3600:5100]].reset_index(drop=True),
            "target",
            "sex",
            "Female",
            ["sex", "race"],
            "hours-per-week",
        )
    if ds_name == "German Credit":
        assert full_df is not None
        idx = rng.permutation(len(full_df))
        return (
            full_df.iloc[idx[:550]].reset_index(drop=True),
            full_df.iloc[idx[:660]].reset_index(drop=True),
            full_df.iloc[idx[700:1000]].reset_index(drop=True),
            "target",
            "sex",
            "female",
            ["sex", "housing"],
            "credit_amount",
        )
    if ds_name == "Synthetic Credit":
        train_more, test_df = _generate_data(seed=seed, n=3600)
        train_more = _normalize_frame(train_more, "default")
        test_df = _normalize_frame(test_df, "default")
        return (
            train_more.iloc[:3000].reset_index(drop=True),
            train_more.reset_index(drop=True),
            test_df.reset_index(drop=True),
            "default",
            "gender",
            "female",
            ["region", "gender"],
            "income",
        )
    if ds_name == "Bank Marketing":
        assert full_df is not None
        idx = rng.choice(len(full_df), size=5100, replace=False)
        return (
            full_df.iloc[idx[:3000]].reset_index(drop=True),
            full_df.iloc[idx[:3600]].reset_index(drop=True),
            full_df.iloc[idx[3600:5100]].reset_index(drop=True),
            "target",
            "marital",
            "divorced",
            ["marital", "education"],
            "duration",
        )
    if ds_name == "COMPAS":
        assert full_df is not None
        idx = rng.permutation(len(full_df))
        return (
            full_df.iloc[idx[:3000]].reset_index(drop=True),
            full_df.iloc[idx[:3600]].reset_index(drop=True),
            full_df.iloc[idx[3600:5100]].reset_index(drop=True),
            "target",
            "race",
            "Caucasian",
            ["race", "sex"],
            "priors_count",
        )
    raise ValueError(f"Unknown dataset: {ds_name}")


# ── Suite 1: Synthetic 7-fault on fresh seeds ──


def _run_synth_7fault_trial(args: tuple[str, int]) -> dict[str, Any]:
    fault_type, seed = args
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
    else:
        candidate = _train_model(X_faulty, y_faulty, seed=cand_seed)

    test_eval = test_df.copy()
    if fault_type == "feature_leakage":
        test_rng = np.random.RandomState(seed + 1000)
        test_eval["leaked"] = test_eval["default"] * 0.9 + test_rng.normal(0, 0.1, len(test_eval))

    X_test = test_eval.drop(columns=["default"])
    y_test = test_eval["default"]
    champ_preds = champion.predict(X_test)
    cand_preds = candidate.predict(X_test)
    champ_acc = float(accuracy_score(y_test, champ_preds))
    cand_acc = float(accuracy_score(y_test, cand_preds))

    verdict, checks = _run_all_frozen_checks(
        faulty_train,
        clean_train,
        test_eval,
        "default",
        "gender",
        ["region", "gender"],
        champion,
        candidate,
        seed=seed,
        run_budgets=fault_type in ("latency_regression", "noop"),
    )
    evidently_det = _run_evidently_eval(
        clean_train, faulty_train, test_eval, "default", champ_preds, cand_preds
    )

    return {
        "fault_type": fault_type,
        "seed": seed,
        "predgate_fail": verdict.status == "fail",
        "predgate_detected": verdict.status in ("warn", "fail"),
        "predgate_status": verdict.status,
        "check_statuses": {c.name: c.status for c in checks},
        "baseline_detected": abs(champ_acc - cand_acc) > 0.01,
        "per_group_acc_detected": per_group_accuracy_drop_baseline(
            y_test.values, champ_preds, cand_preds, test_eval, ["region", "gender"]
        ),
        "fixed_dp_detected": fixed_dp_limit_baseline(
            cand_preds, test_eval["gender"].values, dp_limit=0.10
        ),
        "evidently_detected": evidently_det,
        "champion_acc": round(champ_acc, 4),
        "candidate_acc": round(cand_acc, 4),
    }


# ── Unified per-(dataset, seed) worker running ALL 9 benign changes + ALL 7 faults ──


def _run_dataset_seed(args: tuple[str, pd.DataFrame | None, int]) -> dict[str, Any]:
    """Reuse one champion per (dataset, seed) across all 9 benign changes and all 7 faults."""
    ds_name, full_df, seed = args
    (
        train_base,
        train_more,
        test_base,
        target_col,
        protected_col,
        minority_value,
        slice_cols,
        null_col,
    ) = _split_dataset(ds_name, full_df, seed)

    X_base = train_base.drop(columns=[target_col])
    y_base = train_base[target_col]
    champion = _train_pipeline(X_base, y_base, seed=seed, variant="champion_gbdt")
    champ_preds = champion.predict(test_base.drop(columns=[target_col]))
    y_test = test_base[target_col].values
    champ_acc = float(accuracy_score(y_test, champ_preds))
    champ_dp = float(demographic_parity_diff(champ_preds, test_base[protected_col].values))

    # For `drop_irrelevant_feature`: train a paired champion that has a pure noise column
    nrng_drop = np.random.RandomState(seed + 888)
    train_with_noise = train_base.copy()
    test_with_noise = test_base.copy()
    train_with_noise["irrelevant_noise_col"] = nrng_drop.normal(
        0.0, 1.0, size=len(train_with_noise)
    )
    test_with_noise["irrelevant_noise_col"] = nrng_drop.normal(0.0, 1.0, size=len(test_with_noise))
    champ_with_noise = _train_pipeline(
        train_with_noise.drop(columns=[target_col]),
        train_with_noise[target_col],
        seed=seed,
        variant="champion_gbdt",
    )
    champ_noise_preds = champ_with_noise.predict(test_with_noise.drop(columns=[target_col]))

    # For `change_imputation`: introduce 5% MCAR NaNs on non-slice features so median vs mean+indicator genuinely differ
    mrng_imp = np.random.RandomState(seed + 999)
    train_miss = train_base.copy()
    test_miss = test_base.copy()
    protected_set = {target_col, protected_col, *slice_cols}
    for col in train_miss.columns:
        if col in protected_set:
            continue
        if pd.api.types.is_numeric_dtype(train_miss[col]):
            train_miss[col] = train_miss[col].astype(float)
            test_miss[col] = test_miss[col].astype(float)
            m_tr = mrng_imp.rand(len(train_miss)) < 0.05
            m_te = mrng_imp.rand(len(test_miss)) < 0.05
            train_miss.loc[m_tr, col] = np.nan
            test_miss.loc[m_te, col] = np.nan
        else:
            train_miss[col] = train_miss[col].astype(object)
            test_miss[col] = test_miss[col].astype(object)
            m_tr = mrng_imp.rand(len(train_miss)) < 0.05
            m_te = mrng_imp.rand(len(test_miss)) < 0.05
            train_miss.loc[m_tr, col] = None
            test_miss.loc[m_te, col] = None
    champ_impute = _train_pipeline(
        train_miss.drop(columns=[target_col]),
        train_miss[target_col],
        seed=seed,
        variant="champ_imputation",
    )
    champ_impute_preds = champ_impute.predict(test_miss.drop(columns=[target_col]))

    # ── Part A: 9 Benign Changes ──
    benign_recs: list[dict[str, Any]] = []
    for change in BENIGN_CHANGES:
        ref_train_use = train_base
        champ_use = champion
        champ_preds_use = champ_preds
        train_cand = train_base.copy()
        test_eval = test_base.copy()

        if change in (
            "hyperparam_change",
            "algo_swap_rf",
            "algo_swap_logreg",
            "preprocessing_change",
            "rescale_inputs",
        ):
            candidate = _train_pipeline(
                train_cand.drop(columns=[target_col]),
                train_cand[target_col],
                seed=seed + 100,
                variant=change,
            )
        elif change == "change_imputation":
            ref_train_use = train_miss
            champ_use = champ_impute
            champ_preds_use = champ_impute_preds
            train_cand = train_miss.copy()
            test_eval = test_miss.copy()
            candidate = _train_pipeline(
                train_cand.drop(columns=[target_col]),
                train_cand[target_col],
                seed=seed + 100,
                variant="change_imputation",
            )
        elif change == "more_data_20pct":
            train_cand = train_more.copy()
            candidate = _train_pipeline(
                train_cand.drop(columns=[target_col]),
                train_cand[target_col],
                seed=seed + 100,
                variant="champion_gbdt",
            )
        elif change == "irrelevant_feature":
            nrng = np.random.RandomState(seed + 777)
            train_cand["benign_noise_feat"] = nrng.normal(0.0, 1.0, size=len(train_cand))
            test_eval["benign_noise_feat"] = nrng.normal(0.0, 1.0, size=len(test_eval))
            candidate = _train_pipeline(
                train_cand.drop(columns=[target_col]),
                train_cand[target_col],
                seed=seed + 100,
                variant="champion_gbdt",
            )
        elif change == "drop_irrelevant_feature":
            ref_train_use = train_with_noise
            champ_use = champ_with_noise
            champ_preds_use = champ_noise_preds
            train_cand = train_base.copy()
            test_eval = test_with_noise.copy()
            # Candidate trained WITHOUT irrelevant_noise_col (ColumnTransformer drops it at predict time)
            candidate = _train_pipeline(
                train_cand.drop(columns=[target_col]),
                train_cand[target_col],
                seed=seed + 100,
                variant="champion_gbdt",
            )
        else:
            raise ValueError(f"Unknown benign change: {change}")

        cand_preds = candidate.predict(test_eval.drop(columns=[target_col]))
        verdict, check_results = _run_all_frozen_checks(
            train_cand,
            ref_train_use,
            test_eval,
            target_col,
            protected_col,
            slice_cols,
            champ_use,
            candidate,
            seed=seed,
            run_budgets=True,
        )
        statuses = {cr.name: cr.status for cr in check_results}
        fair_cr = next(cr for cr in check_results if cr.name == "fairness")

        q_statuses = [statuses[c] for c in QUALITY_CHECKS]
        q_fail = "fail" in q_statuses
        q_warn_only = (not q_fail) and ("warn" in q_statuses)
        lat_status = statuses.get("budgets", "pass")

        evidently_det = _run_evidently_eval(
            ref_train_use, train_cand, test_eval, target_col, champ_preds_use, cand_preds
        )

        benign_recs.append(
            {
                "dataset": ds_name,
                "change": change,
                "seed": seed,
                "status": verdict.status,
                "is_fail": verdict.status == "fail",
                "is_warn_only": verdict.status == "warn",
                "is_flagged": verdict.status in ("warn", "fail"),
                "quality_fail": q_fail,
                "quality_warn_only": q_warn_only,
                "quality_flagged": q_fail or q_warn_only,
                "latency_fail": lat_status == "fail",
                "latency_warn": lat_status == "warn",
                "check_statuses": statuses,
                "fairness_fail_metrics": fair_cr.details.get("triggered_fail_metrics", []),
                "fairness_warn_metrics": fair_cr.details.get("triggered_warn_metrics", []),
                "per_group_acc_detected": per_group_accuracy_drop_baseline(
                    y_test, champ_preds_use, cand_preds, test_eval, slice_cols
                ),
                "fixed_dp_detected": fixed_dp_limit_baseline(
                    cand_preds, test_eval[protected_col].values, dp_limit=0.10
                ),
                "evidently_detected": evidently_det,
            }
        )

    # ── Part B: All 7 Fault Conditions on This Dataset ──
    num_cols = X_base.select_dtypes(include=[np.number]).columns.tolist()
    numeric_stds = {c: float(X_base[c].std()) for c in num_cols}

    fault_recs: list[dict[str, Any]] = []
    pub_fair_recs: list[dict[str, Any]] = []

    for fault, _ in ALL_DATASET_FAULTS:
        frng = np.random.RandomState(seed)
        train_mod = train_base.copy()
        test_mod = test_base.copy()

        if fault == "silent_null_imputation":
            corrupt_idx = train_mod.sample(frac=0.40, random_state=seed).index
            train_mod.loc[corrupt_idx, null_col] = 0.0
            candidate = _train_pipeline(
                train_mod.drop(columns=[target_col]),
                train_mod[target_col],
                seed=seed,
                variant="champion_gbdt",
            )
        elif fault == "cohort_feature_corruption":
            base_cand = _train_pipeline(
                train_mod.drop(columns=[target_col]),
                train_mod[target_col],
                seed=seed + 50,
                variant="champion_gbdt",
            )
            candidate = CohortCorruptedModel(
                base_cand,
                cohort_col=protected_col,
                target_cohort=minority_value,
                numeric_stds=numeric_stds,
                seed=seed,
            )
        elif fault == "fairness_threshold_shift":
            candidate = GroupThresholdModel(
                champion,
                protected_col,
                minority_value,
                group_threshold=0.25,
                default_threshold=0.50,
            )
        elif fault == "fairness_threshold_shift_hi":
            candidate = GroupThresholdModel(
                champion,
                protected_col,
                minority_value,
                group_threshold=0.75,
                default_threshold=0.50,
            )
        elif fault == "fairness_proxy_feature":
            is_min_tr = (train_mod[protected_col].astype(str) == minority_value).astype(float)
            train_mod["proxy_score"] = is_min_tr * 0.85 + frng.normal(0.0, 0.25, len(train_mod))
            high_proxy_neg = train_mod[
                (train_mod["proxy_score"] > 0.6) & (train_mod[target_col] == 0)
            ]
            if len(high_proxy_neg) > 0:
                flip_idx = high_proxy_neg.sample(frac=0.35, random_state=seed).index
                train_mod.loc[flip_idx, target_col] = 1
            candidate = _train_pipeline(
                train_mod.drop(columns=[target_col, protected_col]),
                train_mod[target_col],
                seed=seed,
                variant="champion_gbdt",
            )
            test_rng = np.random.RandomState(seed + 1000)
            is_min_te = (test_mod[protected_col].astype(str) == minority_value).astype(float)
            test_mod["proxy_score"] = is_min_te * 0.85 + test_rng.normal(0.0, 0.25, len(test_mod))
        elif fault == "fairness_regression":
            min_neg = train_mod[
                (train_mod[protected_col].astype(str) == minority_value)
                & (train_mod[target_col] == 0)
            ]
            if len(min_neg) > 0:
                flip_idx = min_neg.sample(frac=0.30, random_state=seed).index
                train_mod.loc[flip_idx, target_col] = 1
            candidate = _train_pipeline(
                train_mod.drop(columns=[target_col]),
                train_mod[target_col],
                seed=seed,
                variant="champion_gbdt",
            )
        elif fault == "noop":
            candidate = _train_pipeline(X_base, y_base, seed=seed + 500, variant="champion_gbdt")
        else:
            raise ValueError(f"Unknown fault: {fault}")

        cand_preds = candidate.predict(test_mod.drop(columns=[target_col]))
        cand_acc = float(accuracy_score(y_test, cand_preds))
        cand_dp = float(demographic_parity_diff(cand_preds, test_mod[protected_col].values))

        # Run full quality gate checks (skip redundant budgets on non-latency faults to save CPU; budgets tested on all 9 benign + latency_regression + noop)
        verdict, check_results = _run_all_frozen_checks(
            train_mod,
            train_base,
            test_mod,
            target_col,
            protected_col,
            slice_cols,
            champion,
            candidate,
            seed=seed,
            run_budgets=(fault == "noop"),
        )
        fair_cr = next(c for c in check_results if c.name == "fairness")
        grp = fair_cr.details.get("groups", [{}])[0]
        evidently_det = _run_evidently_eval(
            train_base, train_mod, test_mod, target_col, champ_preds, cand_preds
        )

        bl_det = abs(champ_acc - cand_acc) > 0.01
        pga_det = per_group_accuracy_drop_baseline(
            y_test, champ_preds, cand_preds, test_mod, slice_cols
        )
        dp_det = fixed_dp_limit_baseline(cand_preds, test_mod[protected_col].values, dp_limit=0.10)

        fault_recs.append(
            {
                "dataset": ds_name,
                "fault": fault,
                "seed": seed,
                "status": verdict.status,
                "is_fail": verdict.status == "fail",
                "is_warn": verdict.status == "warn",
                "detected": verdict.status in ("warn", "fail"),
                "check_statuses": {c.name: c.status for c in check_results},
                "baseline_detected": bl_det,
                "per_group_acc_detected": pga_det,
                "fixed_dp_detected": dp_det,
                "evidently_detected": evidently_det,
                "champ_dp": round(champ_dp, 4),
                "cand_dp": round(cand_dp, 4),
                "min_group_n": int(grp.get("min_group_n", 0)),
                "mde_dp": float(grp.get("min_detectable_gap_dp", 0.0)),
                "mde_eo": float(grp.get("min_detectable_gap_eo", 0.0)),
                "power_note_emitted": bool(fair_cr.details.get("power_notes")),
            }
        )

        if fault in {f for f, _ in PUBLIC_FAULTS}:
            pub_fair_recs.append(
                {
                    "dataset": ds_name,
                    "fault": fault,
                    "seed": seed,
                    "status": fair_cr.status,
                    "detected": fair_cr.status in ("warn", "fail"),
                    "is_fail": fair_cr.status == "fail",
                    "is_warn": fair_cr.status == "warn",
                    "baseline_detected": bl_det,
                    "per_group_acc_detected": per_group_accuracy_drop_baseline(
                        y_test, champ_preds, cand_preds, test_mod, [protected_col]
                    ),
                    "fixed_dp_detected": dp_det,
                    "champ_dp": round(champ_dp, 4),
                    "cand_dp": round(cand_dp, 4),
                    "min_group_n": int(grp.get("min_group_n", 0)),
                    "mde_dp": float(grp.get("min_detectable_gap_dp", 0.0)),
                    "mde_eo": float(grp.get("min_detectable_gap_eo", 0.0)),
                    "power_note_emitted": bool(fair_cr.details.get("power_notes")),
                }
            )

    return {
        "benign": benign_recs,
        "faults": fault_recs,
        "pub_fairness": pub_fair_recs,
    }


def _agg_rate(k: int, n: int) -> tuple[float, list[float]]:
    rate = round(k / n, 3) if n else 0.0
    ci = wilson_ci(k, n)
    return rate, [round(ci[0], 3), round(ci[1], 3)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fresh-seed PredGate evaluation across all 5 datasets, 9 benign changes, and all fault families."
    )
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=9000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--evidently-python",
        type=str,
        default="/tmp/evidently-venv/bin/python",
        help="Path to Python interpreter with evidently==0.7.23 (pass '' to disable)",
    )
    parser.add_argument("--output-json", type=Path, default=Path("docs/benchmark_results.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/benchmark.md"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("release_artifacts"))
    args = parser.parse_args()

    t0 = time.time()
    commit_hash = _get_git_commit()
    seeds = [args.start_seed + i for i in range(args.n_seeds)]
    ev_py = (
        args.evidently_python
        if (args.evidently_python and Path(args.evidently_python).exists())
        else None
    )

    comparator_config: dict[str, Any] = {}
    if ev_py:
        comp_script = Path(__file__).resolve().parent / "evidently_comparator.py"
        v_out = subprocess.check_output([ev_py, str(comp_script), "--version"], text=True).strip()
        v_meta = json.loads(v_out)
        comparator_config = {
            "evidently_version": v_meta["evidently_version"],
            "pandas_version": v_meta["pandas_version"],
            "presets": ["DataDriftPreset(default)", "ClassificationPreset(default)"],
            "include_tests": True,
            "decision_rule": "Flagged if any default Evidently test returns status FAIL",
            "description": (
                "Runs Report([DataDriftPreset()], include_tests=True) on candidate vs reference training data "
                "and Report([ClassificationPreset()], include_tests=True) on candidate vs champion test predictions "
                "with 100% Evidently default thresholds; flagged if any test returns FAIL."
            ),
        }
        print(f"Evidently comparator enabled: {v_meta}")
    else:
        print("Evidently comparator disabled.")

    frozen_cfg = FairnessConfig(enabled=True, protected=["sex"])
    frozen_summary = {
        "enforce_absolute": frozen_cfg.enforce_absolute,
        "max_gap_increase": frozen_cfg.max_gap_increase,
        "fail_gap_increase": frozen_cfg.fail_gap_increase,
        "selection_rate_shift_warn": frozen_cfg.selection_rate_shift_warn,
        "confidence": frozen_cfg.confidence,
        "n_bootstrap": frozen_cfg.n_bootstrap,
        "selection_rate_n_bootstrap": frozen_cfg.selection_rate_n_bootstrap,
    }

    print("[1/3] Loading all 4 OpenML datasets (Adult, German Credit, Bank Marketing, COMPAS)...")
    ds_frames: dict[str, pd.DataFrame | None] = {
        "UCI Adult": _load_adult_frame(),
        "German Credit": _load_german_frame(),
        "Synthetic Credit": None,
        "Bank Marketing": _load_bank_marketing_frame(),
        "COMPAS": _load_compas_frame(),
    }

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(ev_py,),
    ) as pool:
        print(f"[2/3] Running Synthetic 7-fault suite (fresh seeds {seeds[0]}..{seeds[-1]})...")
        synth_records = list(
            pool.map(_run_synth_7fault_trial, [(f, s) for f in SYNTHETIC_FAULTS for s in seeds])
        )

        print(
            f"[3/3] Running ALL 9 benign changes + ALL 7 fault families across ALL 5 datasets "
            f"(fresh seeds {seeds[0]}..{seeds[-1]})..."
        )
        ds_tasks: list[tuple[str, pd.DataFrame | None, int]] = [
            (ds_name, ds_frames[ds_name], s) for ds_name in ALL_DATASETS for s in seeds
        ]
        benign_records: list[dict[str, Any]] = []
        fault_records: list[dict[str, Any]] = []
        pub_fair_records: list[dict[str, Any]] = []
        for idx_done, out in enumerate(pool.map(_run_dataset_seed, ds_tasks), start=1):
            benign_records.extend(out["benign"])
            fault_records.extend(out["faults"])
            pub_fair_records.extend(out["pub_fairness"])
            if idx_done % 15 == 0 or idx_done == len(ds_tasks):
                print(
                    f"  progress: {idx_done}/{len(ds_tasks)} dataset-seeds ({time.time() - t0:.1f}s)",
                    flush=True,
                )

    # ── Aggregate Suite 1: Synthetic 7-Fault ──
    synth_summary: dict[str, dict[str, Any]] = {}
    for fault in SYNTHETIC_FAULTS:
        rows = [r for r in synth_records if r["fault_type"] == fault]
        n = len(rows)
        mg_fail = sum(1 for r in rows if r["predgate_fail"])
        mg_det = sum(1 for r in rows if r["predgate_detected"])
        bl_det = sum(1 for r in rows if r["baseline_detected"])
        pg_acc_det = sum(1 for r in rows if r["per_group_acc_detected"])
        dp_det = sum(1 for r in rows if r["fixed_dp_detected"])

        fail_r, fail_ci = _agg_rate(mg_fail, n)
        det_r, det_ci = _agg_rate(mg_det, n)
        bl_r, bl_ci = _agg_rate(bl_det, n)
        pga_r, pga_ci = _agg_rate(pg_acc_det, n)
        dp_r, dp_ci = _agg_rate(dp_det, n)
        entry: dict[str, Any] = {
            "description": FAULT_DESCRIPTIONS[fault],
            "n_trials": n,
            "mg_fail": mg_fail,
            "mg_warn": mg_det - mg_fail,
            "mg_detected": mg_det,
            "bl_detected": bl_det,
            "per_group_acc_detected": pg_acc_det,
            "fixed_dp_detected": dp_det,
            "predgate_fail_rate": fail_r,
            "predgate_fail_ci_95": fail_ci,
            "predgate_detection_rate": det_r,
            "predgate_ci_95": det_ci,
            "baseline_detection_rate": bl_r,
            "baseline_ci_95": bl_ci,
            "per_group_acc_rate": pga_r,
            "per_group_acc_ci_95": pga_ci,
            "fixed_dp_rate": dp_r,
            "fixed_dp_ci_95": dp_ci,
            "ci_95_overlap": (det_ci[0] <= bl_ci[1]) and (bl_ci[0] <= det_ci[1]),
            "predgate_wins": (det_ci[0] > bl_ci[1]) if fault != "noop" else (det_ci[1] < bl_ci[0]),
        }
        if ev_py is not None:
            ev_det = sum(1 for r in rows if r["evidently_detected"])
            ev_r, ev_ci = _agg_rate(ev_det, n)
            entry["evidently_detected"] = ev_det
            entry["evidently_rate"] = ev_r
            entry["evidently_ci_95"] = ev_ci
        synth_summary[fault] = entry
        print(
            f"  SYNTH {fault:20s}: FAIL {mg_fail:2d}/{n} | Flag {mg_det:2d}/{n} | "
            f"Acc {bl_det:2d}/{n} | GrpAcc {pg_acc_det:2d}/{n} | FixDP {dp_det:2d}/{n}"
            + (f" | Evid {entry['evidently_detected']:2d}/{n}" if ev_py else "")
        )

    # ── Aggregate Suite 2: Benign-Change Control (Quality vs Latency Separated) ──
    benign_rows: list[dict[str, Any]] = []
    for ds_name in ALL_DATASETS:
        for chg in BENIGN_CHANGES:
            sub = [r for r in benign_records if r["dataset"] == ds_name and r["change"] == chg]
            n = len(sub)
            f_cnt = sum(1 for r in sub if r["is_fail"])
            w_cnt = sum(1 for r in sub if r["is_warn_only"])
            fl_cnt = sum(1 for r in sub if r["is_flagged"])
            qf_cnt = sum(1 for r in sub if r["quality_fail"])
            qw_cnt = sum(1 for r in sub if r["quality_warn_only"])
            lf_cnt = sum(1 for r in sub if r["latency_fail"])
            lw_cnt = sum(1 for r in sub if r["latency_warn"])
            pga_d = sum(1 for r in sub if r["per_group_acc_detected"])
            dp_d = sum(1 for r in sub if r["fixed_dp_detected"])

            f_r, f_ci = _agg_rate(f_cnt, n)
            w_r, w_ci = _agg_rate(w_cnt, n)
            fl_r, fl_ci = _agg_rate(fl_cnt, n)
            qf_r, qf_ci = _agg_rate(qf_cnt, n)
            qw_r, qw_ci = _agg_rate(qw_cnt, n)
            lf_r, lf_ci = _agg_rate(lf_cnt, n)
            lw_r, lw_ci = _agg_rate(lw_cnt, n)
            pga_r, pga_ci = _agg_rate(pga_d, n)
            dp_r, dp_ci = _agg_rate(dp_d, n)

            b_entry: dict[str, Any] = {
                "dataset": ds_name,
                "change": chg,
                "n_trials": n,
                "fail_count": f_cnt,
                "fail_rate": f_r,
                "fail_ci_95": f_ci,
                "warn_only_count": w_cnt,
                "warn_only_rate": w_r,
                "warn_only_ci_95": w_ci,
                "flagged_count": fl_cnt,
                "flagged_rate": fl_r,
                "flagged_ci_95": fl_ci,
                "quality_fail_count": qf_cnt,
                "quality_fail_rate": qf_r,
                "quality_fail_ci_95": qf_ci,
                "quality_warn_only_count": qw_cnt,
                "quality_warn_only_rate": qw_r,
                "quality_warn_only_ci_95": qw_ci,
                "latency_fail_count": lf_cnt,
                "latency_fail_rate": lf_r,
                "latency_fail_ci_95": lf_ci,
                "latency_warn_count": lw_cnt,
                "latency_warn_rate": lw_r,
                "latency_warn_ci_95": lw_ci,
                "per_group_acc_detected": pga_d,
                "per_group_acc_rate": pga_r,
                "per_group_acc_ci_95": pga_ci,
                "fixed_dp_detected": dp_d,
                "fixed_dp_rate": dp_r,
                "fixed_dp_ci_95": dp_ci,
            }
            if ev_py is not None:
                ev_d = sum(1 for r in sub if r["evidently_detected"])
                ev_r, ev_ci = _agg_rate(ev_d, n)
                b_entry["evidently_detected"] = ev_d
                b_entry["evidently_rate"] = ev_r
                b_entry["evidently_ci_95"] = ev_ci
            benign_rows.append(b_entry)
            print(
                f"  BENIGN {ds_name:16s} | {chg:23s}: Q_FAIL {qf_cnt:2d}/{n} | Q_WARN {qw_cnt:2d}/{n} | "
                f"LAT_FAIL {lf_cnt:2d}/{n} | GrpAcc {pga_d:2d}/{n} | FixDP {dp_d:2d}/{n}"
                + (f" | Evid {b_entry['evidently_detected']:2d}/{n}" if ev_py else "")
            )

    n_b_tot = len(benign_records)
    ov_f = sum(1 for r in benign_records if r["is_fail"])
    ov_w = sum(1 for r in benign_records if r["is_warn_only"])
    ov_fl = sum(1 for r in benign_records if r["is_flagged"])
    ov_qf = sum(1 for r in benign_records if r["quality_fail"])
    ov_qw = sum(1 for r in benign_records if r["quality_warn_only"])
    ov_lf = sum(1 for r in benign_records if r["latency_fail"])
    ov_lw = sum(1 for r in benign_records if r["latency_warn"])

    ov_f_r, ov_f_ci = _agg_rate(ov_f, n_b_tot)
    ov_w_r, ov_w_ci = _agg_rate(ov_w, n_b_tot)
    ov_fl_r, ov_fl_ci = _agg_rate(ov_fl, n_b_tot)
    ov_qf_r, ov_qf_ci = _agg_rate(ov_qf, n_b_tot)
    ov_qw_r, ov_qw_ci = _agg_rate(ov_qw, n_b_tot)
    ov_lf_r, ov_lf_ci = _agg_rate(ov_lf, n_b_tot)
    ov_lw_r, ov_lw_ci = _agg_rate(ov_lw, n_b_tot)

    per_check_benign: list[dict[str, Any]] = []
    for chk in ALL_CHECKS:
        cf = sum(1 for r in benign_records if r["check_statuses"].get(chk) == "fail")
        cw = sum(1 for r in benign_records if r["check_statuses"].get(chk) == "warn")
        cfl = cf + cw
        cf_r, cf_ci = _agg_rate(cf, n_b_tot)
        cw_r, cw_ci = _agg_rate(cw, n_b_tot)
        cfl_r, cfl_ci = _agg_rate(cfl, n_b_tot)
        per_check_benign.append(
            {
                "check": chk,
                "n_trials": n_b_tot,
                "fail_count": cf,
                "fail_rate": cf_r,
                "fail_ci_95": cf_ci,
                "warn_count": cw,
                "warn_rate": cw_r,
                "warn_ci_95": cw_ci,
                "flagged_count": cfl,
                "flagged_rate": cfl_r,
                "flagged_ci_95": cfl_ci,
            }
        )
        print(
            f"  BENIGN CHECK {chk:12s}: FAIL {cf:4d}/{n_b_tot} ({cf_r:.1%}) | "
            f"WARN {cw:4d}/{n_b_tot} ({cw_r:.1%})"
        )

    algo_swaps = [r for r in benign_records if r["change"] in ("algo_swap_rf", "algo_swap_logreg")]
    algo_fair_fail_ctr = Counter(
        m
        for r in algo_swaps
        if r["check_statuses"].get("fairness") == "fail"
        for m in r["fairness_fail_metrics"]
    )
    algo_fair_warn_ctr = Counter(
        m
        for r in algo_swaps
        if r["check_statuses"].get("fairness") == "warn"
        for m in r["fairness_warn_metrics"]
    )
    algo_swap_fairness_metrics = {
        "n_trials": len(algo_swaps),
        "n_fail": sum(1 for r in algo_swaps if r["check_statuses"].get("fairness") == "fail"),
        "n_warn": sum(1 for r in algo_swaps if r["check_statuses"].get("fairness") == "warn"),
        "fail_metrics": dict(algo_fair_fail_ctr),
        "warn_metrics": dict(algo_fair_warn_ctr),
    }

    # ── Aggregate Suite 3: All 5 Datasets x All 7 Faults ──
    all_fault_rows: list[dict[str, Any]] = []
    for ds_name in ALL_DATASETS:
        for fault, flt_label in ALL_DATASET_FAULTS:
            sub = [r for r in fault_records if r["dataset"] == ds_name and r["fault"] == fault]
            n = len(sub)
            det_f = sum(1 for r in sub if r["is_fail"])
            det_w = sum(1 for r in sub if r["is_warn"])
            det = sum(1 for r in sub if r["detected"])
            bl_d = sum(1 for r in sub if r["baseline_detected"])
            pga_d = sum(1 for r in sub if r["per_group_acc_detected"])
            dp_d = sum(1 for r in sub if r["fixed_dp_detected"])

            fail_r, fail_ci = _agg_rate(det_f, n)
            det_r, det_ci = _agg_rate(det, n)
            bl_r, bl_ci = _agg_rate(bl_d, n)
            pga_r, pga_ci = _agg_rate(pga_d, n)
            dp_r, dp_ci = _agg_rate(dp_d, n)
            h_entry: dict[str, Any] = {
                "dataset": ds_name,
                "fault": fault,
                "fault_label": flt_label,
                "n_trials": n,
                "detected_fail": det_f,
                "detected_warn": det_w,
                "detected_total": det,
                "fail_rate": fail_r,
                "fail_ci_95": fail_ci,
                "detection_rate": det_r,
                "ci_95": det_ci,
                "baseline_detected": bl_d,
                "baseline_rate": bl_r,
                "baseline_ci_95": bl_ci,
                "per_group_acc_detected": pga_d,
                "per_group_acc_rate": pga_r,
                "per_group_acc_ci_95": pga_ci,
                "fixed_dp_detected": dp_d,
                "fixed_dp_rate": dp_r,
                "fixed_dp_ci_95": dp_ci,
                "mean_champ_dp": round(float(np.mean([r["champ_dp"] for r in sub])), 3),
                "mean_cand_dp": round(float(np.mean([r["cand_dp"] for r in sub])), 3),
                "mean_min_n": round(float(np.mean([r["min_group_n"] for r in sub])), 1),
                "mean_mde_dp": round(float(np.mean([r["mde_dp"] for r in sub])), 3),
                "mean_mde_eo": round(float(np.mean([r["mde_eo"] for r in sub])), 3),
                "power_note_count": sum(1 for r in sub if r["power_note_emitted"]),
            }
            if ev_py is not None:
                ev_d = sum(1 for r in sub if r["evidently_detected"])
                ev_r, ev_ci = _agg_rate(ev_d, n)
                h_entry["evidently_detected"] = ev_d
                h_entry["evidently_rate"] = ev_r
                h_entry["evidently_ci_95"] = ev_ci
            all_fault_rows.append(h_entry)
            print(
                f"  FAULT  {ds_name:16s} | {fault:27s}: FAIL {det_f:2d}/{n} | Flag {det:2d}/{n} | "
                f"Acc {bl_d:2d}/{n} | GrpAcc {pga_d:2d}/{n} | FixDP {dp_d:2d}/{n}"
                + (f" | Evid {h_entry['evidently_detected']:2d}/{n}" if ev_py else "")
            )

    # ── Aggregate Suite 4: Standalone Fairness Check Across All 5 Datasets ──
    pub_rows: list[dict[str, Any]] = []
    for ds_name in ALL_DATASETS:
        for fault, flt_label in PUBLIC_FAULTS:
            sub = [r for r in pub_fair_records if r["dataset"] == ds_name and r["fault"] == fault]
            n = len(sub)
            det = sum(1 for r in sub if r["detected"])
            det_f = sum(1 for r in sub if r["is_fail"])
            det_w = sum(1 for r in sub if r["is_warn"])
            bl_d = sum(1 for r in sub if r["baseline_detected"])
            pga_d = sum(1 for r in sub if r["per_group_acc_detected"])
            dp_d = sum(1 for r in sub if r["fixed_dp_detected"])

            fail_r, fail_ci = _agg_rate(det_f, n)
            det_r, det_ci = _agg_rate(det, n)
            bl_r, bl_ci = _agg_rate(bl_d, n)
            pga_r, pga_ci = _agg_rate(pga_d, n)
            dp_r, dp_ci = _agg_rate(dp_d, n)
            pub_rows.append(
                {
                    "dataset": ds_name,
                    "fault": fault,
                    "fault_label": flt_label,
                    "n_trials": n,
                    "detected_fail": det_f,
                    "detected_warn": det_w,
                    "detected_total": det,
                    "fail_rate": fail_r,
                    "fail_ci_95": fail_ci,
                    "detection_rate": det_r,
                    "ci_95": det_ci,
                    "baseline_detected": bl_d,
                    "baseline_rate": bl_r,
                    "baseline_ci_95": bl_ci,
                    "per_group_acc_detected": pga_d,
                    "per_group_acc_rate": pga_r,
                    "per_group_acc_ci_95": pga_ci,
                    "fixed_dp_detected": dp_d,
                    "fixed_dp_rate": dp_r,
                    "fixed_dp_ci_95": dp_ci,
                    "mean_champ_dp": round(float(np.mean([r["champ_dp"] for r in sub])), 3),
                    "mean_cand_dp": round(float(np.mean([r["cand_dp"] for r in sub])), 3),
                    "mean_min_n": round(float(np.mean([r["min_group_n"] for r in sub])), 1),
                    "mean_mde_dp": round(float(np.mean([r["mde_dp"] for r in sub])), 3),
                    "mean_mde_eo": round(float(np.mean([r["mde_eo"] for r in sub])), 3),
                    "power_note_count": sum(1 for r in sub if r["power_note_emitted"]),
                }
            )

    # ── Write aggregate-only repo JSON + per-seed release artifact ──
    cmd_str = (
        f"python scripts/bench_public.py --n-seeds {args.n_seeds} --start-seed {args.start_seed}"
    )
    per_seed_path = args.artifact_dir / f"benchmark_per_seed_{commit_hash}.json"
    per_seed_path.parent.mkdir(parents=True, exist_ok=True)
    per_seed_path.write_text(
        json.dumps(
            {
                "commit_hash": commit_hash,
                "command": cmd_str,
                "seeds": seeds,
                "synthetic_details": synth_records,
                "benign_change_details": benign_records,
                "all_dataset_fault_details": fault_records,
                "public_fairness_details": pub_fair_records,
            },
            indent=2,
        )
        + "\n"
    )

    aggregate_payload: dict[str, Any] = {
        "command": cmd_str,
        "commit_hash": commit_hash,
        "n_seeds": args.n_seeds,
        "seed_range": f"{seeds[0]}..{seeds[-1]}",
        "seeds": seeds,
        "per_seed_artifact": str(per_seed_path),
        "comparator_config": comparator_config,
        "summary": synth_summary,
        "benign_change_control": {
            "commit_hash": commit_hash,
            "n_seeds": args.n_seeds,
            "seed_range": f"{seeds[0]}..{seeds[-1]}",
            "seeds": seeds,
            "changes": BENIGN_CHANGES,
            "checks": ALL_CHECKS,
            "overall": {
                "n_trials": n_b_tot,
                "fail_count": ov_f,
                "fail_rate": ov_f_r,
                "fail_ci_95": ov_f_ci,
                "warn_only_count": ov_w,
                "warn_only_rate": ov_w_r,
                "warn_only_ci_95": ov_w_ci,
                "flagged_count": ov_fl,
                "flagged_rate": ov_fl_r,
                "flagged_ci_95": ov_fl_ci,
                "quality_fail_count": ov_qf,
                "quality_fail_rate": ov_qf_r,
                "quality_fail_ci_95": ov_qf_ci,
                "quality_warn_only_count": ov_qw,
                "quality_warn_only_rate": ov_qw_r,
                "quality_warn_only_ci_95": ov_qw_ci,
                "latency_fail_count": ov_lf,
                "latency_fail_rate": ov_lf_r,
                "latency_fail_ci_95": ov_lf_ci,
                "latency_warn_count": ov_lw,
                "latency_warn_rate": ov_lw_r,
                "latency_warn_ci_95": ov_lw_ci,
            },
            "rows": benign_rows,
            "per_check": per_check_benign,
            "algo_swap_fairness_metrics": algo_swap_fairness_metrics,
        },
        "held_out_validation": {
            "commit_hash": commit_hash,
            "n_seeds": args.n_seeds,
            "seed_range": f"{seeds[0]}..{seeds[-1]}",
            "seeds": seeds,
            "datasets": ALL_DATASETS,
            "rows": all_fault_rows,
        },
        "public_fairness_benchmark": {
            "command": cmd_str,
            "commit_hash": commit_hash,
            "n_seeds": args.n_seeds,
            "seed_range": f"{seeds[0]}..{seeds[-1]}",
            "seeds": seeds,
            "frozen_settings": frozen_summary,
            "rows": pub_rows,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(aggregate_payload, indent=2) + "\n")
    generate_benchmark_markdown_from_json(args.output_json, args.output_md)
    print(
        f"Saved aggregate JSON -> {args.output_json}, per-seed artifact -> {per_seed_path}, "
        f"markdown -> {args.output_md} in {time.time() - t0:.1f}s."
    )


if __name__ == "__main__":
    main()
