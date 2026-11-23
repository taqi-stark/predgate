"""Reproducible public-dataset & synthetic fault-injection benchmark runner.

Downloads UCI Adult (OpenML 'adult' v2) and UCI German Credit (OpenML 'credit-g' v1),
runs both the synthetic 7-fault PredGate benchmark and the 30-seed full fairness
validation across Adult, German Credit, and Synthetic Credit with frozen defaults,
writes docs/benchmark_results.json, and renders docs/benchmark.md strictly from
that JSON file.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from predgate.benchmark import (
    FAULT_DESCRIPTIONS,
    GroupThresholdModel,
    _generate_data,
    _get_git_commit,
    generate_benchmark_markdown_from_json,
    run_single_trial,
    wilson_ci,
)
from predgate.checks.fairness import run_fairness_check
from predgate.config import FairnessConfig
from predgate.stats import demographic_parity_diff

PUBLIC_FAULTS: list[tuple[str, str]] = [
    ("fairness_threshold_shift", "`fairness_threshold_shift` (`τ=0.25`)"),
    ("fairness_threshold_shift_hi", "`fairness_threshold_shift_hi` (`τ=0.75`)"),
    ("fairness_proxy_feature", "`fairness_proxy_feature`"),
    ("fairness_regression", "`fairness_regression`"),
    ("noop", "`noop` (FP rate)"),
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


def _load_adult_frame() -> pd.DataFrame:
    """Download and clean UCI Adult ('adult', version=2) from OpenML."""
    bunch = fetch_openml("adult", version=2, as_frame=True)
    df = bunch.frame.copy()
    target_col = bunch.target.name if hasattr(bunch.target, "name") else "class"
    df["target"] = (df[target_col].astype(str).str.strip() == ">50K").astype(int)
    df = df.drop(columns=[target_col])
    for col in df.columns:
        if col == "target":
            continue
        if (
            isinstance(df[col].dtype, pd.CategoricalDtype)
            or df[col].dtype == object
            or str(df[col].dtype) in ("string", "str")
        ):
            df[col] = df[col].astype(str).fillna("missing").str.strip().astype(object)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _load_german_frame() -> pd.DataFrame:
    """Download and clean UCI German Credit ('credit-g', version=1) from OpenML."""
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
    for col in df.columns:
        if col == "target":
            continue
        if (
            isinstance(df[col].dtype, pd.CategoricalDtype)
            or df[col].dtype == object
            or str(df[col].dtype) in ("string", "str")
        ):
            df[col] = df[col].astype(str).fillna("missing").str.strip().astype(object)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _train_gbdt(X: pd.DataFrame, y: pd.Series, seed: int) -> Pipeline:
    """Train a deterministic GBDT pipeline matching PredGate's benchmark model family."""
    X_norm = X.copy()
    cat_cols = X_norm.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in cat_cols:
        X_norm[c] = X_norm[c].astype(object)
    num_cols = [c for c in X_norm.columns if c not in cat_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=seed),
            ),
        ]
    )
    pipe.fit(X_norm, y)
    return pipe


def _evaluate_fairness_seed(
    dataset_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    protected_col: str,
    minority_value: str,
    seed: int,
    frozen_cfg: FairnessConfig,
) -> list[dict[str, Any]]:
    """Run champion + all 5 fairness fault conditions for a single seed on one dataset."""
    X_clean = train_df.drop(columns=[target_col])
    y_clean = train_df[target_col]
    champion = _train_gbdt(X_clean, y_clean, seed=seed)

    X_test_base = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values
    prot_test = test_df[protected_col].values

    champ_preds = champion.predict(X_test_base)
    champ_acc = float(accuracy_score(y_test, champ_preds))
    champ_dp = float(demographic_parity_diff(champ_preds, prot_test))

    records: list[dict[str, Any]] = []
    for fault, _ in PUBLIC_FAULTS:
        rng = np.random.RandomState(seed)
        train_mod = train_df.copy()
        test_mod = test_df.copy()

        if fault == "fairness_threshold_shift":
            candidate: Any = GroupThresholdModel(
                champion,
                protected_col=protected_col,
                target_group=minority_value,
                group_threshold=0.25,
                default_threshold=0.50,
            )
        elif fault == "fairness_threshold_shift_hi":
            candidate = GroupThresholdModel(
                champion,
                protected_col=protected_col,
                target_group=minority_value,
                group_threshold=0.75,
                default_threshold=0.50,
            )
        elif fault == "fairness_proxy_feature":
            is_min_tr = (train_mod[protected_col].astype(str) == minority_value).astype(float)
            train_mod["proxy_score"] = is_min_tr * 0.85 + rng.normal(0.0, 0.25, len(train_mod))
            high_proxy_neg = train_mod[
                (train_mod["proxy_score"] > 0.6) & (train_mod[target_col] == 0)
            ]
            if len(high_proxy_neg) > 0:
                flip_idx = high_proxy_neg.sample(frac=0.35, random_state=seed).index
                train_mod.loc[flip_idx, target_col] = 1

            X_proxy = train_mod.drop(columns=[target_col, protected_col])
            candidate = _train_gbdt(X_proxy, train_mod[target_col], seed=seed)

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
            candidate = _train_gbdt(
                train_mod.drop(columns=[target_col]), train_mod[target_col], seed=seed
            )
        elif fault == "noop":
            candidate = _train_gbdt(X_clean, y_clean, seed=seed + 500)
        else:
            raise ValueError(f"Unknown fault: {fault}")

        X_test_eval = test_mod.drop(columns=[target_col])
        cand_preds = candidate.predict(X_test_eval)
        cand_acc = float(accuracy_score(y_test, cand_preds))
        cand_dp = float(demographic_parity_diff(cand_preds, test_mod[protected_col].values))
        baseline_detected = abs(champ_acc - cand_acc) > 0.01

        cfg = frozen_cfg.model_copy(update={"protected": [protected_col]})
        res = run_fairness_check(test_mod, target_col, candidate, champion, cfg, seed=seed)

        grp = res.details.get("groups", [{}])[0]
        records.append(
            {
                "dataset": dataset_name,
                "fault": fault,
                "seed": seed,
                "status": res.status,
                "detected": res.status in ("warn", "fail"),
                "is_fail": res.status == "fail",
                "is_warn": res.status == "warn",
                "baseline_detected": baseline_detected,
                "champ_dp": round(champ_dp, 4),
                "cand_dp": round(cand_dp, 4),
                "min_group_n": int(grp.get("min_group_n", 0)),
                "mde_dp": float(grp.get("min_detectable_gap_dp", 0.0)),
                "mde_eo": float(grp.get("min_detectable_gap_eo", 0.0)),
                "power_note_emitted": bool(res.details.get("power_notes")),
                "max_selection_rate_shift": float(res.metrics.get("max_selection_rate_shift", 0.0)),
            }
        )
    return records


def _run_synthetic_trial_arg(args: tuple[str, int]) -> dict[str, Any]:
    fault, seed = args
    return run_single_trial(fault, seed)


def _run_adult_seed(args: tuple[pd.DataFrame, int, dict[str, Any]]) -> list[dict[str, Any]]:
    adult_df, seed, cfg_dict = args
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(adult_df), size=4500, replace=False)
    train_df = adult_df.iloc[idx[:3000]].reset_index(drop=True)
    test_df = adult_df.iloc[idx[3000:]].reset_index(drop=True)
    return _evaluate_fairness_seed(
        "UCI Adult", train_df, test_df, "target", "sex", "Female", seed, FairnessConfig(**cfg_dict)
    )


def _run_german_seed(args: tuple[pd.DataFrame, int, dict[str, Any]]) -> list[dict[str, Any]]:
    german_df, seed, cfg_dict = args
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(german_df))
    split = int(len(german_df) * 0.65)
    train_df = german_df.iloc[idx[:split]].reset_index(drop=True)
    test_df = german_df.iloc[idx[split:]].reset_index(drop=True)
    return _evaluate_fairness_seed(
        "German Credit",
        train_df,
        test_df,
        "target",
        "sex",
        "female",
        seed,
        FairnessConfig(**cfg_dict),
    )


def _run_synth_fairness_seed(args: tuple[int, dict[str, Any]]) -> list[dict[str, Any]]:
    seed, cfg_dict = args
    train_df, test_df = _generate_data(seed=seed, n=3000)
    return _evaluate_fairness_seed(
        "Synthetic Credit",
        train_df,
        test_df,
        "default",
        "gender",
        "female",
        seed,
        FairnessConfig(**cfg_dict),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible public & synthetic PredGate benchmarks."
    )
    parser.add_argument(
        "--n-seeds", type=int, default=30, help="Seeds per dataset/fault (default: 30)"
    )
    parser.add_argument(
        "--synth-start-seed", type=int, default=1000, help="Start seed for synthetic 7-fault suite"
    )
    parser.add_argument(
        "--fairness-start-seed",
        type=int,
        default=5000,
        help="Fresh start seed for public fairness suite",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes")
    parser.add_argument("--output-json", type=Path, default=Path("docs/benchmark_results.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/benchmark.md"))
    args = parser.parse_args()

    t0 = time.time()
    commit_hash = _get_git_commit()
    synth_seeds = [args.synth_start_seed + i for i in range(args.n_seeds)]
    fairness_seeds = [args.fairness_start_seed + i for i in range(args.n_seeds)]

    frozen_cfg = FairnessConfig(enabled=True, protected=["sex"])
    frozen_dict = frozen_cfg.model_dump()
    frozen_summary = {
        "enforce_absolute": frozen_cfg.enforce_absolute,
        "max_gap_increase": frozen_cfg.max_gap_increase,
        "fail_gap_increase": frozen_cfg.fail_gap_increase,
        "selection_rate_shift_warn": frozen_cfg.selection_rate_shift_warn,
        "confidence": frozen_cfg.confidence,
        "n_bootstrap": frozen_cfg.n_bootstrap,
        "selection_rate_n_bootstrap": frozen_cfg.selection_rate_n_bootstrap,
    }

    print("[1/4] Downloading OpenML datasets (adult v2, credit-g v1)...")
    adult_df = _load_adult_frame()
    german_df = _load_german_frame()
    print(f"      Loaded UCI Adult {adult_df.shape} and German Credit {german_df.shape}")

    print(
        f"[2/4] Running synthetic 7-fault benchmark (seeds {synth_seeds[0]}..{synth_seeds[-1]})..."
    )
    synth_tasks = [(fault, s) for fault in SYNTHETIC_FAULTS for s in synth_seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        all_synth_results = list(pool.map(_run_synthetic_trial_arg, synth_tasks))

    synth_summary: dict[str, dict[str, Any]] = {}
    for fault in SYNTHETIC_FAULTS:
        rows = [r for r in all_synth_results if r["fault_type"] == fault]
        n = len(rows)
        mg_k = sum(1 for r in rows if r["predgate_detected"])
        bl_k = sum(1 for r in rows if r["baseline_detected"])
        mg_rate = mg_k / n if n else 0.0
        bl_rate = bl_k / n if n else 0.0
        mg_ci = wilson_ci(mg_k, n)
        bl_ci = wilson_ci(bl_k, n)
        ci_overlap = (mg_ci[0] <= bl_ci[1]) and (bl_ci[0] <= mg_ci[1])
        synth_summary[fault] = {
            "description": FAULT_DESCRIPTIONS[fault],
            "n_trials": n,
            "mg_detected": mg_k,
            "bl_detected": bl_k,
            "predgate_detection_rate": round(mg_rate, 3),
            "baseline_detection_rate": round(bl_rate, 3),
            "predgate_ci_95": [round(mg_ci[0], 3), round(mg_ci[1], 3)],
            "baseline_ci_95": [round(bl_ci[0], 3), round(bl_ci[1], 3)],
            "ci_95_overlap": ci_overlap,
            "predgate_wins": (mg_ci[0] > bl_ci[1]) if fault != "noop" else (mg_ci[1] < bl_ci[0]),
        }
        print(f"      {fault:22s}: MG {mg_k}/{n} ({mg_rate:.1%}) | BL {bl_k}/{n} ({bl_rate:.1%})")

    print(
        f"[3/4] Running full fairness benchmark on Adult, German Credit & Synthetic "
        f"(fresh seeds {fairness_seeds[0]}..{fairness_seeds[-1]})..."
    )
    pub_records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for batch in pool.map(
            _run_adult_seed, [(adult_df, s, frozen_dict) for s in fairness_seeds]
        ):
            pub_records.extend(batch)
        for batch in pool.map(
            _run_german_seed, [(german_df, s, frozen_dict) for s in fairness_seeds]
        ):
            pub_records.extend(batch)
        for batch in pool.map(_run_synth_fairness_seed, [(s, frozen_dict) for s in fairness_seeds]):
            pub_records.extend(batch)

    pub_rows: list[dict[str, Any]] = []
    for ds_name in ("UCI Adult", "German Credit", "Synthetic Credit"):
        for fault, flt_label in PUBLIC_FAULTS:
            subset = [r for r in pub_records if r["dataset"] == ds_name and r["fault"] == fault]
            n = len(subset)
            det = sum(1 for r in subset if r["detected"])
            det_fail = sum(1 for r in subset if r["is_fail"])
            det_warn = sum(1 for r in subset if r["is_warn"])
            bl_det = sum(1 for r in subset if r["baseline_detected"])
            ci = wilson_ci(det, n)
            bl_ci = wilson_ci(bl_det, n)
            row = {
                "dataset": ds_name,
                "fault": fault,
                "fault_label": flt_label,
                "n_trials": n,
                "detected_total": det,
                "detected_fail": det_fail,
                "detected_warn": det_warn,
                "detection_rate": round(det / n, 3) if n else 0.0,
                "ci_95": [round(ci[0], 3), round(ci[1], 3)],
                "baseline_detected": bl_det,
                "baseline_rate": round(bl_det / n, 3) if n else 0.0,
                "baseline_ci_95": [round(bl_ci[0], 3), round(bl_ci[1], 3)],
                "mean_champ_dp": round(float(np.mean([r["champ_dp"] for r in subset])), 3),
                "mean_cand_dp": round(float(np.mean([r["cand_dp"] for r in subset])), 3),
                "mean_min_n": round(float(np.mean([r["min_group_n"] for r in subset])), 1),
                "mean_mde_dp": round(float(np.mean([r["mde_dp"] for r in subset])), 3),
                "mean_mde_eo": round(float(np.mean([r["mde_eo"] for r in subset])), 3),
                "power_note_count": sum(1 for r in subset if r["power_note_emitted"]),
            }
            pub_rows.append(row)
            print(
                f"      {ds_name:16s} | {fault:27s}: "
                f"{det:2d}/{n} ({row['detection_rate']:.1%}, {det_fail}F/{det_warn}W) | "
                f"BL {bl_det:2d}/{n} | DP {row['mean_champ_dp']:.3f}->{row['mean_cand_dp']:.3f}"
            )

    payload = {
        "command": f"python scripts/bench_public.py --n-seeds {args.n_seeds} --synth-start-seed {args.synth_start_seed} --fairness-start-seed {args.fairness_start_seed}",
        "commit_hash": commit_hash,
        "n_seeds": args.n_seeds,
        "seed_range": f"{synth_seeds[0]}..{synth_seeds[-1]}",
        "seeds": synth_seeds,
        "summary": synth_summary,
        "details": all_synth_results,
        "public_fairness_benchmark": {
            "command": f"python scripts/bench_public.py --n-seeds {args.n_seeds} --fairness-start-seed {args.fairness_start_seed}",
            "commit_hash": commit_hash,
            "n_seeds": args.n_seeds,
            "seed_range": f"{fairness_seeds[0]}..{fairness_seeds[-1]}",
            "seeds": fairness_seeds,
            "frozen_settings": frozen_summary,
            "rows": pub_rows,
            "details": pub_records,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[4/4] Generating {args.output_md} strictly from {args.output_json}...")
    generate_benchmark_markdown_from_json(args.output_json, args.output_md)
    print(f"Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
