"""Measure benign-change WARN and FAIL rates for config presets (strict, balanced, quiet) on fresh seeds.

Evaluates all 3 presets across 30 fresh seeds (default: 11000..11029) x 5 datasets x 9 benign changes
(1,350 trials per preset; 4,050 paired evaluations total) without modifying default thresholds.
Writes aggregate summary JSON to docs/preset_benchmark_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow importing shared dataset loaders & model builders from scripts/bench_public.py
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import bench_public as bp  # noqa: E402

from predgate.bench import _get_git_commit, wilson_ci  # noqa: E402
from predgate.checks.budgets import run_budgets_check  # noqa: E402
from predgate.checks.drift import run_drift_check  # noqa: E402
from predgate.checks.fairness import run_fairness_check  # noqa: E402
from predgate.checks.label_shift import run_label_shift_check  # noqa: E402
from predgate.checks.leakage import run_leakage_check  # noqa: E402
from predgate.checks.slices import run_slices_check  # noqa: E402
from predgate.config import (  # noqa: E402
    PRESET_CONFIGS,
    BudgetConfig,
    PresetName,
    get_preset_checks_config,
)
from predgate.verdict import CheckResult, compute_verdict  # noqa: E402

PRESETS: tuple[PresetName, ...] = ("strict", "balanced", "quiet")


def _init_preset_worker() -> None:
    warnings.filterwarnings("ignore", message="Found unknown categories")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _drift_status_for_cfg(base_cr: CheckResult, cfg: Any) -> str:
    m = base_cr.metrics
    psi_max = float(m.get("psi_max", 0.0))
    has_feats = float(m.get("features_checked", 0.0)) > 0
    max_pm = float(m.get("max_point_mass_delta", 0.0))
    n_new = int(m.get("n_new_features", 0.0))
    has_fail = (psi_max >= cfg.fail_at and has_feats) or (max_pm >= cfg.point_mass_fail_delta)
    if has_fail:
        return "fail"
    has_warn = (
        (psi_max >= cfg.warn_at and has_feats)
        or (max_pm >= cfg.point_mass_warn_delta)
        or (n_new > 0 and cfg.on_new_feature == "warn")
    )
    return "warn" if has_warn else "pass"


def _leakage_status_for_cfg(base_cr: CheckResult, cfg: Any) -> str:
    feats = base_cr.details.get("features", [])
    has_warn = False
    for f in feats:
        r = float(f.get("abs_corr", 0.0))
        auc = float(f.get("single_feature_auc", 0.5))
        if r >= cfg.fail_corr or auc >= cfg.fail_auc:
            return "fail"
        if r >= cfg.warn_corr or auc >= cfg.warn_auc:
            has_warn = True
    return "warn" if has_warn else "pass"


def _label_shift_status_for_cfg(base_cr: CheckResult, cfg: Any) -> str:
    subgroups = base_cr.details.get("subgroups", [])
    has_warn = False
    for e in subgroups:
        if not e.get("significant", False):
            continue
        d = float(e.get("delta", 0.0))
        if d >= cfg.fail_delta:
            return "fail"
        if d >= cfg.warn_delta:
            has_warn = True
    return "warn" if has_warn else "pass"


def _slices_status_for_cfg(base_cr: CheckResult, cfg: Any) -> tuple[str, int]:
    slices = base_cr.details.get("slices", [])
    has_fail = False
    has_warn = False
    n_inc = 0
    for sr in slices:
        if float(sr.get("ci_width", 0.0)) > cfg.max_ci_width:
            n_inc += 1
        if float(sr.get("ci_lower", 0.0)) > cfg.max_drop:
            has_fail = True
        elif float(sr.get("diff", 0.0)) > cfg.max_drop:
            has_warn = True
    if has_fail:
        return "fail", n_inc
    if has_warn:
        return "warn", n_inc
    if n_inc > 0 and cfg.on_inconclusive == "warn":
        return "warn", n_inc
    return "pass", n_inc


def _fairness_status_for_cfg(base_cr: CheckResult, cfg: Any) -> str:
    has_fail = False
    has_warn = False
    for g in base_cr.details.get("groups", []):
        for boot in g.get("all_gap_bootstraps", {}).values():
            pt = float(boot.get("point_estimate", 0.0))
            lo = float(boot.get("ci_lower", 0.0))
            if lo > 0.0 and (pt >= cfg.fail_gap_increase or lo >= cfg.max_gap_increase):
                has_fail = True
            elif lo > 0.0 and pt >= cfg.max_gap_increase:
                has_warn = True
        for s in g.get("selection_rate_shifts", []):
            is_sig = bool(s.get("significant", False))
            lo = float(s.get("ci_lower", 0.0))
            hi = float(s.get("ci_upper", 0.0))
            delta = abs(float(s.get("delta", 0.0)))
            if is_sig and (lo > 0.0 or hi < 0.0) and delta >= cfg.selection_rate_shift_warn:
                has_warn = True
    if has_fail:
        return "fail"
    if has_warn:
        return "warn"
    return "pass"


def _budget_status_for_limit(
    base_budget_res: CheckResult,
    max_latency_ratio: float | None,
    max_model_mb: float,
    max_peak_mem_mb: float,
) -> str:
    """Evaluate exact budgets.py decision rule for a preset using the single measured paired run."""
    m = base_budget_res.metrics
    ratio = m.get("latency_ratio")
    ci_lo = m.get("latency_ratio_ci_lower")
    size_mb = m.get("model_size_mb", 0.0)
    peak_mb = m.get("peak_mem_mb", 0.0)

    has_fail = False
    has_warn = False
    if ratio is not None and ci_lo is not None and max_latency_ratio is not None:
        if ci_lo > max_latency_ratio:
            has_fail = True
        elif ratio > max_latency_ratio and ci_lo > 1.0:
            has_warn = True
    if size_mb > max_model_mb or peak_mb > max_peak_mem_mb:
        has_fail = True
    if has_fail:
        return "fail"
    if has_warn:
        return "warn"
    return "pass"


def _run_preset_dataset_seed(args: tuple[str, pd.DataFrame | None, int]) -> list[dict[str, Any]]:
    ds_name, full_df, seed = args
    (
        train_base,
        train_more,
        test_base,
        target_col,
        protected_col,
        _minority_value,
        slice_cols,
        _null_col,
    ) = bp._split_dataset(ds_name, full_df, seed)

    X_base = train_base.drop(columns=[target_col])
    y_base = train_base[target_col]
    champion = bp._train_pipeline(X_base, y_base, seed=seed, variant="champion_gbdt")

    # Paired champion for drop_irrelevant_feature
    nrng_drop = np.random.RandomState(seed + 888)
    train_with_noise = train_base.copy()
    test_with_noise = test_base.copy()
    train_with_noise["irrelevant_noise_col"] = nrng_drop.normal(
        0.0, 1.0, size=len(train_with_noise)
    )
    test_with_noise["irrelevant_noise_col"] = nrng_drop.normal(0.0, 1.0, size=len(test_with_noise))
    champ_with_noise = bp._train_pipeline(
        train_with_noise.drop(columns=[target_col]),
        train_with_noise[target_col],
        seed=seed,
        variant="champion_gbdt",
    )

    # Paired champion for change_imputation (5% MCAR NaNs)
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
    champ_impute = bp._train_pipeline(
        train_miss.drop(columns=[target_col]),
        train_miss[target_col],
        seed=seed,
        variant="champ_imputation",
    )

    preset_cfgs = {p: get_preset_checks_config(p) for p in PRESETS}
    records: list[dict[str, Any]] = []

    for change in bp.BENIGN_CHANGES:
        ref_train_use = train_base
        champ_use = champion
        train_cand = train_base.copy()
        test_eval = test_base.copy()

        if change in (
            "hyperparam_change",
            "algo_swap_rf",
            "algo_swap_logreg",
            "preprocessing_change",
            "rescale_inputs",
        ):
            candidate = bp._train_pipeline(
                train_cand.drop(columns=[target_col]),
                train_cand[target_col],
                seed=seed + 100,
                variant=change,
            )
        elif change == "change_imputation":
            ref_train_use = train_miss
            champ_use = champ_impute
            train_cand = train_miss.copy()
            test_eval = test_miss.copy()
            candidate = bp._train_pipeline(
                train_cand.drop(columns=[target_col]),
                train_cand[target_col],
                seed=seed + 100,
                variant="change_imputation",
            )
        elif change == "more_data_20pct":
            train_cand = train_more.copy()
            candidate = bp._train_pipeline(
                train_cand.drop(columns=[target_col]),
                train_cand[target_col],
                seed=seed + 100,
                variant="champion_gbdt",
            )
        elif change == "irrelevant_feature":
            nrng = np.random.RandomState(seed + 777)
            train_cand["benign_noise_feat"] = nrng.normal(0.0, 1.0, size=len(train_cand))
            test_eval["benign_noise_feat"] = nrng.normal(0.0, 1.0, size=len(test_eval))
            candidate = bp._train_pipeline(
                train_cand.drop(columns=[target_col]),
                train_cand[target_col],
                seed=seed + 100,
                variant="champion_gbdt",
            )
        elif change == "drop_irrelevant_feature":
            ref_train_use = train_with_noise
            champ_use = champ_with_noise
            train_cand = train_base.copy()
            test_eval = test_with_noise.copy()
            candidate = bp._train_pipeline(
                train_cand.drop(columns=[target_col]),
                train_cand[target_col],
                seed=seed + 100,
                variant="champion_gbdt",
            )
        else:
            raise ValueError(f"Unknown benign change: {change}")

        # Run all 6 checks ONCE with balanced config (identical sampling, bootstraps, alpha, seed)
        # so all 3 presets compare identical statistical draws with zero redundant bootstrap loops.
        bal_cfg = preset_cfgs["balanced"]
        base_drift = run_drift_check(train_cand, ref_train_use, target_col, bal_cfg.drift)
        base_leak = run_leakage_check(
            train_cand, test_eval, target_col, bal_cfg.leakage, reference_df=ref_train_use
        )
        base_lshift = run_label_shift_check(
            train_cand,
            ref_train_use,
            target_col,
            bal_cfg.label_shift.model_copy(update={"by": slice_cols}),
        )
        base_slices = run_slices_check(
            test_eval,
            target_col,
            champ_use,
            candidate,
            bal_cfg.slices.model_copy(update={"by": slice_cols, "n_bootstrap": 200}),
            seed=seed,
        )
        base_fair = run_fairness_check(
            test_eval,
            target_col,
            candidate,
            champ_use,
            bal_cfg.fairness.model_copy(update={"protected": [protected_col]}),
            seed=seed,
        )
        base_budget = run_budgets_check(
            test_eval,
            target_col,
            candidate,
            champ_use,
            BudgetConfig(enabled=True),
            seed=seed,
        )

        for preset in PRESETS:
            pc = preset_cfgs[preset]
            drift_st = _drift_status_for_cfg(base_drift, pc.drift)
            leak_st = _leakage_status_for_cfg(base_leak, pc.leakage)
            lshift_st = _label_shift_status_for_cfg(base_lshift, pc.label_shift)
            slices_st, n_inc = _slices_status_for_cfg(base_slices, pc.slices)
            fair_st = _fairness_status_for_cfg(base_fair, pc.fairness)
            bud_st = _budget_status_for_limit(
                base_budget,
                pc.budgets.max_latency_ratio,
                pc.budgets.max_model_mb,
                pc.budgets.max_peak_mem_mb,
            )

            all_crs = [
                CheckResult(name="drift", status=drift_st, summary=""),  # type: ignore[arg-type]
                CheckResult(name="leakage", status=leak_st, summary=""),  # type: ignore[arg-type]
                CheckResult(name="label_shift", status=lshift_st, summary=""),  # type: ignore[arg-type]
                CheckResult(name="slices", status=slices_st, summary=""),  # type: ignore[arg-type]
                CheckResult(name="fairness", status=fair_st, summary=""),  # type: ignore[arg-type]
                CheckResult(name="budgets", status=bud_st, summary=""),  # type: ignore[arg-type]
            ]
            verdict = compute_verdict(all_crs)
            statuses = {cr.name: cr.status for cr in all_crs}
            q_statuses = [statuses[c] for c in bp.QUALITY_CHECKS]
            q_fail = "fail" in q_statuses
            q_warn_only = (not q_fail) and ("warn" in q_statuses)

            records.append(
                {
                    "preset": preset,
                    "dataset": ds_name,
                    "change": change,
                    "seed": seed,
                    "is_fail": verdict.status == "fail",
                    "is_warn_only": verdict.status == "warn",
                    "quality_fail": q_fail,
                    "quality_warn_only": q_warn_only,
                    "latency_fail": bud_st == "fail",
                    "latency_warn": bud_st == "warn",
                    "check_statuses": statuses,
                    "n_inconclusive_slices": n_inc,
                }
            )

    return records


def _agg(k: int, n: int) -> dict[str, Any]:
    rate = round(k / n, 3) if n else 0.0
    lo, hi = wilson_ci(k, n)
    return {"count": k, "rate": rate, "ci_95": [round(lo, 3), round(hi, 3)]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure benign-change WARN/FAIL rates for strict, balanced, and quiet presets on fresh seeds."
    )
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=11000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output-json", type=Path, default=Path("docs/preset_benchmark_results.json")
    )
    args = parser.parse_args()

    t0 = time.time()
    commit_hash = _get_git_commit()
    seeds = [args.start_seed + i for i in range(args.n_seeds)]

    print("Loading OpenML datasets (Adult, German Credit, Bank Marketing, COMPAS)...", flush=True)
    ds_frames: dict[str, pd.DataFrame | None] = {
        "UCI Adult": bp._load_adult_frame(),
        "German Credit": bp._load_german_frame(),
        "Synthetic Credit": None,
        "Bank Marketing": bp._load_bank_marketing_frame(),
        "COMPAS": bp._load_compas_frame(),
    }

    tasks = [(ds_name, ds_frames[ds_name], s) for ds_name in bp.ALL_DATASETS for s in seeds]
    all_records: list[dict[str, Any]] = []
    print(
        f"Evaluating 3 presets x 9 benign changes x 5 datasets x {len(seeds)} fresh seeds "
        f"({seeds[0]}..{seeds[-1]})...",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_preset_worker) as pool:
        for idx, recs in enumerate(pool.map(_run_preset_dataset_seed, tasks), start=1):
            all_records.extend(recs)
            if idx % 5 == 0 or idx == len(tasks):
                elapsed = time.time() - t0
                print(
                    f"  progress: {idx}/{len(tasks)} dataset-seeds ({elapsed:.1f}s)",
                    flush=True,
                )

    summary_by_preset: dict[str, Any] = {}
    for preset in PRESETS:
        sub = [r for r in all_records if r["preset"] == preset]
        non_swap = [r for r in sub if r["change"] not in ("algo_swap_rf", "algo_swap_logreg")]
        algo_swap = [r for r in sub if r["change"] in ("algo_swap_rf", "algo_swap_logreg")]
        n = len(sub)
        n_ns = len(non_swap)
        n_as = len(algo_swap)
        per_check = {}
        for chk in bp.ALL_CHECKS:
            f_cnt = sum(r["check_statuses"][chk] == "fail" for r in sub)
            w_cnt = sum(r["check_statuses"][chk] == "warn" for r in sub)
            per_check[chk] = {
                "fail": _agg(f_cnt, n),
                "warn": _agg(w_cnt, n),
                "flagged": _agg(f_cnt + w_cnt, n),
            }
        q_fail_cnt = sum(r["quality_fail"] for r in sub)
        q_warn_cnt = sum(r["quality_warn_only"] for r in sub)
        c_fail_cnt = sum(r["is_fail"] for r in sub)
        c_warn_cnt = sum(r["is_warn_only"] for r in sub)
        summary_by_preset[preset] = {
            "n_trials": n,
            "thresholds": PRESET_CONFIGS[preset],
            "quality_fail": _agg(q_fail_cnt, n),
            "quality_warn_only": _agg(q_warn_cnt, n),
            "quality_flagged": _agg(q_fail_cnt + q_warn_cnt, n),
            "latency_fail": _agg(sum(r["latency_fail"] for r in sub), n),
            "latency_warn": _agg(sum(r["latency_warn"] for r in sub), n),
            "combined_fail": _agg(c_fail_cnt, n),
            "combined_warn_only": _agg(c_warn_cnt, n),
            "combined_flagged": _agg(c_fail_cnt + c_warn_cnt, n),
            "non_swap_n_trials": n_ns,
            "non_swap_quality_fail": _agg(sum(r["quality_fail"] for r in non_swap), n_ns),
            "non_swap_quality_warn_only": _agg(sum(r["quality_warn_only"] for r in non_swap), n_ns),
            "algo_swap_n_trials": n_as,
            "algo_swap_quality_fail": _agg(sum(r["quality_fail"] for r in algo_swap), n_as),
            "algo_swap_quality_warn_only": _agg(
                sum(r["quality_warn_only"] for r in algo_swap), n_as
            ),
            "algo_swap_latency_fail": _agg(sum(r["latency_fail"] for r in algo_swap), n_as),
            "inconclusive_slice_trials": _agg(sum(r["n_inconclusive_slices"] > 0 for r in sub), n),
            "per_check": per_check,
        }
        ps = summary_by_preset[preset]
        print(
            f"  PRESET {preset:8s} (N={n}): "
            f"Q_FAIL {ps['quality_fail']['count']}/{n} ({ps['quality_fail']['rate']:.1%}) | "
            f"Q_WARN {ps['quality_warn_only']['count']}/{n} ({ps['quality_warn_only']['rate']:.1%}) | "
            f"LAT_FAIL {ps['latency_fail']['count']}/{n} ({ps['latency_fail']['rate']:.1%}) | "
            f"COMBINED_FAIL {ps['combined_fail']['count']}/{n} ({ps['combined_fail']['rate']:.1%}) | "
            f"COMBINED_WARN {ps['combined_warn_only']['count']}/{n} ({ps['combined_warn_only']['rate']:.1%}) | "
            f"NON_SWAP_Q_FAIL {ps['non_swap_quality_fail']['count']}/{n_ns} ({ps['non_swap_quality_fail']['rate']:.1%})",
            flush=True,
        )

    payload = {
        "command": f"python scripts/bench_presets.py --n-seeds {args.n_seeds} --start-seed {args.start_seed}",
        "commit_hash": commit_hash,
        "n_seeds": args.n_seeds,
        "seed_range": f"{seeds[0]}..{seeds[-1]}",
        "seeds": seeds,
        "datasets": bp.ALL_DATASETS,
        "changes": bp.BENIGN_CHANGES,
        "presets": summary_by_preset,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved preset benchmark summary -> {args.output_json} in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
