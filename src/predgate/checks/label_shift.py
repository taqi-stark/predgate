"""Label-shift detection check.

Detects overall prior label shift P_train(Y) vs P_ref(Y) and conditional
subgroup label shift P_train(Y | S=s) vs P_ref(Y | S=s) using a two-proportion
z-test with Holm-Bonferroni multiple-testing correction.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from predgate.config import LabelShiftConfig
from predgate.stats import holm_bonferroni
from predgate.verdict import CheckResult


def _two_proportion_z_pvalue(k1: int, n1: int, k2: int, n2: int) -> float:
    """Compute two-sided p-value for difference between two binomial proportions."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p1 = k1 / float(n1)
    p2 = k2 / float(n2)
    p_pool = (k1 + k2) / float(n1 + n2)
    denom = p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)
    if denom <= 1e-15:
        return 1.0 if abs(p1 - p2) < 1e-12 else 0.0
    z = abs(p1 - p2) / float(np.sqrt(denom))
    return float(2.0 * (1.0 - sp_stats.norm.cdf(z)))


def run_label_shift_check(
    train_df: pd.DataFrame,
    reference_df: pd.DataFrame | None,
    target_column: str,
    config: LabelShiftConfig,
) -> CheckResult:
    """Run overall and per-subgroup label-shift detection between train_df and reference_df."""
    start = time.time()

    if not config.enabled:
        return CheckResult(
            name="label_shift",
            status="skip",
            summary="Label-shift check disabled",
            duration_s=time.time() - start,
        )

    if reference_df is None:
        return CheckResult(
            name="label_shift",
            status="skip",
            summary="No reference data available for label-shift comparison",
            duration_s=time.time() - start,
        )

    if target_column not in train_df.columns or target_column not in reference_df.columns:
        return CheckResult(
            name="label_shift",
            status="skip",
            summary=f"Target column '{target_column}' not found",
            duration_s=time.time() - start,
        )

    y_train = np.asarray(train_df[target_column].values, dtype=int)
    y_ref = np.asarray(reference_df[target_column].values, dtype=int)

    n_train = len(y_train)
    n_ref = len(y_ref)
    if n_train == 0 or n_ref == 0:
        return CheckResult(
            name="label_shift",
            status="skip",
            summary="Empty training or reference target",
            duration_s=time.time() - start,
        )

    k_train = int(np.sum(y_train == 1))
    k_ref = int(np.sum(y_ref == 1))
    overall_train_rate = k_train / float(n_train)
    overall_ref_rate = k_ref / float(n_ref)
    overall_delta = abs(overall_train_rate - overall_ref_rate)
    overall_p = _two_proportion_z_pvalue(k_train, n_train, k_ref, n_ref)

    slice_cols = (
        config.by
        if config.by
        else [
            c
            for c in train_df.select_dtypes(include=["object", "string", "category"]).columns
            if c != target_column and c in reference_df.columns
        ]
    )

    subgroup_entries: list[dict[str, Any]] = [
        {
            "slice": "overall",
            "n_train": n_train,
            "n_ref": n_ref,
            "train_rate": round(overall_train_rate, 4),
            "ref_rate": round(overall_ref_rate, 4),
            "delta": round(overall_delta, 4),
            "p_value": round(overall_p, 6),
        }
    ]
    raw_p_values: list[float] = [overall_p]

    for col in slice_cols:
        if col not in train_df.columns or col not in reference_df.columns:
            continue
        t_col = train_df[col].astype(str).values
        r_col = reference_df[col].astype(str).values
        for val in np.unique(r_col):
            t_mask = t_col == val
            r_mask = r_col == val
            nt = int(np.sum(t_mask))
            nr = int(np.sum(r_mask))
            if nt < config.min_slice_size or nr < config.min_slice_size:
                continue
            kt = int(np.sum(y_train[t_mask] == 1))
            kr = int(np.sum(y_ref[r_mask] == 1))
            tr_rate = kt / float(nt)
            rf_rate = kr / float(nr)
            delta = abs(tr_rate - rf_rate)
            pval = _two_proportion_z_pvalue(kt, nt, kr, nr)
            raw_p_values.append(pval)
            subgroup_entries.append(
                {
                    "slice": f"{col}={val}",
                    "n_train": nt,
                    "n_ref": nr,
                    "train_rate": round(tr_rate, 4),
                    "ref_rate": round(rf_rate, 4),
                    "delta": round(delta, 4),
                    "p_value": round(pval, 6),
                }
            )

    rejected = holm_bonferroni(raw_p_values, alpha=config.alpha)
    fail_slices: list[str] = []
    warn_slices: list[str] = []
    max_delta = 0.0

    for i, entry in enumerate(subgroup_entries):
        sig = rejected[i] if i < len(rejected) else False
        entry["significant"] = sig
        d = float(entry["delta"])
        max_delta = max(max_delta, d)
        if sig and d >= config.fail_delta:
            entry["verdict"] = "fail"
            fail_slices.append(
                f"{entry['slice']} rate {entry['ref_rate']:.2f}->{entry['train_rate']:.2f} "
                f"(Δ={d:.2f}, p={entry['p_value']:.4f})"
            )
        elif sig and d >= config.warn_delta:
            entry["verdict"] = "warn"
            warn_slices.append(
                f"{entry['slice']} rate {entry['ref_rate']:.2f}->{entry['train_rate']:.2f} "
                f"(Δ={d:.2f}, p={entry['p_value']:.4f})"
            )
        else:
            entry["verdict"] = "pass"

    subgroup_entries.sort(key=lambda e: e["delta"], reverse=True)
    duration = time.time() - start

    status: Literal["pass", "warn", "fail", "skip"]
    if fail_slices:
        status = "fail"
        summary = f"Label shift detected: {'; '.join(fail_slices[:2])}"
    elif warn_slices:
        status = "warn"
        summary = f"Moderate label shift: {'; '.join(warn_slices[:2])}"
    else:
        status = "pass"
        summary = f"Max label rate shift {max_delta:.2f} (overall {overall_delta:.2f})"

    return CheckResult(
        name="label_shift",
        status=status,
        summary=summary,
        metrics={
            "overall_label_delta": round(overall_delta, 4),
            "max_slice_label_delta": round(max_delta, 4),
            "n_shifted_slices": float(len(fail_slices) + len(warn_slices)),
        },
        details={"subgroups": subgroup_entries},
        duration_s=duration,
    )
