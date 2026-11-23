"""Drift detection check.

Compares shared feature distributions between candidate training data and the
reference snapshot using reference-quantile PSI, KS test, and numeric point-mass
(value-spike) shift detection (catching silent null-to-zero or sentinel fills).
Features absent from the reference snapshot are reported cleanly as
"new feature, no reference" (info or warn per config) without fabricating psi=1.0.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import pandas as pd

from predgate.config import DriftConfig
from predgate.stats import compute_point_mass_shift, compute_psi, compute_psi_categorical, ks_test
from predgate.verdict import CheckResult


def run_drift_check(
    train_df: pd.DataFrame,
    reference_df: pd.DataFrame | None,
    target_column: str,
    config: DriftConfig,
) -> CheckResult:
    """Run drift detection on shared features between candidate training data and reference."""
    start = time.time()

    if not config.enabled:
        return CheckResult(
            name="drift",
            status="skip",
            summary="Drift check disabled",
            duration_s=time.time() - start,
        )

    if reference_df is None:
        return CheckResult(
            name="drift",
            status="skip",
            summary="No reference data available",
            duration_s=time.time() - start,
        )

    feature_cols = [c for c in train_df.columns if c != target_column]
    feature_scores: list[dict[str, Any]] = []
    new_features: list[dict[str, str]] = []
    point_mass_spikes: list[dict[str, Any]] = []
    max_psi = 0.0
    max_point_mass_delta = 0.0

    for col in feature_cols:
        if col not in reference_df.columns:
            new_features.append(
                {
                    "feature": col,
                    "note": "new feature, no reference",
                }
            )
            continue

        train_vals = train_df[col].dropna()
        ref_vals = reference_df[col].dropna()

        if len(train_vals) == 0 or len(ref_vals) == 0:
            continue

        if pd.api.types.is_numeric_dtype(train_df[col]) and pd.api.types.is_numeric_dtype(
            reference_df[col]
        ):
            ref_num = ref_vals.values.astype(float)
            tr_num = train_vals.values.astype(float)
            psi = compute_psi(ref_num, tr_num)
            ks_stat, ks_p = ks_test(ref_num, tr_num)
            pm_delta, pm_val, pm_ref, pm_cur = compute_point_mass_shift(ref_num, tr_num)

            entry: dict[str, Any] = {
                "feature": col,
                "psi": round(psi, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": round(ks_p, 4),
                "point_mass_delta": round(pm_delta, 4),
            }
            if pm_val is not None and pm_delta > 0.0:
                entry["point_mass_value"] = round(pm_val, 4)
                entry["point_mass_ref_share"] = round(pm_ref, 4)
                entry["point_mass_cur_share"] = round(pm_cur, 4)
            feature_scores.append(entry)

            if pm_delta >= config.point_mass_warn_delta:
                point_mass_spikes.append(
                    {
                        "feature": col,
                        "value": pm_val,
                        "delta": round(pm_delta, 4),
                        "ref_share": round(pm_ref, 4),
                        "cur_share": round(pm_cur, 4),
                    }
                )
            max_point_mass_delta = max(max_point_mass_delta, pm_delta)
        else:
            psi = compute_psi_categorical(
                ref_vals.values.astype(str), train_vals.values.astype(str)
            )
            feature_scores.append(
                {
                    "feature": col,
                    "psi": round(psi, 4),
                }
            )

        max_psi = max(max_psi, feature_scores[-1]["psi"])

    feature_scores.sort(key=lambda x: x["psi"], reverse=True)
    point_mass_spikes.sort(key=lambda x: x["delta"], reverse=True)
    top_5 = feature_scores[:5]

    duration = time.time() - start

    fail_reasons: list[str] = []
    warn_reasons: list[str] = []

    if max_psi >= config.fail_at and top_5:
        fail_reasons.append(
            f"Max PSI {max_psi:.2f} >= {config.fail_at} (top: {top_5[0]['feature']})"
        )
    elif max_psi >= config.warn_at and top_5:
        warn_reasons.append(
            f"Max PSI {max_psi:.2f} >= {config.warn_at} (top: {top_5[0]['feature']})"
        )

    if max_point_mass_delta >= config.point_mass_fail_delta and point_mass_spikes:
        sp = point_mass_spikes[0]
        fail_reasons.append(
            f"Value spike on '{sp['feature']}' at {sp['value']} "
            f"(+{sp['delta']:.2f}, {sp['ref_share']:.0%}->{sp['cur_share']:.0%})"
        )
    elif max_point_mass_delta >= config.point_mass_warn_delta and point_mass_spikes:
        sp = point_mass_spikes[0]
        warn_reasons.append(
            f"Value spike on '{sp['feature']}' at {sp['value']} "
            f"(+{sp['delta']:.2f}, {sp['ref_share']:.0%}->{sp['cur_share']:.0%})"
        )

    new_feat_names = [nf["feature"] for nf in new_features]
    if new_features and config.on_new_feature == "warn":
        warn_reasons.append(f"new feature, no reference: {', '.join(new_feat_names[:3])}")

    status: Literal["pass", "warn", "fail", "skip"]
    if fail_reasons:
        status = "fail"
        summary = "; ".join(fail_reasons)
    elif warn_reasons:
        status = "warn"
        summary = "; ".join(warn_reasons)
    else:
        status = "pass"
        summary = f"Max PSI {max_psi:.2f}, no significant drift"

    if new_features and config.on_new_feature == "info":
        summary = f"{summary} [new feature, no reference: {', '.join(new_feat_names[:3])}]"

    return CheckResult(
        name="drift",
        status=status,
        summary=summary,
        metrics={
            "psi_max": max_psi,
            "max_point_mass_delta": round(max_point_mass_delta, 4),
            "features_checked": float(len(feature_scores)),
            "n_new_features": float(len(new_features)),
        },
        details={
            "feature_scores": feature_scores,
            "top_drifted": top_5,
            "new_features": new_features,
            "point_mass_spikes": point_mass_spikes,
        },
        duration_s=duration,
    )
