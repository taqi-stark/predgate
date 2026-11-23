"""Drift detection check.

Compares feature distributions between candidate training data and the
reference snapshot using PSI (numeric/categorical) and KS test (numeric).
"""

from __future__ import annotations

import time

import pandas as pd

from predgate.config import DriftConfig
from predgate.stats import compute_psi, compute_psi_categorical, ks_test
from predgate.verdict import CheckResult


def run_drift_check(
    train_df: pd.DataFrame,
    reference_df: pd.DataFrame | None,
    target_column: str,
    config: DriftConfig,
) -> CheckResult:
    """Run drift detection between training data and reference.

    Args:
        train_df: Current training data.
        reference_df: Reference/baseline data snapshot.
        target_column: Name of the target column to exclude.
        config: Drift check configuration.

    Returns:
        CheckResult with per-feature drift scores.
    """
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
    feature_scores: list[dict] = []
    max_psi = 0.0

    for col in feature_cols:
        if col not in reference_df.columns:
            feature_scores.append(
                {
                    "feature": col,
                    "psi": 1.0,
                    "ks_statistic": 1.0,
                    "ks_pvalue": 0.0,
                    "note": "New feature not in reference data",
                }
            )
            max_psi = max(max_psi, 1.0)
            continue

        train_vals = train_df[col].dropna()
        ref_vals = reference_df[col].dropna()

        if len(train_vals) == 0 or len(ref_vals) == 0:
            continue

        if pd.api.types.is_numeric_dtype(train_df[col]):
            psi = compute_psi(ref_vals.values.astype(float), train_vals.values.astype(float))
            ks_stat, ks_p = ks_test(ref_vals.values.astype(float), train_vals.values.astype(float))
            feature_scores.append(
                {
                    "feature": col,
                    "psi": round(psi, 4),
                    "ks_statistic": round(ks_stat, 4),
                    "ks_pvalue": round(ks_p, 4),
                }
            )
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

    # Sort by PSI descending
    feature_scores.sort(key=lambda x: x["psi"], reverse=True)
    top_5 = feature_scores[:5]

    duration = time.time() - start

    from typing import Literal

    status: Literal["pass", "warn", "fail", "skip"]
    if max_psi >= config.fail_at:
        status = "fail"
        summary = f"Max PSI {max_psi:.2f} >= {config.fail_at} (top: {top_5[0]['feature']})"
    elif max_psi >= config.warn_at:
        status = "warn"
        summary = f"Max PSI {max_psi:.2f} >= {config.warn_at} (top: {top_5[0]['feature']})"
    else:
        status = "pass"
        summary = f"Max PSI {max_psi:.2f}, no significant drift"

    return CheckResult(
        name="drift",
        status=status,
        summary=summary,
        metrics={"psi_max": max_psi, "features_checked": float(len(feature_scores))},
        details={"feature_scores": feature_scores, "top_drifted": top_5},
        duration_s=duration,
    )
