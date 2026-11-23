"""Target leakage detection check.

Scans training and test features for suspiciously high univariate association
with the target (Pearson |r| and rank-based single-feature ROC-AUC), as well as
newly introduced high-signal columns not present in the reference snapshot.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from predgate.config import LeakageConfig
from predgate.verdict import CheckResult


def _single_feature_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute symmetric rank-based single-feature ROC-AUC max(AUC, 1 - AUC) in [0.5, 1.0]."""
    pos = y == 1
    n_pos = int(np.sum(pos))
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = sp_stats.rankdata(x)
    auc = (float(np.sum(ranks[pos])) - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(max(auc, 1.0 - auc))


def _abs_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Compute |Pearson r(x, y)| safely, returning 0.0 for constant arrays."""
    if len(x) < 2:
        return 0.0
    std_x = float(np.std(x))
    std_y = float(np.std(y))
    if std_x < 1e-12 or std_y < 1e-12:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return abs(corr) if not np.isnan(corr) else 0.0


def run_leakage_check(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target_column: str,
    config: LeakageConfig,
    reference_df: pd.DataFrame | None = None,
) -> CheckResult:
    """Run target leakage check on training (and optional test/reference) data."""
    start = time.time()

    if not config.enabled:
        return CheckResult(
            name="leakage",
            status="skip",
            summary="Leakage check disabled",
            duration_s=time.time() - start,
        )

    if target_column not in train_df.columns:
        return CheckResult(
            name="leakage",
            status="skip",
            summary=f"Target column '{target_column}' not found",
            duration_s=time.time() - start,
        )

    y_train = np.asarray(train_df[target_column].values, dtype=np.float64)
    ref_cols = set(reference_df.columns) if reference_df is not None else set(train_df.columns)
    feature_cols = [c for c in train_df.columns if c != target_column]

    feature_scores: list[dict[str, Any]] = []
    max_corr = 0.0
    max_auc = 0.5
    fail_features: list[str] = []
    warn_features: list[str] = []

    for col in feature_cols:
        series = train_df[col]
        valid_mask = series.notna().values
        if int(np.sum(valid_mask)) < 10:
            continue

        y_valid = y_train[valid_mask]
        if pd.api.types.is_numeric_dtype(series):
            x_valid = series.values[valid_mask].astype(np.float64)
            # Ignore monotonic row index / ID columns that have near-uniform spacing and low correlation
            abs_r = _abs_pearson(x_valid, y_valid)
            auc = _single_feature_auc(x_valid, y_valid)
        else:
            # Target-mean encoding for categorical feature
            s_valid = series[valid_mask].astype(str)
            y_series = pd.Series(y_valid, index=s_valid.index, dtype=np.float64)
            x_valid = y_series.groupby(s_valid).transform("mean").values.astype(np.float64)
            abs_r = _abs_pearson(x_valid, y_valid)
            auc = _single_feature_auc(x_valid, y_valid)

        is_new_col = col not in ref_cols
        max_corr = max(max_corr, abs_r)
        max_auc = max(max_auc, auc)

        entry: dict[str, Any] = {
            "feature": col,
            "abs_corr": round(abs_r, 4),
            "single_feature_auc": round(auc, 4),
            "is_new_feature": is_new_col,
        }

        if test_df is not None and col in test_df.columns and target_column in test_df.columns:
            t_series = test_df[col]
            t_mask = t_series.notna().values
            if int(np.sum(t_mask)) >= 10 and pd.api.types.is_numeric_dtype(t_series):
                t_x = t_series.values[t_mask].astype(np.float64)
                t_y = np.asarray(test_df[target_column].values, dtype=np.float64)[t_mask]
                entry["test_abs_corr"] = round(_abs_pearson(t_x, t_y), 4)
                entry["test_single_feature_auc"] = round(_single_feature_auc(t_x, t_y), 4)

        if abs_r >= config.fail_corr or auc >= config.fail_auc:
            entry["verdict"] = "fail"
            fail_features.append(f"{col} (|r|={abs_r:.2f}, AUC={auc:.2f})")
        elif abs_r >= config.warn_corr or auc >= config.warn_auc:
            entry["verdict"] = "warn"
            warn_features.append(f"{col} (|r|={abs_r:.2f}, AUC={auc:.2f})")
        else:
            entry["verdict"] = "pass"

        feature_scores.append(entry)

    feature_scores.sort(key=lambda d: (d["abs_corr"], d["single_feature_auc"]), reverse=True)
    duration = time.time() - start

    status: Literal["pass", "warn", "fail", "skip"]
    if fail_features:
        status = "fail"
        summary = f"Target leakage detected: {'; '.join(fail_features[:3])}"
    elif warn_features:
        status = "warn"
        summary = f"Potential target leakage: {'; '.join(warn_features[:3])}"
    else:
        status = "pass"
        summary = f"Max |r| {max_corr:.2f}, max single-feature AUC {max_auc:.2f}"

    return CheckResult(
        name="leakage",
        status=status,
        summary=summary,
        metrics={
            "max_abs_corr": round(max_corr, 4),
            "max_single_feature_auc": round(max_auc, 4),
            "n_leaky_features": float(len(fail_features) + len(warn_features)),
        },
        details={
            "features": feature_scores,
            "top_5": feature_scores[:5],
        },
        duration_s=duration,
    )
