"""Slice regression check.

For each configured slicing column, computes the metric for champion and
candidate on the test set using paired bootstrap for confidence intervals.
Applies Holm-Bonferroni correction across slices.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from predgate.config import SliceConfig
from predgate.stats import holm_bonferroni, paired_bootstrap_ci
from predgate.verdict import CheckResult

METRIC_FUNCTIONS: dict[str, Any] = {
    "f1": lambda y, p: f1_score(y, p, average="weighted", zero_division=0),
    "accuracy": accuracy_score,
    "precision": lambda y, p: precision_score(y, p, average="weighted", zero_division=0),
    "recall": lambda y, p: recall_score(y, p, average="weighted", zero_division=0),
}


def _auto_bin(series: pd.Series, n_bins: int = 5) -> pd.Series:
    """Auto-bin a numeric column into quantile buckets.

    Args:
        series: Numeric pandas Series.
        n_bins: Number of quantile bins.

    Returns:
        Categorical Series with bucket labels.
    """
    try:
        return pd.qcut(series, q=n_bins, labels=False, duplicates="drop").astype(str)
    except ValueError:
        return series.astype(str)


def run_slices_check(
    test_df: pd.DataFrame,
    target_column: str,
    champion_model: Any | None,
    candidate_model: Any,
    config: SliceConfig,
    seed: int = 42,
) -> CheckResult:
    """Run slice regression check comparing candidate to champion.

    Args:
        test_df: Test DataFrame.
        target_column: Name of the target column.
        champion_model: The champion model (None if first run).
        candidate_model: The candidate model.
        config: Slice check configuration.
        seed: Random seed for bootstrap.

    Returns:
        CheckResult with per-slice analysis.
    """
    start = time.time()

    if not config.enabled:
        return CheckResult(
            name="slices",
            status="skip",
            summary="Slice check disabled",
            duration_s=time.time() - start,
        )

    if champion_model is None:
        return CheckResult(
            name="slices",
            status="skip",
            summary="No champion model; skipping slice comparison",
            duration_s=time.time() - start,
        )

    metric_fn = METRIC_FUNCTIONS.get(config.metric, METRIC_FUNCTIONS["f1"])

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column].values

    champion_preds = champion_model.predict(X_test)
    candidate_preds = candidate_model.predict(X_test)

    # Compute overall metrics
    overall_champion = metric_fn(y_test, champion_preds)
    overall_candidate = metric_fn(y_test, candidate_preds)

    slice_results: list[dict[str, Any]] = []
    p_values: list[float] = []
    slice_names: list[str] = []

    for col in config.by:
        if col not in test_df.columns:
            continue

        series = test_df[col]
        # Auto-bin numeric columns
        if pd.api.types.is_numeric_dtype(series):
            slice_col = _auto_bin(series)
        else:
            slice_col = series.astype(str)

        for slice_val in slice_col.unique():
            mask = (slice_col == slice_val).values
            n_rows = int(np.sum(mask))

            if n_rows < config.min_slice_size:
                continue

            y_slice = y_test[mask]
            champ_slice = champion_preds[mask]
            cand_slice = candidate_preds[mask]

            # Paired bootstrap for CI
            bootstrap = paired_bootstrap_ci(
                y_slice,
                champ_slice,
                cand_slice,
                metric_fn,
                n_bootstrap=config.n_bootstrap,
                confidence=config.confidence,
                seed=seed,
                metric_name=config.metric,
            )

            p_values.append(bootstrap.p_value)
            slice_names.append(f"{col}={slice_val}")

            champ_metric = float(metric_fn(y_slice, champ_slice))
            cand_metric = float(metric_fn(y_slice, cand_slice))

            slice_results.append(
                {
                    "slice": f"{col}={slice_val}",
                    "column": col,
                    "value": str(slice_val),
                    "n_rows": n_rows,
                    "champion_metric": round(champ_metric, 4),
                    "candidate_metric": round(cand_metric, 4),
                    "diff": round(bootstrap.point_estimate, 4),
                    "ci_lower": round(bootstrap.ci_lower, 4),
                    "ci_upper": round(bootstrap.ci_upper, 4),
                    "p_value": round(bootstrap.p_value, 4),
                }
            )

    # Apply Holm-Bonferroni
    rejected = holm_bonferroni(p_values, alpha=1.0 - config.confidence) if p_values else []

    # Determine status per slice and overall
    has_fail = False
    has_warn = False
    failing_slices: list[str] = []

    for i, sr in enumerate(slice_results):
        sr["significant"] = rejected[i] if i < len(rejected) else False

        # FAIL if upper bound of CI exceeds max_drop
        if sr["ci_upper"] > config.max_drop:
            sr["verdict"] = "fail"
            has_fail = True
            failing_slices.append(
                f"{sr['slice']} {config.metric} -{sr['diff']:.2f} "
                f"(95% CI {sr['ci_lower']:.2f} to {sr['ci_upper']:.2f})"
            )
        elif sr["diff"] > config.max_drop and sr["ci_lower"] <= 0:
            sr["verdict"] = "warn"
            has_warn = True
        else:
            sr["verdict"] = "pass"

    # Sort by diff descending (worst first)
    slice_results.sort(key=lambda x: x["diff"], reverse=True)

    duration = time.time() - start

    status: Literal["pass", "warn", "fail", "skip"]
    if has_fail:
        status = "fail"
        summary = "; ".join(failing_slices[:3])
    elif has_warn:
        status = "warn"
        summary = "Point estimate drops in some slices but CIs include zero"
    else:
        status = "pass"
        summary = f"No slice regression detected (overall {config.metric}: {overall_candidate:.3f})"

    return CheckResult(
        name="slices",
        status=status,
        summary=summary,
        metrics={
            f"overall_champion_{config.metric}": overall_champion,
            f"overall_candidate_{config.metric}": overall_candidate,
            "n_slices_checked": float(len(slice_results)),
            "n_failing_slices": float(len(failing_slices)),
        },
        details={
            "slices": slice_results,
            "worst_5": slice_results[:5],
        },
        duration_s=duration,
    )
