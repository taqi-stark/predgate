"""Slice regression check.

For each configured slicing column, computes the metric for champion and
candidate on the test set using paired bootstrap for confidence intervals.
Applies Holm-Bonferroni correction across slices.

Gating rules:
- FAIL only when the LOWER bound of the drop CI exceeds ``max_drop``
  (statistically confident regression beyond the allowed drop).
- WARN when the point drop exceeds ``max_drop`` but the interval includes values
  at or below ``max_drop``.
- Mark a slice as ``inconclusive`` only when its CI width exceeds ``max_ci_width``
  AND the interval straddles ``max_drop`` (``ci_lower <= max_drop < ci_upper``).
  A slice whose CI lies entirely beyond ``max_drop`` (``ci_lower > max_drop``) is
  a conclusive regression and is not listed as inconclusive. When
  ``on_inconclusive == "warn"`` (default), inconclusive slices elevate an
  otherwise-passing check to ``WARN``.
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
    """Auto-bin a numeric column into quantile buckets."""
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
    """Run slice regression check comparing candidate to champion."""
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

    overall_champion = metric_fn(y_test, champion_preds)
    overall_candidate = metric_fn(y_test, candidate_preds)

    slice_results: list[dict[str, Any]] = []
    p_values: list[float] = []

    for col in config.by:
        if col not in test_df.columns:
            continue

        series = test_df[col]
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

            champ_metric = float(metric_fn(y_slice, champ_slice))
            cand_metric = float(metric_fn(y_slice, cand_slice))
            point_drop = round(bootstrap.point_estimate, 4)
            ci_lower = round(bootstrap.ci_lower, 4)
            ci_upper = round(bootstrap.ci_upper, 4)
            ci_width = round(bootstrap.ci_upper - bootstrap.ci_lower, 4)
            is_inconclusive = (
                ci_width > config.max_ci_width and ci_lower <= config.max_drop < ci_upper
            )

            slice_results.append(
                {
                    "slice": f"{col}={slice_val}",
                    "column": col,
                    "value": str(slice_val),
                    "n_rows": n_rows,
                    "champion_metric": round(champ_metric, 4),
                    "candidate_metric": round(cand_metric, 4),
                    "diff": point_drop,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "ci_width": ci_width,
                    "inconclusive": is_inconclusive,
                    "p_value": round(bootstrap.p_value, 4),
                }
            )

    rejected = holm_bonferroni(p_values, alpha=1.0 - config.confidence) if p_values else []

    has_fail = False
    has_warn = False
    failing_slices: list[str] = []
    warning_slices: list[str] = []
    inconclusive_slices: list[str] = []
    inconclusive_notes: list[str] = []
    inconclusive_entries: list[dict[str, Any]] = []

    for i, sr in enumerate(slice_results):
        sr["significant"] = rejected[i] if i < len(rejected) else False

        if sr["inconclusive"]:
            inconclusive_slices.append(sr["slice"])
            inconclusive_entries.append(
                {
                    "slice": sr["slice"],
                    "n_rows": sr["n_rows"],
                    "diff": sr["diff"],
                    "ci_lower": sr["ci_lower"],
                    "ci_upper": sr["ci_upper"],
                    "ci_width": sr["ci_width"],
                    "max_ci_width": config.max_ci_width,
                }
            )
            inconclusive_notes.append(
                f"inconclusive interval on '{sr['slice']}' (n={sr['n_rows']}, "
                f"drop {sr['diff']:+.3f}, "
                f"95% CI [{sr['ci_lower']:+.3f}, {sr['ci_upper']:+.3f}], "
                f"width={sr['ci_width']:.3f} > {config.max_ci_width:.3f})"
            )

        # FAIL only when the LOWER bound of the drop CI exceeds max_drop.
        # WARN when the point drop exceeds max_drop but the interval includes values <= max_drop.
        if sr["ci_lower"] > config.max_drop:
            sr["verdict"] = "fail"
            has_fail = True
            failing_slices.append(
                f"{sr['slice']} {config.metric} drop {sr['diff']:+.3f} "
                f"(95% CI [{sr['ci_lower']:+.3f}, {sr['ci_upper']:+.3f}])"
            )
        elif sr["diff"] > config.max_drop:
            sr["verdict"] = "warn"
            has_warn = True
            warning_slices.append(
                f"{sr['slice']} {config.metric} drop {sr['diff']:+.3f} "
                f"(95% CI [{sr['ci_lower']:+.3f}, {sr['ci_upper']:+.3f}])"
            )
        else:
            sr["verdict"] = "pass"

    slice_results.sort(key=lambda x: x["diff"], reverse=True)

    duration = time.time() - start

    status: Literal["pass", "warn", "fail", "skip"]
    if has_fail:
        status = "fail"
        summary = "; ".join(failing_slices[:3])
    elif has_warn:
        status = "warn"
        summary = f"Point drop exceeds {config.max_drop:.2f} (CI overlaps threshold): " + "; ".join(
            warning_slices[:3]
        )
    elif inconclusive_slices and config.on_inconclusive == "warn":
        status = "warn"
        summary = (
            f"Inconclusive slices ({len(inconclusive_slices)} slice(s) CI width > "
            f"{config.max_ci_width:.2f}): " + "; ".join(inconclusive_slices[:3])
        )
    else:
        status = "pass"
        summary = f"No slice regression detected (overall {config.metric}: {overall_candidate:.3f})"

    if inconclusive_slices and not (
        not has_fail and not has_warn and config.on_inconclusive == "warn"
    ):
        summary = (
            f"{summary} [inconclusive: {len(inconclusive_slices)} slice(s) wider than "
            f"{config.max_ci_width:.2f}]"
        )

    return CheckResult(
        name="slices",
        status=status,
        summary=summary,
        metrics={
            f"overall_champion_{config.metric}": overall_champion,
            f"overall_candidate_{config.metric}": overall_candidate,
            "n_slices_checked": float(len(slice_results)),
            "n_failing_slices": float(len(failing_slices)),
            "n_warning_slices": float(len(warning_slices)),
            "n_inconclusive_slices": float(len(inconclusive_slices)),
        },
        details={
            "slices": slice_results,
            "worst_5": slice_results[:5],
            "inconclusive_slices": inconclusive_slices,
            "inconclusive_entries": inconclusive_entries,
            "inconclusive_notes": inconclusive_notes,
            "on_inconclusive": config.on_inconclusive,
        },
        duration_s=duration,
    )
