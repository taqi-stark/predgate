"""Behavioral check.

CheckList-style tests: invariance (perturb irrelevant features),
directional (verify expected behavior changes), and minimum functionality.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from predgate.config import BehavioralConfig
from predgate.verdict import CheckResult


def _perturb_random_id(series: pd.Series, seed: int = 42) -> pd.Series:
    """Replace values with random IDs."""
    rng = np.random.RandomState(seed)
    return pd.Series(rng.randint(0, 1_000_000, size=len(series)), index=series.index)


def _perturb_shuffle(series: pd.Series, seed: int = 42) -> pd.Series:
    """Shuffle values randomly."""
    rng = np.random.RandomState(seed)
    values = series.values.copy()
    rng.shuffle(values)
    return pd.Series(values, index=series.index)


PERTURBATIONS: dict[str, Any] = {
    "random_id": _perturb_random_id,
    "shuffle": _perturb_shuffle,
}


def run_behavioral_check(
    test_df: pd.DataFrame,
    target_column: str,
    model: Any,
    config: BehavioralConfig,
    seed: int = 42,
) -> CheckResult:
    """Run behavioral tests on the candidate model.

    Args:
        test_df: Test DataFrame.
        target_column: Name of the target column.
        model: The candidate model.
        config: Behavioral check configuration.
        seed: Random seed.

    Returns:
        CheckResult with test details.
    """
    start = time.time()

    if not config.enabled:
        return CheckResult(
            name="behavioral",
            status="skip",
            summary="Behavioral check disabled",
            duration_s=time.time() - start,
        )

    X_test = test_df.drop(columns=[target_column])
    results: list[dict] = []
    has_fail = False
    has_warn = False

    # Get baseline predictions
    baseline_preds = model.predict(X_test)

    # Get probability predictions if available
    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        baseline_proba = model.predict_proba(X_test)

    # Invariance tests
    for inv_test in config.invariance:
        if inv_test.feature not in X_test.columns:
            results.append(
                {
                    "type": "invariance",
                    "feature": inv_test.feature,
                    "status": "skip",
                    "message": f"Feature '{inv_test.feature}' not in data",
                }
            )
            continue

        perturb_fn = PERTURBATIONS.get(inv_test.perturb, _perturb_shuffle)

        X_perturbed = X_test.copy()
        X_perturbed[inv_test.feature] = perturb_fn(X_test[inv_test.feature], seed=seed)

        perturbed_preds = model.predict(X_perturbed)
        flip_rate = float(np.mean(baseline_preds != perturbed_preds))

        test_status = "pass"
        if flip_rate > inv_test.max_flip_rate:
            test_status = "fail"
            has_fail = True

        results.append(
            {
                "type": "invariance",
                "feature": inv_test.feature,
                "perturb": inv_test.perturb,
                "flip_rate": round(flip_rate, 4),
                "max_flip_rate": inv_test.max_flip_rate,
                "status": test_status,
            }
        )

    # Directional tests
    for dir_test in config.directional:
        if dir_test.feature not in X_test.columns:
            results.append(
                {
                    "type": "directional",
                    "feature": dir_test.feature,
                    "status": "skip",
                    "message": f"Feature '{dir_test.feature}' not in data",
                }
            )
            continue

        if not has_proba:
            results.append(
                {
                    "type": "directional",
                    "feature": dir_test.feature,
                    "status": "skip",
                    "message": "Model does not support predict_proba",
                }
            )
            continue

        X_modified = X_test.copy()
        col = X_modified[dir_test.feature]

        if pd.api.types.is_numeric_dtype(col):
            if dir_test.direction == "increase":
                X_modified[dir_test.feature] = col + col.std()
            else:
                X_modified[dir_test.feature] = col - col.std()
        else:
            results.append(
                {
                    "type": "directional",
                    "feature": dir_test.feature,
                    "status": "skip",
                    "message": "Directional test requires numeric feature",
                }
            )
            continue

        modified_proba = model.predict_proba(X_modified)

        # Check if probabilities moved in expected direction
        positive_class_idx = 1 if modified_proba.shape[1] > 1 else 0
        diff = modified_proba[:, positive_class_idx] - baseline_proba[:, positive_class_idx]
        if dir_test.expect in ("prob_not_decrease", "prob_increase"):
            violation_rate = float(np.mean(diff < -1e-6))
        elif dir_test.expect in ("prob_not_increase", "prob_decrease"):
            violation_rate = float(np.mean(diff > 1e-6))
        else:
            violation_rate = 0.0
        compliance_rate = 1.0 - violation_rate

        test_status = "pass" if compliance_rate >= 0.85 else "fail"
        if test_status == "fail":
            has_fail = True

        results.append(
            {
                "type": "directional",
                "feature": dir_test.feature,
                "direction": dir_test.direction,
                "expect": dir_test.expect,
                "compliance_rate": round(compliance_rate, 4),
                "status": test_status,
            }
        )

    # Minimum functionality tests from CSV
    if config.min_functionality_csv:
        try:
            mf_df = pd.read_csv(config.min_functionality_csv)
            if target_column in mf_df.columns and "expected" in mf_df.columns:
                X_mf = mf_df.drop(columns=[target_column, "expected"], errors="ignore")
                expected = mf_df["expected"].values
                mf_preds = model.predict(X_mf)

                mismatches = np.where(mf_preds != expected)[0]
                if len(mismatches) > 0:
                    has_fail = True
                    results.append(
                        {
                            "type": "min_functionality",
                            "status": "fail",
                            "n_cases": len(expected),
                            "n_failures": len(mismatches),
                            "failing_indices": mismatches[:10].tolist(),
                        }
                    )
                else:
                    results.append(
                        {
                            "type": "min_functionality",
                            "status": "pass",
                            "n_cases": len(expected),
                            "n_failures": 0,
                        }
                    )
        except Exception as e:
            results.append(
                {
                    "type": "min_functionality",
                    "status": "skip",
                    "message": f"Error loading CSV: {e}",
                }
            )

    duration = time.time() - start
    n_tests = len(results)
    n_pass = sum(1 for r in results if r["status"] == "pass")
    n_fail = sum(1 for r in results if r["status"] == "fail")

    # Find flip rate for invariance tests
    flip_rates = [r.get("flip_rate", 0) for r in results if r["type"] == "invariance"]
    max_flip = max(flip_rates) if flip_rates else 0

    from typing import Literal

    status: Literal["pass", "warn", "fail", "skip"]
    if has_fail:
        status = "fail"
        summary = f"{n_fail}/{n_tests} behavioral tests failed"
    elif has_warn:
        status = "warn"
        summary = "Behavioral tests passed with warnings"
    else:
        status = "pass"
        summary = f"flip rate {max_flip:.1%}" if flip_rates else f"{n_pass}/{n_tests} tests passed"

    return CheckResult(
        name="behavioral",
        status=status,
        summary=summary,
        metrics={
            "n_tests": float(n_tests),
            "n_pass": float(n_pass),
            "n_fail": float(n_fail),
            "max_flip_rate": max_flip,
        },
        details={"tests": results},
        duration_s=duration,
    )
