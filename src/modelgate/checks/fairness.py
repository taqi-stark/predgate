"""Fairness check.

Default gate compares champion vs candidate group gaps using a paired bootstrap
confidence interval on the change in gap across demographic parity, equal
opportunity, equalized odds, and per-group calibration disparity.
Absolute limits are opt-in (enforce_absolute=True) and evaluated relative to
the champion's own gap when a champion model is present.
Also computes minimum detectable gap (MDE) at 80% power and emits an
insufficient-power note when any protected subgroup is small.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from modelgate.config import FairnessConfig
from modelgate.stats import (
    calibration_gap_diff,
    demographic_parity_diff,
    equal_opportunity_diff,
    equalized_odds_diff,
    holm_bonferroni,
    paired_bootstrap_gap_ci,
    paired_bootstrap_group_rate_shift_ci,
    per_group_calibration,
)
from modelgate.verdict import CheckResult


def _get_positive_probs(model: Any, X: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    """Return predicted probability of positive class if available, else binary predictions as float."""
    if hasattr(model, "predict_proba"):
        try:
            probs = np.asarray(model.predict_proba(X), dtype=np.float64)
            if probs.ndim == 2 and probs.shape[1] > 1:
                return probs[:, 1]
            return probs.ravel()
        except Exception:
            pass
    return np.asarray(preds, dtype=np.float64)


def _compute_gap_suite(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    protected: np.ndarray,
) -> dict[str, float]:
    """Compute all four fairness gap metrics."""
    return {
        "demographic_parity_diff": demographic_parity_diff(y_pred, protected),
        "equal_opportunity_diff": equal_opportunity_diff(y_true, y_pred, protected),
        "equalized_odds_diff": equalized_odds_diff(y_true, y_pred, protected),
        "calibration_gap_diff": calibration_gap_diff(y_true, y_prob, protected),
    }


def _compute_min_detectable_gaps(
    y_true: np.ndarray,
    protected: np.ndarray,
    confidence: float = 0.95,
    power: float = 0.80,
) -> tuple[int, int, float, float]:
    """Compute (min_group_n, min_group_positives, mde_dp, mde_eo) at given confidence and power."""
    groups = np.unique(protected)
    if len(groups) < 2:
        return len(y_true), int(np.sum(y_true == 1)), 0.0, 0.0

    group_ns = [int(np.sum(protected == g)) for g in groups]
    group_pos = [int(np.sum((protected == g) & (y_true == 1))) for g in groups]
    min_n = min(group_ns)
    max_n = max(group_ns)
    min_pos = min(group_pos)
    max_pos = max(group_pos)

    alpha = 1.0 - confidence
    z_alpha = float(sp_stats.norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(sp_stats.norm.ppf(power))
    z_sum = z_alpha + z_beta

    p_bar = float(np.clip(np.mean(y_true == 1), 0.05, 0.50))
    var_dp = p_bar * (1.0 - p_bar)
    mde_dp = float(z_sum * np.sqrt(var_dp * (1.0 / max(min_n, 1) + 1.0 / max(max_n, 1))))
    # For TPR / equalized odds on positive instances, use worst-case binomial variance 0.25
    mde_eo = float(z_sum * np.sqrt(0.25 * (1.0 / max(min_pos, 1) + 1.0 / max(max_pos, 1))))
    return min_n, min_pos, min(mde_dp, 1.0), min(mde_eo, 1.0)


def run_fairness_check(
    test_df: pd.DataFrame,
    target_column: str,
    candidate_model: Any,
    champion_model: Any | None,
    config: FairnessConfig,
    seed: int = 42,
) -> CheckResult:
    """Run fairness check on the candidate model and compare against champion if available."""
    start = time.time()

    if not config.enabled:
        return CheckResult(
            name="fairness",
            status="skip",
            summary="Fairness check disabled",
            duration_s=time.time() - start,
        )

    if not config.protected:
        return CheckResult(
            name="fairness",
            status="skip",
            summary="No protected attributes configured",
            duration_s=time.time() - start,
        )

    X_test = test_df.drop(columns=[target_column])
    y_test = np.asarray(test_df[target_column].values, dtype=int)

    candidate_preds = np.asarray(candidate_model.predict(X_test), dtype=int)
    candidate_probs = _get_positive_probs(candidate_model, X_test, candidate_preds)

    champion_preds: np.ndarray | None = None
    champion_probs: np.ndarray | None = None
    if champion_model is not None:
        champion_preds = np.asarray(champion_model.predict(X_test), dtype=int)
        champion_probs = _get_positive_probs(champion_model, X_test, champion_preds)

    group_results: list[dict[str, Any]] = []
    max_metric_value = 0.0
    max_effective_absolute_excess = 0.0
    max_eqodds_value = 0.0
    max_cal_gap_value = 0.0
    max_gap_delta = 0.0
    max_selection_rate_shift = 0.0
    max_mde_dp = 0.0
    max_mde_eo = 0.0
    bootstrap_fail_reasons: list[str] = []
    bootstrap_warn_reasons: list[str] = []
    selection_rate_shift_warn_reasons: list[str] = []
    power_notes: list[str] = []

    for idx_col, protected_col in enumerate(config.protected):
        if protected_col not in test_df.columns:
            group_results.append(
                {
                    "protected_attribute": protected_col,
                    "status": "skip",
                    "message": f"Column '{protected_col}' not found",
                }
            )
            continue

        protected_values = test_df[protected_col].values

        cand_suite = _compute_gap_suite(y_test, candidate_preds, candidate_probs, protected_values)
        candidate_value = cand_suite.get(config.metric, cand_suite["demographic_parity_diff"])

        max_metric_value = max(max_metric_value, candidate_value)
        max_eqodds_value = max(max_eqodds_value, cand_suite["equalized_odds_diff"])
        max_cal_gap_value = max(max_cal_gap_value, cand_suite["calibration_gap_diff"])

        min_n, min_pos, mde_dp, mde_eo = _compute_min_detectable_gaps(
            y_test, protected_values, confidence=config.confidence
        )
        max_mde_dp = max(max_mde_dp, mde_dp)
        max_mde_eo = max(max_mde_eo, mde_eo)
        is_low_power = (
            min_n < config.min_group_size
            or min_pos < config.min_group_positives
            or mde_dp > config.fail_gap_increase
        )

        cand_cal_by_group = {
            row["group"]: row
            for row in per_group_calibration(y_test, candidate_probs, protected_values)
        }
        champ_cal_by_group = (
            {
                row["group"]: row
                for row in per_group_calibration(y_test, champion_probs, protected_values)
            }
            if champion_probs is not None
            else {}
        )

        groups = np.unique(protected_values)
        per_group: list[dict[str, Any]] = []
        for g in groups:
            mask = protected_values == g
            n_group = int(np.sum(mask))
            pos_rate = float(np.mean(candidate_preds[mask])) if n_group > 0 else 0.0
            pos_mask = mask & (y_test == 1)
            neg_mask = mask & (y_test == 0)
            tpr_g = float(np.mean(candidate_preds[pos_mask])) if np.sum(pos_mask) > 0 else 0.0
            fpr_g = float(np.mean(candidate_preds[neg_mask])) if np.sum(neg_mask) > 0 else 0.0
            cal_info = cand_cal_by_group.get(str(g), {})

            group_info: dict[str, Any] = {
                "group": str(g),
                "n": n_group,
                "n_positives": int(np.sum(pos_mask)),
                "positive_rate": round(pos_rate, 4),
                "tpr": round(tpr_g, 4),
                "fpr": round(fpr_g, 4),
                "base_rate": cal_info.get("base_rate", 0.0),
                "mean_pred_prob": cal_info.get("mean_pred_prob", round(pos_rate, 4)),
                "calibration_error": cal_info.get("calibration_error", 0.0),
                "brier_score": cal_info.get("brier_score", 0.0),
            }
            if champion_preds is not None:
                champ_rate = float(np.mean(champion_preds[mask])) if n_group > 0 else 0.0
                group_info["champion_positive_rate"] = round(champ_rate, 4)
                champ_cal = champ_cal_by_group.get(str(g), {})
                if champ_cal:
                    group_info["champion_mean_pred_prob"] = champ_cal["mean_pred_prob"]
                    group_info["champion_calibration_error"] = champ_cal["calibration_error"]
            per_group.append(group_info)

        result_entry: dict[str, Any] = {
            "protected_attribute": protected_col,
            "metric": config.metric,
            "candidate_value": round(candidate_value, 4),
            "equalized_odds_diff": round(cand_suite["equalized_odds_diff"], 4),
            "calibration_gap_diff": round(cand_suite["calibration_gap_diff"], 4),
            "candidate_gaps": {k: round(v, 4) for k, v in cand_suite.items()},
            "min_group_n": min_n,
            "min_group_positives": min_pos,
            "min_detectable_gap_dp": round(mde_dp, 4),
            "min_detectable_gap_eo": round(mde_eo, 4),
            "insufficient_power": is_low_power,
            "per_group": per_group,
        }
        if is_low_power:
            note = (
                f"minimum detectable gap / insufficient power on '{protected_col}': "
                f"min_group_n={min_n}, min_positives={min_pos}, "
                f"MDE_DP={mde_dp:.3f}, MDE_EO={mde_eo:.3f} at 80% power"
            )
            result_entry["power_note"] = note
            power_notes.append(note)

        if champion_preds is not None and champion_probs is not None:
            champ_suite = _compute_gap_suite(
                y_test, champion_preds, champion_probs, protected_values
            )
            champion_value = champ_suite.get(config.metric, champ_suite["demographic_parity_diff"])
            relative_excess = max(0.0, candidate_value - champion_value)
            max_effective_absolute_excess = max(max_effective_absolute_excess, relative_excess)
            result_entry["champion_value"] = round(champion_value, 4)
            result_entry["delta"] = round(candidate_value - champion_value, 4)
            result_entry["relative_excess_over_champion"] = round(relative_excess, 4)
            result_entry["champion_gaps"] = {k: round(v, 4) for k, v in champ_suite.items()}

            monitored_metrics = list(
                dict.fromkeys(
                    [
                        config.metric,
                        "demographic_parity_diff",
                        "equal_opportunity_diff",
                        "equalized_odds_diff",
                        "calibration_gap_diff",
                    ]
                )
            )
            boot_details: dict[str, dict[str, float]] = {}
            for m_idx, m_name in enumerate(monitored_metrics):
                a_scores = champion_probs if m_name == "calibration_gap_diff" else champion_preds
                b_scores = candidate_probs if m_name == "calibration_gap_diff" else candidate_preds
                boot = paired_bootstrap_gap_ci(
                    y_test,
                    a_scores,
                    b_scores,
                    protected_values,
                    metric=m_name,
                    n_bootstrap=config.n_bootstrap,
                    confidence=config.confidence,
                    seed=seed + idx_col * 17 + m_idx,
                )
                boot_details[m_name] = {
                    "point_estimate": round(boot.point_estimate, 4),
                    "ci_lower": round(boot.ci_lower, 4),
                    "ci_upper": round(boot.ci_upper, 4),
                    "p_value": round(boot.p_value, 4),
                }
                max_gap_delta = max(max_gap_delta, boot.point_estimate)

                if boot.ci_lower > 0.0 and (
                    boot.point_estimate >= config.fail_gap_increase
                    or boot.ci_lower >= config.max_gap_increase
                ):
                    bootstrap_fail_reasons.append(
                        f"{protected_col} {m_name} gap +{boot.point_estimate:.3f} "
                        f"(95% CI [+{boot.ci_lower:.3f}, +{boot.ci_upper:.3f}])"
                    )
                elif boot.ci_lower > 0.0 and boot.point_estimate >= config.max_gap_increase:
                    bootstrap_warn_reasons.append(
                        f"{protected_col} {m_name} gap +{boot.point_estimate:.3f} "
                        f"(95% CI [+{boot.ci_lower:.3f}, +{boot.ci_upper:.3f}])"
                    )

            result_entry["gap_bootstrap"] = boot_details[config.metric]
            result_entry["all_gap_bootstraps"] = boot_details

            # Per-group selection-rate shift check (two-sided, paired bootstrap, Holm-Bonferroni, WARN level)
            shift_entries: list[dict[str, Any]] = []
            shift_pvals: list[float] = []
            for g_idx, g in enumerate(groups):
                shift_boot = paired_bootstrap_group_rate_shift_ci(
                    champion_preds,
                    candidate_preds,
                    protected_values,
                    g,
                    n_bootstrap=config.selection_rate_n_bootstrap,
                    confidence=config.confidence,
                    seed=seed + idx_col * 31 + g_idx,
                )
                shift_pvals.append(shift_boot.p_value)
                max_selection_rate_shift = max(
                    max_selection_rate_shift, abs(shift_boot.point_estimate)
                )
                shift_entries.append(
                    {
                        "group": str(g),
                        "delta": round(shift_boot.point_estimate, 4),
                        "ci_lower": round(shift_boot.ci_lower, 4),
                        "ci_upper": round(shift_boot.ci_upper, 4),
                        "p_value": round(shift_boot.p_value, 4),
                    }
                )

            rejected_shifts = (
                holm_bonferroni(shift_pvals, alpha=1.0 - config.confidence) if shift_pvals else []
            )
            for g_idx, s_entry in enumerate(shift_entries):
                is_sig = rejected_shifts[g_idx] if g_idx < len(rejected_shifts) else False
                ci_excl_zero = s_entry["ci_lower"] > 0.0 or s_entry["ci_upper"] < 0.0
                triggers_warn = (
                    is_sig
                    and ci_excl_zero
                    and abs(s_entry["delta"]) >= config.selection_rate_shift_warn
                )
                s_entry["significant"] = is_sig
                s_entry["warn"] = triggers_warn
                if g_idx < len(per_group):
                    per_group[g_idx]["selection_rate_shift"] = s_entry
                if triggers_warn:
                    selection_rate_shift_warn_reasons.append(
                        f"{protected_col}={s_entry['group']} selection rate {s_entry['delta']:+.3f} "
                        f"(95% CI [{s_entry['ci_lower']:+.3f}, {s_entry['ci_upper']:+.3f}])"
                    )

            result_entry["selection_rate_shifts"] = shift_entries
        else:
            max_effective_absolute_excess = max(max_effective_absolute_excess, candidate_value)

        group_results.append(result_entry)

    duration = time.time() - start

    # Default gate when champion is present: paired bootstrap CI on gap change.
    # Absolute DP limit is opt-in (config.enforce_absolute) and relative to champion's gap when champion is present,
    # or evaluated on standalone candidate when no champion is provided.
    # Per-group two-sided selection-rate shift operates at WARN level.
    check_abs = (champion_model is None) or config.enforce_absolute
    status: Literal["pass", "warn", "fail", "skip"]
    if bootstrap_fail_reasons:
        status = "fail"
        summary = f"Fairness gap regression: {'; '.join(bootstrap_fail_reasons[:2])}"
    elif check_abs and max_effective_absolute_excess >= config.fail_at:
        status = "fail"
        summary = f"{config.metric} {max_effective_absolute_excess:.2f} >= {config.fail_at}"
    elif bootstrap_warn_reasons:
        status = "warn"
        summary = f"Fairness gap increase: {'; '.join(bootstrap_warn_reasons[:2])}"
    elif selection_rate_shift_warn_reasons:
        status = "warn"
        summary = f"Selection-rate shift: {'; '.join(selection_rate_shift_warn_reasons[:2])}"
    elif check_abs and max_effective_absolute_excess >= config.warn_at:
        status = "warn"
        summary = f"{config.metric} {max_effective_absolute_excess:.2f} >= {config.warn_at}"
    else:
        status = "pass"
        summary = (
            f"{config.metric} {max_metric_value:.2f}, "
            f"eq_odds {max_eqodds_value:.2f}, cal_gap {max_cal_gap_value:.2f}"
        )

    if power_notes:
        summary = f"{summary} [note: {power_notes[0]}]"

    return CheckResult(
        name="fairness",
        status=status,
        summary=summary,
        metrics={
            config.metric: max_metric_value,
            "equalized_odds_diff": max_eqodds_value,
            "calibration_gap_diff": max_cal_gap_value,
            "max_gap_delta": max_gap_delta,
            "max_selection_rate_shift": round(max_selection_rate_shift, 4),
            "min_detectable_gap_dp": round(max_mde_dp, 4),
            "min_detectable_gap_eo": round(max_mde_eo, 4),
            "insufficient_power": 1.0 if power_notes else 0.0,
        },
        details={
            "groups": group_results,
            "power_notes": power_notes,
            "selection_rate_shift_warnings": selection_rate_shift_warn_reasons,
            "enforce_absolute": config.enforce_absolute,
        },
        duration_s=duration,
    )
