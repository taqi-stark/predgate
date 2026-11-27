"""Unit tests for PredGate fault injection benchmark."""

from __future__ import annotations

import pandas as pd

from predgate.bench import FAULT_DESCRIPTIONS, _generate_data, inject_fault, run_single_trial


def test_generate_data() -> None:
    train_df, test_df = _generate_data(seed=42, n=100)
    assert len(train_df) == 100
    assert len(test_df) == 50
    assert "income" in train_df.columns
    assert "default" in train_df.columns


def test_inject_faults() -> None:
    train_df, _ = _generate_data(seed=42, n=100)

    for fault in FAULT_DESCRIPTIONS:
        faulty, override = inject_fault(train_df, fault, seed=42)
        assert isinstance(faulty, pd.DataFrame)
        if fault == "feature_leakage":
            assert "leaked" in faulty.columns
        elif fault == "latency_regression":
            assert override == "large_model"


def test_run_single_trial_noop() -> None:
    res = run_single_trial("noop", seed=42)
    assert res["fault_type"] == "noop"
    assert "predgate_detected" in res
    assert "baseline_detected" in res


def test_group_threshold_model_and_markdown_gen(tmp_path, monkeypatch) -> None:
    from rich.console import Console
    from sklearn.linear_model import LogisticRegression

    from predgate.bench import (
        GroupThresholdModel,
        _generate_benchmark_markdown,
        run_benchmark,
        wilson_ci,
    )

    assert wilson_ci(0, 0) == (0.0, 0.0)
    lo, hi = wilson_ci(30, 30)
    assert lo > 0.88 and hi == 1.0

    train_df, test_df = _generate_data(seed=42, n=60)
    X_num = train_df[["income", "age", "credit_score"]]
    clf = LogisticRegression(max_iter=200).fit(X_num, train_df["default"])

    class BaseWrap:
        def predict_proba(self, X: pd.DataFrame):
            return clf.predict_proba(X[["income", "age", "credit_score"]])

    gtm = GroupThresholdModel(BaseWrap(), "gender", "female", 0.05, 0.95)
    assert gtm.predict_proba(test_df).shape == (len(test_df), 2)
    assert len(gtm.predict(test_df)) == len(test_df)
    assert len(gtm.predict(test_df.drop(columns=["gender"]))) == len(test_df)

    md_path = tmp_path / "bench.md"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "predgate.bench.runner.run_single_trial",
        lambda fault, seed: {
            "fault_type": fault,
            "seed": seed,
            "predgate_detected": fault != "noop",
            "predgate_status": "fail" if fault != "noop" else "pass",
            "baseline_detected": False,
            "champion_acc": 0.9,
            "candidate_acc": 0.9,
        },
    )
    res = run_benchmark(Console(), n_seeds=2, start_seed=1000)
    assert res["n_seeds"] == 2
    assert (tmp_path / "docs" / "benchmark.md").exists()
    _generate_benchmark_markdown(res["summary"], md_path, commit_hash="abc1234", seeds=None)
    assert md_path.exists()

    # Verify stronger baselines & JSON rendering of benign-change control and held-out validation
    import json

    import numpy as np

    from predgate.bench import (
        fixed_dp_limit_baseline,
        generate_benchmark_markdown_from_json,
        per_group_accuracy_drop_baseline,
    )

    grp_df = pd.DataFrame({"g": ["a"] * 40 + ["b"] * 40})
    y_g = np.ones(80, dtype=int)
    champ_g = np.ones(80, dtype=int)
    cand_g = np.array([1] * 40 + [0] * 40)
    assert per_group_accuracy_drop_baseline(y_g, champ_g, cand_g, grp_df, ["g"]) is True
    assert per_group_accuracy_drop_baseline(y_g, champ_g, champ_g, grp_df, ["g"]) is False
    assert fixed_dp_limit_baseline(cand_g, grp_df["g"].values, dp_limit=0.10) is True
    assert fixed_dp_limit_baseline(champ_g, grp_df["g"].values, dp_limit=0.10) is False

    # Aggregate repo JSON should NOT embed per-seed details
    repo_json = json.loads((tmp_path / "docs" / "benchmark_results.json").read_text())
    assert "details" not in repo_json
    assert "per_seed_artifact" in repo_json
    assert (tmp_path / repo_json["per_seed_artifact"]).exists()

    json_file = tmp_path / "custom_bench.json"
    md_out = tmp_path / "custom_bench.md"
    payload = {
        "command": "python scripts/bench_public.py",
        "commit_hash": "deadbee",
        "n_seeds": 2,
        "seeds": [1000, 1001],
        "per_seed_artifact": "release_artifacts/benchmark_per_seed_deadbee.json",
        "comparator_config": {"evidently_version": "0.7.23", "pandas_version": "2.3.3"},
        "summary": res["summary"],
        "benign_change_control": {
            "seeds": [9000, 9001],
            "changes": ["hyperparam_change"],
            "overall": {
                "n_trials": 2,
                "fail_count": 0,
                "fail_rate": 0.0,
                "fail_ci_95": [0.0, 0.658],
                "warn_only_count": 1,
                "warn_only_rate": 0.5,
                "warn_only_ci_95": [0.095, 0.905],
                "flagged_count": 1,
                "flagged_rate": 0.5,
                "flagged_ci_95": [0.095, 0.905],
                "quality_fail_count": 0,
                "quality_fail_rate": 0.0,
                "quality_fail_ci_95": [0.0, 0.658],
                "quality_warn_only_count": 1,
                "quality_warn_only_rate": 0.5,
                "quality_warn_only_ci_95": [0.095, 0.905],
                "latency_fail_count": 0,
                "latency_fail_rate": 0.0,
                "latency_fail_ci_95": [0.0, 0.658],
                "latency_warn_count": 0,
                "latency_warn_rate": 0.0,
                "latency_warn_ci_95": [0.0, 0.658],
            },
            "rows": [
                {
                    "dataset": "CustomDS",
                    "change": "hyperparam_change",
                    "n_trials": 2,
                    "fail_count": 0,
                    "fail_rate": 0.0,
                    "fail_ci_95": [0.0, 0.658],
                    "warn_only_count": 1,
                    "warn_only_rate": 0.5,
                    "warn_only_ci_95": [0.095, 0.905],
                    "flagged_count": 1,
                    "flagged_rate": 0.5,
                    "flagged_ci_95": [0.095, 0.905],
                    "quality_fail_count": 0,
                    "quality_fail_rate": 0.0,
                    "quality_fail_ci_95": [0.0, 0.658],
                    "quality_warn_only_count": 1,
                    "quality_warn_only_rate": 0.5,
                    "quality_warn_only_ci_95": [0.095, 0.905],
                    "latency_fail_count": 0,
                    "latency_fail_rate": 0.0,
                    "latency_fail_ci_95": [0.0, 0.658],
                    "latency_warn_count": 0,
                    "latency_warn_rate": 0.0,
                    "latency_warn_ci_95": [0.0, 0.658],
                    "per_group_acc_detected": 0,
                    "per_group_acc_rate": 0.0,
                    "per_group_acc_ci_95": [0.0, 0.658],
                    "fixed_dp_detected": 0,
                    "fixed_dp_rate": 0.0,
                    "fixed_dp_ci_95": [0.0, 0.658],
                }
            ],
            "per_check": [
                {
                    "check": "fairness",
                    "n_trials": 2,
                    "fail_count": 0,
                    "fail_rate": 0.0,
                    "fail_ci_95": [0.0, 0.658],
                    "warn_count": 1,
                    "warn_rate": 0.5,
                    "warn_ci_95": [0.095, 0.905],
                    "flagged_count": 1,
                    "flagged_rate": 0.5,
                    "flagged_ci_95": [0.095, 0.905],
                }
            ],
        },
        "held_out_validation": {
            "seeds": [7000, 7001],
            "rows": [
                {
                    "dataset": "Bank Marketing",
                    "fault": "silent_null_imputation",
                    "fault_label": "`silent_null_imputation`",
                    "n_trials": 2,
                    "detected_fail": 2,
                    "detected_warn": 0,
                    "detected_total": 2,
                    "fail_rate": 1.0,
                    "fail_ci_95": [0.342, 1.0],
                    "detection_rate": 1.0,
                    "ci_95": [0.342, 1.0],
                    "baseline_detected": 0,
                    "baseline_rate": 0.0,
                    "baseline_ci_95": [0.0, 0.658],
                    "per_group_acc_detected": 0,
                    "per_group_acc_rate": 0.0,
                    "per_group_acc_ci_95": [0.0, 0.658],
                    "fixed_dp_detected": 0,
                    "fixed_dp_rate": 0.0,
                    "fixed_dp_ci_95": [0.0, 0.658],
                    "mean_min_n": 180.0,
                    "mean_mde_dp": 0.08,
                    "power_note_count": 2,
                }
            ],
        },
        "public_fairness_benchmark": {
            "command": "python scripts/bench_public.py",
            "commit_hash": "deadbee",
            "seeds": [5000, 5001],
            "frozen_settings": {"max_gap_increase": 0.02, "selection_rate_shift_warn": 0.02},
            "rows": [
                {
                    "dataset": "CustomDS",
                    "fault": "noop",
                    "fault_label": "`noop` (FP rate)",
                    "n_trials": 2,
                    "detected_total": 0,
                    "detected_fail": 0,
                    "detected_warn": 0,
                    "fail_rate": 0.0,
                    "fail_ci_95": [0.0, 0.658],
                    "detection_rate": 0.0,
                    "ci_95": [0.0, 0.658],
                    "baseline_detected": 0,
                    "baseline_rate": 0.0,
                    "baseline_ci_95": [0.0, 0.658],
                    "mean_champ_dp": 0.123,
                    "mean_cand_dp": 0.123,
                    "mean_min_n": 250.0,
                    "mean_mde_dp": 0.055,
                    "mean_mde_eo": 0.111,
                    "power_note_count": 2,
                }
            ],
        },
    }
    json_file.write_text(json.dumps(payload))
    rendered = generate_benchmark_markdown_from_json(json_file, md_out)
    assert "CustomDS" in rendered
    assert "0.123 -> 0.123" in rendered
    assert "5000..5001" in rendered
    assert "Benign-Change Control" in rendered
    assert "Quality FAIL (95% CI)" in rendered
    assert "Latency FAIL (95% CI)" in rendered
    assert "Full Fault Validation" in rendered
    assert "Evidently" in rendered
    assert "All five benchmark datasets" in rendered
    assert "## How to Reproduce" in rendered
    assert "### Exact Evidently Comparator Configuration" in rendered
    assert "ClassificationPreset" in rendered
    assert "| **UCI Adult** |" not in rendered

    # Verify optional preset_benchmark_results.json table renders cleanly when sibling file exists
    preset_json = tmp_path / "preset_benchmark_results.json"
    preset_entry = {
        "n_trials": 10,
        "non_swap_n_trials": 8,
        "quality_fail": {"count": 0, "rate": 0.0, "ci_95": [0.0, 0.278]},
        "quality_warn_only": {"count": 1, "rate": 0.1, "ci_95": [0.018, 0.404]},
        "latency_fail": {"count": 1, "rate": 0.1, "ci_95": [0.018, 0.404]},
        "combined_fail": {"count": 1, "rate": 0.1, "ci_95": [0.018, 0.404]},
        "combined_warn_only": {"count": 1, "rate": 0.1, "ci_95": [0.018, 0.404]},
        "non_swap_quality_fail": {"count": 0, "rate": 0.0, "ci_95": [0.0, 0.324]},
    }
    preset_json.write_text(
        json.dumps(
            {
                "command": "python scripts/bench_presets.py --n-seeds 2 --start-seed 11000",
                "commit_hash": "deadbee",
                "seed_range": "11000..11001",
                "presets": {
                    "strict": preset_entry,
                    "balanced": preset_entry,
                    "quiet": preset_entry,
                },
            }
        )
    )
    rendered_with_presets = generate_benchmark_markdown_from_json(json_file, md_out)
    assert "Config Presets Benign-Change Comparison" in rendered_with_presets
    assert "`strict`" in rendered_with_presets
    assert "`balanced`" in rendered_with_presets
    assert "`quiet`" in rendered_with_presets
