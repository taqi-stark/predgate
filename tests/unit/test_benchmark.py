"""Unit tests for ModelGate fault injection benchmark."""

from __future__ import annotations

import pandas as pd

from modelgate.benchmark import FAULT_DESCRIPTIONS, _generate_data, inject_fault, run_single_trial


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
    assert "modelgate_detected" in res
    assert "baseline_detected" in res


def test_group_threshold_model_and_markdown_gen(tmp_path, monkeypatch) -> None:
    from rich.console import Console
    from sklearn.linear_model import LogisticRegression

    from modelgate.benchmark import (
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
        "modelgate.benchmark.run_single_trial",
        lambda fault, seed: {
            "fault_type": fault,
            "seed": seed,
            "modelgate_detected": fault != "noop",
            "modelgate_status": "fail" if fault != "noop" else "pass",
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

    # Verify generate_benchmark_markdown_from_json renders public fairness section purely from JSON
    import json

    from modelgate.benchmark import generate_benchmark_markdown_from_json

    json_file = tmp_path / "custom_bench.json"
    md_out = tmp_path / "custom_bench.md"
    payload = {
        "command": "python scripts/bench_public.py",
        "commit_hash": "deadbee",
        "n_seeds": 2,
        "seeds": [1000, 1001],
        "summary": res["summary"],
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
    assert "| **UCI Adult** |" not in rendered
