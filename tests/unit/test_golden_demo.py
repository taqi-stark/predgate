"""Golden-output regression test verifying deterministic `predgate run` verdict on the demo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from predgate.cli import app

runner = CliRunner()

DEMO_CONFIG = Path("predgate.yaml")


def test_demo_run_matches_golden_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Running `predgate run` on the demo reproduces the golden verdict.json metrics."""
    if not DEMO_CONFIG.exists():
        pytest.skip("Demo config not present at repository root")

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("API_URL", "http://127.0.0.1:1")

    out_dir = tmp_path / "golden_out"
    res = runner.invoke(
        app,
        ["run", "--config", str(DEMO_CONFIG), "--output", str(out_dir)],
    )
    assert res.exit_code == 0, res.stdout

    verdict = json.loads((out_dir / "verdict.json").read_text())
    assert verdict["status"] == "pass"
    assert verdict["summary"] == "8 checks run, 8 passed"
    assert verdict["reasons"] == []
    assert verdict["model_family_change"] == {
        "changed": False,
        "champion_family": "GradientBoostingClassifier",
        "candidate_family": "GradientBoostingClassifier",
        "champion_qualified": "sklearn.GradientBoostingClassifier",
        "candidate_qualified": "sklearn.GradientBoostingClassifier",
        "note": "",
    }

    checks = {cr["name"]: cr for cr in verdict["check_results"]}
    assert list(checks.keys()) == [
        "data_contract",
        "drift",
        "leakage",
        "label_shift",
        "slices",
        "behavioral",
        "budgets",
        "fairness",
    ]
    for name, cr in checks.items():
        assert cr["status"] == "pass", f"Check {name} unexpectedly returned {cr['status']}"

    assert checks["data_contract"]["metrics"] == {"violations": 0}
    assert checks["drift"]["metrics"] == {
        "psi_max": 0.022,
        "max_point_mass_delta": 0.017,
        "features_checked": 9.0,
        "n_new_features": 0.0,
    }
    assert checks["leakage"]["metrics"] == {
        "max_abs_corr": 0.1194,
        "max_single_feature_auc": 0.6378,
        "n_leaky_features": 0.0,
    }
    assert checks["label_shift"]["metrics"] == {
        "overall_label_delta": 0.011,
        "max_slice_label_delta": 0.0386,
        "n_shifted_slices": 0.0,
    }
    assert checks["slices"]["metrics"]["n_slices_checked"] == 9.0
    assert checks["slices"]["metrics"]["n_failing_slices"] == 0.0
    assert checks["slices"]["metrics"]["n_inconclusive_slices"] == 0.0
    assert checks["slices"]["metrics"]["overall_champion_f1"] == pytest.approx(
        0.8792102243088158, rel=1e-9
    )
    assert checks["slices"]["metrics"]["overall_candidate_f1"] == pytest.approx(
        0.8792102243088158, rel=1e-9
    )
    assert checks["behavioral"]["metrics"] == {
        "n_tests": 2.0,
        "n_pass": 2.0,
        "n_fail": 0.0,
        "max_flip_rate": 0.0,
    }
    assert checks["budgets"]["metrics"]["model_size_mb"] == pytest.approx(0.07, abs=0.01)
    assert checks["fairness"]["metrics"]["demographic_parity_diff"] == pytest.approx(
        0.004405286343612335, rel=1e-9
    )
    assert checks["fairness"]["metrics"]["equalized_odds_diff"] == pytest.approx(
        0.004807692307692308, rel=1e-9
    )
    assert checks["fairness"]["metrics"]["min_detectable_gap_dp"] == pytest.approx(0.0485)
    assert checks["fairness"]["metrics"]["min_detectable_gap_eo"] == pytest.approx(0.3119)
