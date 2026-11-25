"""Unit tests for PredGate CLI."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from predgate.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "CI/CD quality gate for machine learning models" in result.stdout


def test_cli_init() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(app, ["init", tmpdir])
        assert result.exit_code == 0
        assert (Path(tmpdir) / "predgate.yaml").exists()
        assert (Path(tmpdir) / "demo" / "train.py").exists()

        # Running again reports already exists
        result2 = runner.invoke(app, ["init", tmpdir])
        assert result2.exit_code == 0
        assert "already exists" in result2.stdout


def test_cli_run_missing_config() -> None:
    result = runner.invoke(app, ["run", "--config", "nonexistent_config.yaml"])
    assert result.exit_code == 1
    assert "Config file not found" in result.stdout


def test_cli_run_success(monkeypatch) -> None:
    # Test predgate run with a passing mock
    from predgate.loader import LineageInfo, LoadResult
    from predgate.verdict import CheckResult, Verdict

    dummy_lineage = LineageInfo(
        git_sha="abcdef123456",
        data_hashes={"train": "hash123"},
        dvc_hash="dvc999",
        python_version="3.11.0",
        package_versions={},
        seed=42,
    )
    dummy_verdict = Verdict(
        status="pass",
        reasons=[],
        check_results=[
            CheckResult(name="data_contract", status="pass", summary="Passed"),
            CheckResult(name="drift", status="pass", summary="Passed"),
            CheckResult(name="slices", status="pass", summary="Passed"),
            CheckResult(name="behavioral", status="pass", summary="Passed"),
            CheckResult(name="budgets", status="pass", summary="Passed"),
            CheckResult(name="fairness", status="pass", summary="Passed"),
        ],
        summary="All checks passed",
    )

    dummy_load_result = LoadResult(
        candidate_model=MagicMock(),
        champion_model=MagicMock(),
        train_df=MagicMock(),
        test_df=MagicMock(),
        reference_df=MagicMock(),
        target_column="default",
        train_time_s=0.5,
        lineage=dummy_lineage,
        is_first_run=False,
    )

    monkeypatch.setattr(
        "predgate.loader.load_candidate_and_champion", lambda cfg: dummy_load_result
    )
    monkeypatch.setattr(
        "predgate.checks.data_contract.run_data_contract_check",
        lambda *args, **kwargs: dummy_verdict.check_results[0],
    )
    monkeypatch.setattr(
        "predgate.checks.drift.run_drift_check",
        lambda *args, **kwargs: dummy_verdict.check_results[1],
    )
    monkeypatch.setattr(
        "predgate.checks.leakage.run_leakage_check",
        lambda *args, **kwargs: CheckResult(name="leakage", status="pass", summary="Passed"),
    )
    monkeypatch.setattr(
        "predgate.checks.label_shift.run_label_shift_check",
        lambda *args, **kwargs: CheckResult(name="label_shift", status="pass", summary="Passed"),
    )
    monkeypatch.setattr(
        "predgate.checks.slices.run_slices_check",
        lambda *args, **kwargs: dummy_verdict.check_results[2],
    )
    monkeypatch.setattr(
        "predgate.checks.behavioral.run_behavioral_check",
        lambda *args, **kwargs: dummy_verdict.check_results[3],
    )
    monkeypatch.setattr(
        "predgate.checks.budgets.run_budgets_check",
        lambda *args, **kwargs: dummy_verdict.check_results[4],
    )
    monkeypatch.setattr(
        "predgate.checks.fairness.run_fairness_check",
        lambda *args, **kwargs: dummy_verdict.check_results[5],
    )
    monkeypatch.setattr(
        "predgate.verdict.compute_verdict", lambda results, cfg=None, **kw: dummy_verdict
    )
    monkeypatch.setattr(
        "predgate.registry.log_run_to_mlflow", lambda *args, **kwargs: "run-test-123"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "output"
        result = runner.invoke(app, ["run", "--config", "predgate.yaml", "--output", str(out_dir)])
        assert result.exit_code == 0
        assert (out_dir / "verdict.json").exists()
        assert (out_dir / "pr_comment.md").exists()
        assert (out_dir / "report.html").exists()
        assert "Verdict:" in result.stdout


def test_cli_compare() -> None:
    # When API is not running, falls back cleanly
    result = runner.invoke(app, ["compare", "run_1", "run_2"])
    assert result.exit_code == 0
    assert "Comparing runs" in result.stdout


def test_cli_promote_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "predgate.registry.promote_to_champion",
        lambda project, run_id, alias="champion", tracking_uri=None: True,
    )
    result = runner.invoke(app, ["promote", "run-123"])
    assert result.exit_code == 0
    assert "promoted to champion" in result.stdout


def test_cli_promote_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "predgate.registry.promote_to_champion",
        lambda project, run_id, alias="champion", tracking_uri=None: False,
    )
    result = runner.invoke(app, ["promote", "run-failed"])
    assert result.exit_code == 1
    assert "Failed to promote" in result.stdout


def test_cli_reproduce_success(monkeypatch) -> None:
    mock_run = MagicMock()
    mock_run.data.params = {"seed": "42", "git_sha": "abc1234", "dvc_hash": "dvc98765"}
    mock_run.data.metrics = {"accuracy": 0.88}

    mock_client = MagicMock()
    mock_client.get_run.return_value = mock_run

    mock_mlflow = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = mock_client
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda *args, **kwargs: mock_client)
    monkeypatch.setattr("mlflow.set_tracking_uri", lambda *args, **kwargs: None)

    result = runner.invoke(app, ["reproduce", "run-123"])
    assert result.exit_code == 0
    assert "Reproducing run" in result.stdout
    assert "Seed: 42" in result.stdout


def test_cli_run_fail_verdict_and_invalid_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from predgate.loader import LineageInfo, LoadResult
    from predgate.verdict import CheckResult, Reason, Verdict

    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("seed: -5\n")
    res_bad = runner.invoke(app, ["run", "--config", str(bad_cfg)])
    assert res_bad.exit_code == 1

    # Load failure
    monkeypatch.setattr(
        "predgate.loader.load_candidate_and_champion",
        lambda cfg: (_ for _ in ()).throw(RuntimeError("load crash")),
    )
    res_crash = runner.invoke(app, ["run", "--config", "predgate.yaml"])
    assert res_crash.exit_code == 1

    # Failing verdict with reasons and API post
    dummy_lineage = LineageInfo(git_sha="abc", data_hashes={"train": "h"}, seed=42)
    fail_verdict = Verdict(
        status="fail",
        reasons=[Reason(check_name="drift", severity="fail", message="PSI high")],
        check_results=[CheckResult(name="drift", status="fail", summary="PSI high")],
        summary="1 check failed",
    )
    dummy_load = LoadResult(
        candidate_model=MagicMock(),
        champion_model=None,
        train_df=MagicMock(),
        test_df=MagicMock(),
        reference_df=MagicMock(),
        target_column="default",
        train_time_s=0.1,
        lineage=dummy_lineage,
        is_first_run=True,
    )
    monkeypatch.setattr("predgate.loader.load_candidate_and_champion", lambda cfg: dummy_load)
    for mod_fn in [
        "predgate.checks.data_contract.run_data_contract_check",
        "predgate.checks.drift.run_drift_check",
        "predgate.checks.leakage.run_leakage_check",
        "predgate.checks.label_shift.run_label_shift_check",
        "predgate.checks.slices.run_slices_check",
        "predgate.checks.behavioral.run_behavioral_check",
        "predgate.checks.budgets.run_budgets_check",
        "predgate.checks.fairness.run_fairness_check",
    ]:
        monkeypatch.setattr(
            mod_fn, lambda *a, **kw: CheckResult(name="c", status="fail", summary="bad")
        )
    monkeypatch.setattr(
        "predgate.verdict.compute_verdict", lambda results, cfg=None, **kw: fail_verdict
    )
    monkeypatch.setattr("predgate.registry.log_run_to_mlflow", lambda *a, **kw: "run-fail-1")
    monkeypatch.setattr("httpx.post", lambda *a, **kw: MagicMock(status_code=200))

    out_dir = tmp_path / "fail_out"
    res_fail = runner.invoke(app, ["run", "--config", "predgate.yaml", "--output", str(out_dir)])
    assert res_fail.exit_code == 1
    assert "FAIL" in res_fail.stdout


def test_cli_compare_and_promote_api_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "httpx.get",
        lambda url, **kw: MagicMock(
            status_code=200,
            json=lambda: {
                "metrics": [{"name": "acc", "value_a": 0.9, "value_b": 0.92, "delta": 0.02}]
            },
        ),
    )
    res_cmp = runner.invoke(app, ["compare", "1", "2"])
    assert res_cmp.exit_code == 0
    assert "acc" in res_cmp.stdout

    monkeypatch.setattr("httpx.get", lambda url, **kw: MagicMock(status_code=500))
    res_cmp_err = runner.invoke(app, ["compare", "1", "2"])
    assert res_cmp_err.exit_code == 0
    assert "API error" in res_cmp_err.stdout


def test_cli_bench_serve_baseline_and_users(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("predgate.benchmark.run_benchmark", lambda console: {})
    assert runner.invoke(app, ["bench"]).exit_code == 0

    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: None)
    assert runner.invoke(app, ["serve", "--port", "8123"]).exit_code == 0

    monkeypatch.setattr(
        "predgate.registry.export_champion_baseline",
        lambda cfg, baseline_dir="champion": {
            "baseline_dir": str(baseline_dir),
            "metadata": {"source": "mlflow", "model_sha256": "abcdef1234567890"},
            "metrics": {"accuracy": 0.91, "f1": 0.88},
        },
    )
    assert runner.invoke(app, ["baseline", "export"]).exit_code == 0

    monkeypatch.setattr(
        "predgate.registry.import_champion_baseline",
        lambda cfg, baseline_dir="champion": {
            "baseline_dir": str(baseline_dir),
            "run_id": "r1",
            "promoted": True,
            "metrics": {"accuracy": 0.91, "f1": 0.88},
        },
    )
    assert runner.invoke(app, ["baseline", "import"]).exit_code == 0

    # baseline verify (valid and tampered)
    monkeypatch.setattr(
        "predgate.registry.verify_champion_integrity",
        lambda **kw: {
            "verified": True,
            "baseline_dir": str(kw.get("baseline_dir", "champion")),
            "sha256": "abcdef1234567890",
        },
    )
    assert (
        runner.invoke(app, ["baseline", "verify", "--pinned-sha256", "abcdef1234567890"]).exit_code
        == 0
    )

    def _fail_verify(**kw: object) -> dict[str, object]:
        raise ValueError("Champion model SHA-256 mismatch")

    monkeypatch.setattr("predgate.registry.verify_champion_integrity", _fail_verify)
    res_bad_verify = runner.invoke(
        app, ["baseline", "verify", "--changed-file", "champion/model.pkl"]
    )
    assert res_bad_verify.exit_code == 1
    assert "verification failed" in res_bad_verify.stdout.lower()

    # users add / list
    assert runner.invoke(app, ["users", "add", "bad_user", "--role", "superadmin"]).exit_code == 1
    assert runner.invoke(app, ["users", "add", "cli_eng", "--role", "engineer"]).exit_code == 0
    res_list = runner.invoke(app, ["users", "list"])
    assert res_list.exit_code == 0
    assert "cli_eng" in res_list.stdout


def test_cli_run_preset_and_model_family_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify --preset CLI override and 'Model family changed' note across CLI & report outputs."""
    import json

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    from predgate.loader import LineageInfo, LoadResult

    monkeypatch.chdir(tmp_path)
    (tmp_path / "predgate.yaml").write_text(
        "project: fam-test\n"
        "seed: 42\n"
        "model:\n"
        "  entrypoint: demo/train.py:build_model\n"
        "data:\n"
        "  train: train.csv\n"
        "  test: test.csv\n"
        "  target: default\n"
        "checks:\n"
        "  data_contract:\n"
        "    enabled: false\n"
        "  behavioral:\n"
        "    enabled: false\n"
        "  budgets:\n"
        "    enabled: false\n"
        "  fairness:\n"
        "    enabled: false\n"
        "  slices:\n"
        "    enabled: false\n"
    )
    rng = np.random.RandomState(42)
    x_vals = rng.normal(0.0, 1.0, size=60)
    y_vals = (x_vals + rng.normal(0.0, 1.5, size=60) > 0).astype(int)
    df = pd.DataFrame({"x": x_vals, "default": y_vals})
    champ = GradientBoostingClassifier(n_estimators=5, random_state=42).fit(
        df[["x"]], df["default"]
    )
    cand = RandomForestClassifier(n_estimators=5, random_state=42).fit(df[["x"]], df["default"])

    monkeypatch.setattr(
        "predgate.loader.load_candidate_and_champion",
        lambda cfg: LoadResult(
            candidate_model=cand,
            champion_model=champ,
            train_df=df,
            test_df=df,
            reference_df=df,
            target_column="default",
            train_time_s=0.05,
            lineage=LineageInfo(git_sha="abc1234", seed=42),
            is_first_run=False,
        ),
    )
    monkeypatch.setattr("predgate.registry.log_run_to_mlflow", lambda *a, **kw: "run-fam-1")
    monkeypatch.setattr("httpx.post", lambda *a, **kw: MagicMock(status_code=200))

    out_dir = tmp_path / "fam_out"
    res = runner.invoke(
        app,
        ["run", "--config", "predgate.yaml", "--preset", "quiet", "--output", str(out_dir)],
    )
    assert res.exit_code == 0
    assert "Preset: quiet" in res.stdout
    assert "Model family changed" in res.stdout

    verdict_json = json.loads((out_dir / "verdict.json").read_text())
    assert verdict_json["model_family_change"]["changed"] is True
    assert verdict_json["model_family_change"]["champion_family"] == "GradientBoostingClassifier"
    assert verdict_json["model_family_change"]["candidate_family"] == "RandomForestClassifier"
    assert "Model family changed" in (out_dir / "pr_comment.md").read_text()
    assert "Model family changed" in (out_dir / "report.html").read_text()
