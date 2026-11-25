"""Unit tests for `predgate doctor` diagnostics."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from predgate.cli import app
from predgate.doctor import run_doctor

runner = CliRunner()


def _write_valid_project(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir(exist_ok=True)
    csv_body = "applicant_id,income,default\n1,50000,0\n2,60000,1\n"
    (tmp_path / "data" / "train.csv").write_text(csv_body)
    (tmp_path / "data" / "test.csv").write_text(csv_body)
    (tmp_path / "data" / "reference.csv").write_text(csv_body)

    (tmp_path / "model.py").write_text(
        "def build_model(X_train, y_train, seed=42):\n    return None\n"
    )
    (tmp_path / "schema_mod.py").write_text(
        textwrap.dedent("""\
        try:
            from pandera.pandas import Column, DataFrameSchema
        except ImportError:
            from pandera import Column, DataFrameSchema

        schema = DataFrameSchema({"income": Column(float), "default": Column(int)}, strict=False)
        """)
    )
    cfg_path = tmp_path / "predgate.yaml"
    cfg_path.write_text(
        textwrap.dedent(f"""\
        project: doctor-test
        seed: 42
        model:
          entrypoint: {tmp_path / "model.py"}:build_model
        data:
          train: {tmp_path / "data" / "train.csv"}
          test: {tmp_path / "data" / "test.csv"}
          reference: {tmp_path / "data" / "reference.csv"}
          target: default
          schema: {tmp_path / "schema_mod.py"}:schema
        """)
    )
    return cfg_path


def test_doctor_all_ok(tmp_path: Path) -> None:
    cfg_path = _write_valid_project(tmp_path)
    report = run_doctor(config_path=cfg_path, cwd=tmp_path)
    assert report.ok is True
    by_name = {c.name: c for c in report.checks}
    assert by_name["python"].status == "ok"
    assert by_name["mlflow"].status == "ok"
    assert by_name["config"].status == "ok"
    assert by_name["entrypoint"].status == "ok"
    assert by_name["schema"].status == "ok"
    assert by_name["data_paths"].status == "ok"


def test_doctor_missing_config_and_cli_exit_code(tmp_path: Path) -> None:
    res = runner.invoke(app, ["doctor", "--config", str(tmp_path / "missing.yaml")])
    assert res.exit_code == 1
    assert "predgate init" in res.stdout

    res_json = runner.invoke(app, ["doctor", "--config", str(tmp_path / "missing.yaml"), "--json"])
    assert res_json.exit_code == 1
    parsed = json.loads(res_json.stdout)
    assert parsed["ok"] is False
    assert parsed["n_fail"] >= 1


def test_doctor_bad_schema_bad_entrypoint_and_missing_data(tmp_path: Path) -> None:
    cfg_path = tmp_path / "predgate.yaml"
    cfg_path.write_text(
        textwrap.dedent("""\
        project: broken-test
        model:
          entrypoint: nonexistent_mod:build_model
        data:
          train: does_not_exist_train.csv
          test: does_not_exist_test.csv
          target: default
          schema: nonexistent_schema_mod:schema
        """)
    )
    report = run_doctor(config_path=cfg_path, cwd=tmp_path)
    assert report.ok is False
    by_name = {c.name: c for c in report.checks}
    assert by_name["config"].status == "ok"
    assert by_name["entrypoint"].status == "fail"
    assert by_name["entrypoint"].fix
    assert by_name["schema"].status == "fail"
    assert by_name["schema"].fix
    assert by_name["data_paths"].status == "fail"
    assert by_name["data_paths"].fix


def test_doctor_missing_target_column_and_non_callable_entrypoint(tmp_path: Path) -> None:
    cfg_path = _write_valid_project(tmp_path)
    (tmp_path / "data" / "train.csv").write_text("income,other\n50000,1\n")
    (tmp_path / "model.py").write_text("build_model = 123\n")
    (tmp_path / "schema_mod.py").write_text("schema = 'not-a-schema'\n")

    report = run_doctor(config_path=cfg_path, cwd=tmp_path)
    assert report.ok is False
    by_name = {c.name: c for c in report.checks}
    assert by_name["entrypoint"].status == "fail"
    assert by_name["schema"].status == "fail"
    assert by_name["data_paths"].status == "fail"
    assert "missing target" in by_name["data_paths"].detail


def test_doctor_cli_passes_on_valid_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _write_valid_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, ["doctor", "--config", str(cfg_path)])
    assert res.exit_code == 0
    assert "PredGate Doctor" in res.stdout

    res_json = runner.invoke(app, ["doctor", "--config", str(cfg_path), "--json"])
    assert res_json.exit_code == 0
    assert json.loads(res_json.stdout)["ok"] is True
