"""Unit tests for registry, loader, lineage, and GitHub PR bot integration."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from predgate.config import (
    ChampionConfig,
    DataConfig,
    MLflowConfig,
    ModelConfig,
    PredGateConfig,
)
from predgate.integrations.github.pr_bot import (
    COMMENT_MARKER,
    post_or_update_pr_comment,
    set_commit_status,
    verdict_to_github_state,
)
from predgate.lineage import capture_lineage, load_lineage, save_lineage, verify_reproduction
from predgate.loader import (
    compute_file_hash,
    get_dvc_hash,
    get_git_sha,
    get_package_versions,
    import_entrypoint,
    load_candidate_and_champion,
    load_data,
)
from predgate.registry import (
    _get_active_tracking_uri,
    export_champion_baseline,
    get_reproduce_command,
    import_champion_baseline,
    log_run_to_mlflow,
    promote_to_champion,
    verify_champion_integrity,
)
from predgate.verdict import CheckResult, Verdict


def _make_tiny_project(tmp_path: Path) -> PredGateConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "region": ["North", "South", "North", "South", "East", "West"],
            "default": [0, 1, 0, 1, 0, 1],
        }
    )
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    ref_path = data_dir / "ref.csv"
    df.to_csv(train_path, index=False)
    df.to_csv(test_path, index=False)
    df.to_csv(ref_path, index=False)

    train_script = tmp_path / "tiny_train.py"
    train_script.write_text(
        "from sklearn.dummy import DummyClassifier\n"
        "def build_model(X, y, seed=42):\n"
        "    return DummyClassifier(strategy='most_frequent').fit(X, y)\n"
    )

    return PredGateConfig(
        project="tiny-proj",
        seed=42,
        model=ModelConfig(entrypoint=f"{train_script}:build_model", task="classification"),
        data=DataConfig(
            train=str(train_path),
            test=str(test_path),
            reference=str(ref_path),
            target="default",
        ),
        champion=ChampionConfig(
            mlflow_alias="champion",
            baseline_dir=str(tmp_path / "champion"),
        ),
        mlflow=MLflowConfig(
            tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
            artifact_location=str(tmp_path / "mlruns"),
        ),
    )


def test_tracking_uri_resolution_and_reproduce_cmd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert _get_active_tracking_uri("sqlite:///custom.db") == "sqlite:///custom.db"
    assert _get_active_tracking_uri(None) == "sqlite:///mlflow.db"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert _get_active_tracking_uri("sqlite:///custom.db") == "http://mlflow:5000"
    assert get_reproduce_command("abc123") == "predgate reproduce abc123"


def test_log_run_and_promote_mocked_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_mlflow = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "run-999"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_mv = MagicMock()
    mock_mv.version = "3"
    mock_mlflow.register_model.return_value = mock_mv

    mock_client = MagicMock()
    mock_client.get_experiment_by_name.return_value = None
    mock_client.get_registered_model.side_effect = Exception("not found")
    mock_mlflow.tracking.MlflowClient.return_value = mock_client

    monkeypatch.setitem(__import__("sys").modules, "mlflow", mock_mlflow)
    monkeypatch.setitem(__import__("sys").modules, "mlflow.sklearn", mock_mlflow.sklearn)

    verdict = Verdict(
        status="pass",
        reasons=[],
        check_results=[
            CheckResult(name="drift", status="pass", summary="ok", metrics={"psi_max": 0.01})
        ],
        summary="ok",
    )
    lineage_info = {
        "git_sha": "1234567",
        "dvc_hash": "dvc_abc",
        "data_hashes": {"train": "sha_train"},
    }
    model = DummyClassifier().fit(np.array([[1], [2]]), np.array([0, 1]))

    run_id = log_run_to_mlflow(
        "proj",
        verdict,
        lineage_info,
        model,
        seed=42,
        metrics={"acc": 0.9},
        tracking_uri="http://mlflow:5000",
    )
    assert run_id == "run-999"
    mock_client.create_experiment.assert_called_once_with("proj", artifact_location=None)

    assert promote_to_champion("proj", "run-999", tracking_uri="http://mlflow:5000") is True
    mock_client.create_registered_model.assert_called_once_with("proj")
    mock_client.set_registered_model_alias.assert_called_once_with("proj", "champion", "3")

    # Failure path
    mock_mlflow.register_model.side_effect = RuntimeError("db error")
    assert promote_to_champion("proj", "run-999") is False


def test_export_and_import_champion_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _make_tiny_project(tmp_path)
    champ_dir = tmp_path / "champion"

    # Mock MLflow functions inside registry so export/import don't start real SQLite migrations
    monkeypatch.setattr(
        "predgate.registry.log_run_to_mlflow", lambda *args, **kwargs: "imported-run-1"
    )
    monkeypatch.setattr("predgate.registry.promote_to_champion", lambda *args, **kwargs: True)

    exp_res = export_champion_baseline(cfg, baseline_dir=champ_dir)
    assert (champ_dir / "model.pkl").exists()
    assert (champ_dir / "metadata.json").exists()
    assert (champ_dir / "metrics.json").exists()
    assert (tmp_path / "champion.dvc").exists()
    assert exp_res["metrics"]["n_test_rows"] == 6

    imp_res = import_champion_baseline(cfg, baseline_dir=champ_dir)
    assert imp_res["run_id"] == "imported-run-1"
    assert imp_res["promoted"] is True

    # Tamper with metadata sha256 -> ValueError
    meta = json.loads((champ_dir / "metadata.json").read_text())
    meta["model_sha256"] = "0000000000000000"
    (champ_dir / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        import_champion_baseline(cfg, baseline_dir=champ_dir)

    # Missing model.pkl -> FileNotFoundError
    with pytest.raises(FileNotFoundError):
        import_champion_baseline(cfg, baseline_dir=tmp_path / "nonexistent")


def test_loader_and_lineage_functions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_tiny_project(tmp_path)

    # DVC hash extraction
    dvc_file = tmp_path / "data.dvc"
    dvc_file.write_text("outs:\n- md5: deadbeef1234\n  path: data\n")
    assert get_dvc_hash(dvc_file) == "deadbeef1234"
    assert get_dvc_hash(tmp_path / "missing.dvc") == ""

    # File hash, git sha, package versions
    h = compute_file_hash(cfg.data.train)
    assert len(h) == 64
    assert isinstance(get_git_sha(), str)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: MagicMock(stdout="pytest==8.0.0\nruff==0.5.0\n", returncode=0),
    )
    pkgs = get_package_versions()
    assert pkgs.get("pytest") == "8.0.0"

    # Import entrypoint from file and stdlib module
    fn = import_entrypoint(cfg.model.entrypoint)
    assert callable(fn)
    sqrt_fn = import_entrypoint("math:sqrt")
    assert sqrt_fn(16) == 4.0

    # load_data and load_candidate_and_champion with champion/model.pkl fallback
    train_df, test_df, ref_df = load_data(cfg)
    assert len(train_df) == 6 and ref_df is not None

    champ_dir = Path(cfg.champion.baseline_dir)
    champ_dir.mkdir(parents=True, exist_ok=True)
    model = fn(train_df.drop(columns=["default"]), train_df["default"])
    with open(champ_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    (champ_dir / "metadata.json").write_text(
        json.dumps({"model_sha256": compute_file_hash(champ_dir / "model.pkl")})
    )

    load_res = load_candidate_and_champion(cfg)
    assert load_res.is_first_run is False
    assert load_res.champion_model is not None

    # Lineage capture, save, load, verify_reproduction
    lin = capture_lineage({"train": cfg.data.train}, seed=42, model=model)
    lin_path = tmp_path / "lineage.json"
    save_lineage(lin, lin_path)
    loaded_lin = load_lineage(lin_path)
    assert loaded_lin.seed == 42
    assert "train" in loaded_lin.data_hashes

    ok, msg_ok = verify_reproduction(loaded_lin, 0.9001, 0.9000, tolerance=0.001)
    assert ok is True and "verified" in msg_ok
    bad, msg_bad = verify_reproduction(loaded_lin, 0.85, 0.90, tolerance=0.001)
    assert bad is False and "failed" in msg_bad


def test_github_pr_bot(monkeypatch: pytest.MonkeyPatch) -> None:
    assert verdict_to_github_state("pass") == "success"
    assert verdict_to_github_state("warn") == "success"
    assert verdict_to_github_state("fail") == "failure"
    assert verdict_to_github_state("other") == "error"

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert post_or_update_pr_comment("org/repo", 1, "body", token="") is False
    assert set_commit_status("org/repo", "sha1", "success", token="") is False

    mock_gh_mod = MagicMock()
    mock_pr = MagicMock()
    mock_pr.get_issue_comments.return_value = []
    mock_commit = MagicMock()
    mock_repo = MagicMock()
    mock_repo.get_pull.return_value = mock_pr
    mock_repo.get_commit.return_value = mock_commit
    mock_gh_mod.Github.return_value.get_repo.return_value = mock_repo
    monkeypatch.setitem(__import__("sys").modules, "github", mock_gh_mod)

    # Create new comment
    assert post_or_update_pr_comment("org/repo", 1, "hello", token="tok") is True
    mock_pr.create_issue_comment.assert_called_once()

    # Update existing comment
    existing = MagicMock()
    existing.body = f"{COMMENT_MARKER}\nold"
    mock_pr.get_issue_comments.return_value = [existing]
    assert post_or_update_pr_comment("org/repo", 1, "new", token="tok") is True
    existing.edit.assert_called_once()

    # Set commit status
    assert set_commit_status("org/repo", "sha1", "success", "all good", token="tok") is True
    mock_commit.create_status.assert_called_once()


def test_champion_integrity_rejects_tampering_and_unapproved_pr_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tampered champion model/manifest or unapproved PR edits to champion/ must be rejected."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLOW_CHAMPION_UPDATE", raising=False)
    monkeypatch.delenv("CHAMPION_PINNED_SHA256", raising=False)

    cfg = _make_tiny_project(tmp_path)
    champ_dir = tmp_path / "champion"
    exp_res = export_champion_baseline(cfg, baseline_dir=champ_dir)
    pinned_sha = exp_res["metadata"]["model_sha256"]

    # 1. Valid unmodified champion passes verification with base-branch pinned SHA
    ok = verify_champion_integrity(
        baseline_dir=champ_dir,
        dvc_path=tmp_path / "champion.dvc",
        pinned_sha256=pinned_sha,
        changed_files=["demo/train.py"],
    )
    assert ok["verified"] is True
    assert ok["sha256"] == pinned_sha

    # 2. Unapproved PR modifying champion/ or champion.dvc is rejected
    with pytest.raises(ValueError, match="Unapproved modification of champion baseline"):
        verify_champion_integrity(
            baseline_dir=champ_dir,
            dvc_path=tmp_path / "champion.dvc",
            pinned_sha256=pinned_sha,
            changed_files=["champion/model.pkl"],
        )

    with pytest.raises(ValueError, match="Unapproved modification of champion baseline"):
        verify_champion_integrity(
            baseline_dir=champ_dir,
            dvc_path=tmp_path / "champion.dvc",
            pinned_sha256=pinned_sha,
            changed_files=["champion.dvc"],
        )

    # 3. Custom approval bypass env var (ALLOW_CHAMPION_UPDATE=true) must NOT bypass rejection
    monkeypatch.setenv("ALLOW_CHAMPION_UPDATE", "true")
    with pytest.raises(ValueError, match="Unapproved modification of champion baseline"):
        verify_champion_integrity(
            baseline_dir=champ_dir,
            dvc_path=tmp_path / "champion.dvc",
            pinned_sha256=pinned_sha,
            changed_files=["champion/model.pkl", "champion.dvc"],
        )
    monkeypatch.delenv("ALLOW_CHAMPION_UPDATE", raising=False)

    # 4. Tampering with champion/model.pkl bytes is rejected by verify, import, and loader
    orig_bytes = (champ_dir / "model.pkl").read_bytes()
    (champ_dir / "model.pkl").write_bytes(orig_bytes + b"TAMPERED")
    with pytest.raises(ValueError, match="SHA-256 mismatch against metadata.json"):
        verify_champion_integrity(baseline_dir=champ_dir, pinned_sha256=pinned_sha)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        import_champion_baseline(cfg, baseline_dir=champ_dir, pinned_sha256=pinned_sha)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_candidate_and_champion(cfg)

    # 5. Even if an attacker rewrites BOTH model.pkl AND metadata.json + champion.dvc,
    #    mismatch against base-branch pinned_sha256 still rejects the tampered champion!
    tampered_sha = compute_file_hash(champ_dir / "model.pkl")
    meta = json.loads((champ_dir / "metadata.json").read_text())
    meta["model_sha256"] = tampered_sha
    (champ_dir / "metadata.json").write_text(json.dumps(meta))
    dvc_text = (tmp_path / "champion.dvc").read_text().replace(pinned_sha, tampered_sha)
    (tmp_path / "champion.dvc").write_text(dvc_text)

    with pytest.raises(ValueError, match="SHA-256 mismatch against pinned base-branch value"):
        verify_champion_integrity(
            baseline_dir=champ_dir,
            dvc_path=tmp_path / "champion.dvc",
            pinned_sha256=pinned_sha,
        )
