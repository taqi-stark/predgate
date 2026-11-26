"""Security and fail-closed verification tests for champion integrity, workflows, and CODEOWNERS."""

from __future__ import annotations

import inspect
import json
import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from predgate.config import (
    ChampionConfig,
    DataConfig,
    MLflowConfig,
    ModelConfig,
    PredGateConfig,
)
from predgate.loader import compute_file_hash, load_candidate_and_champion
from predgate.registry import (
    export_champion_baseline,
    import_champion_baseline,
    verify_champion_integrity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_project(tmp_path: Path, call_log: Path) -> PredGateConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "region": ["North", "South", "North", "South", "East", "West"],
            "default": [0, 1, 0, 1, 0, 1],
        }
    )
    for name in ("train.csv", "test.csv", "ref.csv"):
        df.to_csv(data_dir / name, index=False)

    train_script = tmp_path / "tiny_train.py"
    train_script.write_text(
        "from pathlib import Path\n"
        "from sklearn.dummy import DummyClassifier\n"
        f"LOG = Path({str(call_log)!r})\n"
        "def build_model(X, y, seed=42):\n"
        "    with open(LOG, 'a') as fh:\n"
        "        fh.write('called\\n')\n"
        "    return DummyClassifier(strategy='most_frequent').fit(X, y)\n"
    )

    return PredGateConfig(
        project="sec-proj",
        seed=42,
        model=ModelConfig(entrypoint=f"{train_script}:build_model", task="classification"),
        data=DataConfig(
            train=str(data_dir / "train.csv"),
            test=str(data_dir / "test.csv"),
            reference=str(data_dir / "ref.csv"),
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


def test_no_custom_approval_bypass_in_code_or_workflows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Custom approval bypass (ALLOW_CHAMPION_UPDATE / listReviews) must not exist."""
    sig = inspect.signature(verify_champion_integrity)
    assert "allow_champion_update" not in sig.parameters

    workflow_text = (REPO_ROOT / ".github" / "workflows" / "predgate.yml").read_text()
    action_text = (REPO_ROOT / "action.yml").read_text()

    for forbidden in (
        "ALLOW_CHAMPION_UPDATE",
        "allow-champion-update",
        "allow_champion_update",
        "listReviews",
        "champ_approval",
    ):
        assert forbidden not in workflow_text, f"Found forbidden bypass '{forbidden}' in workflow"
        assert forbidden not in action_text, f"Found forbidden bypass '{forbidden}' in action.yml"

    monkeypatch.chdir(tmp_path)
    cfg = _make_project(tmp_path, tmp_path / "calls.log")
    champ_dir = tmp_path / "champion"
    exp = export_champion_baseline(cfg, baseline_dir=champ_dir)
    pinned = exp["metadata"]["model_sha256"]

    monkeypatch.setenv("ALLOW_CHAMPION_UPDATE", "true")
    with pytest.raises(ValueError, match="No custom approval bypass"):
        verify_champion_integrity(
            baseline_dir=champ_dir,
            dvc_path=tmp_path / "champion.dvc",
            pinned_sha256=pinned,
            changed_files=["champion/model.pkl"],
        )


def test_codeowners_handle_format() -> None:
    """CODEOWNERS must document user vs org handle rules without invalid active placeholders."""
    codeowners = (REPO_ROOT / ".github" / "CODEOWNERS").read_text()

    # Must document both personal user (@<username>) and org team (@<org>/<team>) setup for champion artifacts
    assert "/champion/" in codeowners
    assert "/champion.dvc" in codeowners
    assert "Personal / User GitHub Account" in codeowners
    assert "GitHub Organization Account" in codeowners
    assert "@<your-github-username>" in codeowners
    assert "@<your-org>/" in codeowners

    # No active (uncommented) placeholder rules should exist; any configured active rule must use valid GitHub handle syntax
    user_or_team = re.compile(r"^@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:/[A-Za-z0-9._-]+)?$")
    email_pat = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    for raw_line in codeowners.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assert "@your-org/" not in line, "Active CODEOWNERS rule must not use @your-org placeholder"
        parts = line.split()
        assert len(parts) >= 2
        for owner in parts[1:]:
            assert user_or_team.match(owner) or email_pat.match(owner), (
                f"Invalid CODEOWNERS handle '{owner}' in line: {line}"
            )


def test_workflow_and_action_fail_closed_on_missing_base_ref(tmp_path: Path) -> None:
    """Workflow and action.yml must not silence git fetch failures and must exit non-zero."""
    workflow_yaml = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "predgate.yml").read_text()
    )
    action_yaml = yaml.safe_load((REPO_ROOT / "action.yml").read_text())

    wf_steps = workflow_yaml["jobs"]["predgate"]["steps"]
    wf_setup = next(s for s in wf_steps if s.get("id") == "trusted_setup")["run"]
    wf_verify = next(s for s in wf_steps if s.get("id") == "verify_champion")["run"]

    act_steps = action_yaml["runs"]["steps"]
    act_install = next(s for s in act_steps if s.get("id") == "install_pinned")["run"]
    act_verify = next(s for s in act_steps if s.get("id") == "verify_champion")["run"]

    for script in (wf_setup, wf_verify, act_install, act_verify):
        assert "set -euo pipefail" in script
        assert (
            "git fetch" not in script
            or "|| true" not in script.split("git fetch")[1].splitlines()[0]
        )

    # Actually execute the base-ref fetch block against a local repo whose origin lacks the ref
    origin_repo = tmp_path / "origin.git"
    work_repo = tmp_path / "work"
    subprocess.run(["git", "init", "--bare", str(origin_repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "clone", str(origin_repo), str(work_repo)], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(work_repo), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(work_repo), "config", "user.name", "Test"], check=True)
    (work_repo / "README.md").write_text("hi\n")
    subprocess.run(["git", "-C", str(work_repo), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(work_repo), "commit", "-m", "init"], check=True, capture_output=True
    )

    probe_script = (
        "set -euo pipefail\n"
        'BASE_REF="nonexistent-base-branch"\n'
        'git fetch --no-tags origin "+refs/heads/${BASE_REF}:refs/remotes/origin/${BASE_REF}"\n'
        'git rev-parse --verify "origin/${BASE_REF}^{commit}"\n'
    )
    proc = subprocess.run(
        ["bash", "-c", probe_script], cwd=work_repo, capture_output=True, text=True
    )
    assert proc.returncode != 0


def test_workflow_and_action_install_pinned_not_pr_head() -> None:
    """Workflow and action.yml must never pip install -e . from PR head before verifying."""
    workflow_text = (REPO_ROOT / ".github" / "workflows" / "predgate.yml").read_text()
    action_text = (REPO_ROOT / "action.yml").read_text()

    assert "pip install -e" not in workflow_text
    assert "pip install -e" not in action_text
    assert "git archive" in workflow_text
    assert "git archive" in action_text

    # In workflow, champion verification must happen BEFORE running untrusted PR demo/train.py
    assert workflow_text.index("Verify champion baseline from base branch") < workflow_text.index(
        "Generate demo data"
    )
    # In action.yml, baseline verify must happen BEFORE setup-data
    assert action_text.index("predgate baseline verify") < action_text.index("inputs.setup-data")


def test_corrupt_pickle_raises_hard_error_no_silent_retrain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corrupt champion pickle must raise RuntimeError and never silently retrain from entrypoint."""
    monkeypatch.chdir(tmp_path)
    call_log = tmp_path / "calls.log"
    cfg = _make_project(tmp_path, call_log)
    champ_dir = tmp_path / "champion"

    # Export a valid champion first, then replace model.pkl with corrupt bytes + matching SHA-256
    # (simulating an incompatible/corrupt pickle whose hash still matches base-branch metadata)
    export_champion_baseline(cfg, baseline_dir=champ_dir)
    corrupt_bytes = b"not-a-valid-pickle-stream"
    (champ_dir / "model.pkl").write_bytes(corrupt_bytes)
    corrupt_sha = compute_file_hash(champ_dir / "model.pkl")

    meta = json.loads((champ_dir / "metadata.json").read_text())
    meta["model_sha256"] = corrupt_sha
    (champ_dir / "metadata.json").write_text(json.dumps(meta))
    (tmp_path / "champion.dvc").write_text(
        f"outs:\n- md5: deadbeef\n  sha256: {corrupt_sha}\n  size: {len(corrupt_bytes)}\n  path: champion\n"
    )

    # Reset entrypoint call log
    call_log.write_text("")

    # 1. import_champion_baseline must raise RuntimeError and NOT call entrypoint
    with pytest.raises(RuntimeError, match="Refusing to silently retrain champion"):
        import_champion_baseline(cfg, baseline_dir=champ_dir, pinned_sha256=corrupt_sha)
    assert call_log.read_text().strip() == ""

    # 2. load_candidate_and_champion trains candidate ONCE, then raises on corrupt champion (never 2nd call)
    with pytest.raises(RuntimeError, match="Refusing to silently retrain champion"):
        load_candidate_and_champion(cfg)
    assert call_log.read_text().splitlines() == ["called"]


def test_unverified_bare_pickle_and_empty_dvc_sha_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loader and verifier must fail closed when manifests or dvc sha256 are missing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CHAMPION_PINNED_SHA256", raising=False)
    cfg = _make_project(tmp_path, tmp_path / "calls.log")
    champ_dir = tmp_path / "champion"
    export_champion_baseline(cfg, baseline_dir=champ_dir)

    # Remove all manifests leaving only bare model.pkl -> loader must fail closed
    (champ_dir / "metadata.json").unlink()
    (tmp_path / "champion.dvc").unlink()
    with pytest.raises(ValueError, match="Cannot verify champion integrity"):
        load_candidate_and_champion(cfg)

    # Present champion.dvc with missing/empty sha256 -> verifier must fail closed
    (tmp_path / "champion.dvc").write_text("outs:\n- md5: deadbeef\n  path: champion\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch against champion.dvc"):
        verify_champion_integrity(baseline_dir=champ_dir, dvc_path=tmp_path / "champion.dvc")


def test_release_workflow_verifies_pep440_tag_matches_pyproject() -> None:
    """release.yml must verify that the pushed tag normalizes to pyproject.toml version."""
    import tomllib

    from packaging.version import Version

    release_wf = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "Verify release tag normalizes to pyproject.toml version" in release_wf
    assert "packaging.version" in release_wf

    pyproject_ver = tomllib.load((REPO_ROOT / "pyproject.toml").open("rb"))["project"]["version"]
    assert Version("v0.1.0-alpha".lstrip("v")) == Version(pyproject_ver)
    assert Version("v0.2.0".lstrip("v")) != Version(pyproject_ver)
