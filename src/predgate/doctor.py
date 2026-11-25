"""PredGate environment and project diagnostics (`predgate doctor`).

Checks Python version, git, DVC, MLflow, configuration file validity,
Pandera schema import, model entrypoint import, and dataset paths,
reporting actionable fixes for any issue found.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from predgate.config import PredGateConfig, load_config
from predgate.loader import import_entrypoint

DoctorStatus = Literal["ok", "warn", "fail"]


@dataclass
class DoctorCheck:
    """Result of a single `predgate doctor` diagnostic check."""

    name: str
    status: DoctorStatus
    detail: str
    fix: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "fix": self.fix,
        }


@dataclass
class DoctorReport:
    """Aggregated `predgate doctor` report."""

    checks: list[DoctorCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status != "fail" for c in self.checks)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if c.status == "fail")

    @property
    def n_warn(self) -> int:
        return sum(1 for c in self.checks if c.status == "warn")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n_fail": self.n_fail,
            "n_warn": self.n_warn,
            "checks": [c.to_dict() for c in self.checks],
        }


def _check_python() -> DoctorCheck:
    ver = sys.version_info
    ver_str = f"{ver.major}.{ver.minor}.{ver.micro}"
    if (ver.major, ver.minor) >= (3, 10):
        return DoctorCheck(
            name="python",
            status="ok",
            detail=f"Python {ver_str} (>= 3.10 required)",
        )
    return DoctorCheck(
        name="python",
        status="fail",
        detail=f"Python {ver_str} is below minimum 3.10",
        fix="Install Python 3.10+ (recommended 3.12) and recreate your virtualenv: python3.12 -m venv .venv",
    )


def _check_git(cwd: Path) -> DoctorCheck:
    git_bin = shutil.which("git")
    if not git_bin:
        return DoctorCheck(
            name="git",
            status="fail",
            detail="git executable not found on PATH",
            fix="Install git (e.g. `sudo apt install git` or `brew install git`) and ensure it is on PATH.",
        )
    ver_res = subprocess.run([git_bin, "--version"], capture_output=True, text=True, check=False)
    ver_text = ver_res.stdout.strip() or "git found"
    repo_res = subprocess.run(
        [git_bin, "rev-parse", "--is-inside-work-tree"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if repo_res.returncode != 0:
        return DoctorCheck(
            name="git",
            status="warn",
            detail=f"{ver_text}, but '{cwd}' is not a git repository",
            fix="Run `git init` so PredGate can record git SHA lineage and verify base-branch champions.",
        )
    return DoctorCheck(name="git", status="ok", detail=f"{ver_text} (repository detected)")


def _check_dvc(cwd: Path) -> DoctorCheck:
    dvc_bin = shutil.which("dvc") or str(Path(sys.executable).parent / "dvc")
    dvc_exists = Path(dvc_bin).exists()
    dvc_mod: Any = None
    try:
        dvc_mod = importlib.import_module("dvc")
    except Exception:
        dvc_mod = None

    if not dvc_exists and dvc_mod is None:
        return DoctorCheck(
            name="dvc",
            status="warn",
            detail="DVC is not installed (optional unless tracking data/champion via DVC)",
            fix="Install DVC in your virtualenv: `pip install dvc`",
        )

    ver = getattr(dvc_mod, "__version__", "installed")
    has_dvc_files = (cwd / "data.dvc").exists() or (cwd / "champion.dvc").exists()
    detail = f"dvc {ver}" + (" (DVC lockfiles found)" if has_dvc_files else " (no .dvc files yet)")
    return DoctorCheck(name="dvc", status="ok", detail=detail)


def _check_mlflow(cfg: PredGateConfig | None) -> DoctorCheck:
    try:
        mlflow = importlib.import_module("mlflow")
    except Exception as exc:
        return DoctorCheck(
            name="mlflow",
            status="fail",
            detail=f"mlflow could not be imported: {exc}",
            fix="Install MLflow in your virtualenv: `pip install mlflow`",
        )

    ver = getattr(mlflow, "__version__", "installed")
    uri = cfg.mlflow.tracking_uri if cfg is not None else "sqlite:///mlflow.db"
    return DoctorCheck(name="mlflow", status="ok", detail=f"mlflow {ver} (tracking_uri={uri})")


def _check_config(config_path: Path) -> tuple[DoctorCheck, PredGateConfig | None]:
    if not config_path.exists():
        return (
            DoctorCheck(
                name="config",
                status="fail",
                detail=f"Config file not found: {config_path}",
                fix="Run `predgate init` to scaffold predgate.yaml, or pass `--config <path>`.",
            ),
            None,
        )
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        return (
            DoctorCheck(
                name="config",
                status="fail",
                detail=f"Invalid config in {config_path}: {exc}",
                fix=f"Fix validation errors in {config_path} (see docs/usage.md for the schema).",
            ),
            None,
        )
    preset_info = f", preset={cfg.preset}" if cfg.preset else ""
    return (
        DoctorCheck(
            name="config",
            status="ok",
            detail=f"Valid config ({config_path}, project={cfg.project}{preset_info})",
        ),
        cfg,
    )


def _check_entrypoint(cfg: PredGateConfig | None) -> DoctorCheck:
    if cfg is None:
        return DoctorCheck(
            name="entrypoint",
            status="warn",
            detail="Skipped because config could not be loaded",
            fix="Fix the config file first, then re-run `predgate doctor`.",
        )
    ep = cfg.model.entrypoint
    if ":" not in ep:
        return DoctorCheck(
            name="entrypoint",
            status="fail",
            detail=f"Malformed model.entrypoint '{ep}' (expected 'path/to/file.py:func' or 'pkg.mod:func')",
            fix="Update `model.entrypoint` in predgate.yaml to `module_or_file:callable_name`.",
        )
    try:
        fn = import_entrypoint(ep)
    except Exception as exc:
        return DoctorCheck(
            name="entrypoint",
            status="fail",
            detail=f"Cannot import model entrypoint '{ep}': {exc}",
            fix=f"Ensure the module/file and function in `model.entrypoint: {ep}` exist and import cleanly.",
        )
    if not callable(fn):
        return DoctorCheck(
            name="entrypoint",
            status="fail",
            detail=f"Model entrypoint '{ep}' resolved to non-callable {type(fn).__name__}",
            fix="Point `model.entrypoint` to a callable `build_model(X_train, y_train, seed=...)`.",
        )
    return DoctorCheck(name="entrypoint", status="ok", detail=f"Callable imported: {ep}")


def _check_schema(cfg: PredGateConfig | None) -> DoctorCheck:
    if cfg is None:
        return DoctorCheck(
            name="schema",
            status="warn",
            detail="Skipped because config could not be loaded",
            fix="Fix the config file first, then re-run `predgate doctor`.",
        )
    if not cfg.checks.data_contract.enabled:
        return DoctorCheck(
            name="schema",
            status="ok",
            detail="Data contract check disabled in config",
        )
    schema_ref = cfg.data.schema
    if not schema_ref:
        return DoctorCheck(
            name="schema",
            status="warn",
            detail="No data.schema configured (data_contract check will be skipped)",
            fix="Set `data.schema: path/to/schema.py:schema` in predgate.yaml to enforce a Pandera schema.",
        )
    try:
        obj = import_entrypoint(schema_ref)
    except Exception as exc:
        return DoctorCheck(
            name="schema",
            status="fail",
            detail=f"Failed to import schema '{schema_ref}': {exc}",
            fix=f"Verify `data.schema: {schema_ref}` points to an importable Pandera DataFrameSchema object.",
        )
    if not hasattr(obj, "validate"):
        return DoctorCheck(
            name="schema",
            status="fail",
            detail=f"Imported schema '{schema_ref}' ({type(obj).__name__}) has no `.validate()` method",
            fix="Export a Pandera `DataFrameSchema` instance and reference it as `module.py:schema`.",
        )
    return DoctorCheck(name="schema", status="ok", detail=f"Schema imported: {schema_ref}")


def _check_data_paths(cfg: PredGateConfig | None) -> DoctorCheck:
    if cfg is None:
        return DoctorCheck(
            name="data_paths",
            status="warn",
            detail="Skipped because config could not be loaded",
            fix="Fix the config file first, then re-run `predgate doctor`.",
        )

    missing: list[str] = []
    unreadable: list[str] = []
    missing_target: list[str] = []
    checked: list[str] = []

    required_pairs = [("train", cfg.data.train), ("test", cfg.data.test)]
    optional_pairs = [("reference", cfg.data.reference)] if cfg.data.reference else []

    for label, path_str in required_pairs + optional_pairs:
        p = Path(path_str)
        if not p.exists():
            if label == "reference":
                unreadable.append(f"{label}={path_str} (missing; drift/label_shift will skip)")
            else:
                missing.append(f"{label}={path_str}")
            continue
        try:
            df_head = pd.read_csv(p, nrows=5)
        except Exception as exc:
            unreadable.append(f"{label}={path_str} ({exc})")
            continue
        if cfg.data.target not in df_head.columns:
            missing_target.append(f"{label}={path_str} (missing target '{cfg.data.target}')")
        else:
            checked.append(f"{label}={path_str}")

    if missing or missing_target:
        problems = missing + missing_target
        return DoctorCheck(
            name="data_paths",
            status="fail",
            detail="; ".join(problems),
            fix="Generate or pull your datasets (e.g. `python demo/train.py` or `dvc pull`) and confirm `data.target` matches your CSV columns.",
        )
    if unreadable:
        return DoctorCheck(
            name="data_paths",
            status="warn",
            detail=f"Required datasets OK ({', '.join(checked)}), but: {'; '.join(unreadable)}",
            fix="Generate or configure `data.reference` CSV so drift and label-shift checks can run.",
        )
    return DoctorCheck(
        name="data_paths",
        status="ok",
        detail=f"Readable CSVs with target '{cfg.data.target}': {', '.join(checked)}",
    )


def run_doctor(config_path: str | Path = "predgate.yaml", cwd: Path | None = None) -> DoctorReport:
    """Run all PredGate environment and configuration checks."""
    work_dir = cwd or Path.cwd()
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = work_dir / cfg_path

    checks: list[DoctorCheck] = [
        _check_python(),
        _check_git(work_dir),
        _check_dvc(work_dir),
    ]
    cfg_check, cfg = _check_config(cfg_path)
    checks.append(_check_mlflow(cfg))
    checks.append(cfg_check)
    checks.append(_check_entrypoint(cfg))
    checks.append(_check_schema(cfg))
    checks.append(_check_data_paths(cfg))
    return DoctorReport(checks=checks)
