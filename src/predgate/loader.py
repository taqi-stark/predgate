"""PredGate candidate loader.

Imports the model entrypoint, trains with a fixed seed, loads the champion
from MLflow, and records lineage information.
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predgate.config import PredGateConfig


@dataclass
class LineageInfo:
    """Full lineage information for a model run."""

    git_sha: str = ""
    data_hashes: dict[str, str] = field(default_factory=dict)
    dvc_hash: str = ""
    python_version: str = ""
    package_versions: dict[str, str] = field(default_factory=dict)
    seed: int = 0
    hyperparameters: dict[str, Any] = field(default_factory=dict)


def get_dvc_hash(dvc_path: str | Path = "data.dvc") -> str:
    """Read data hash from DVC file if present.

    Args:
        dvc_path: Path to .dvc file (default: data.dvc).

    Returns:
        DVC data hash string, or empty string.
    """
    path = Path(dvc_path)
    if path.exists():
        try:
            import yaml

            with open(path) as f:
                d = yaml.safe_load(f)
                if d and "outs" in d and len(d["outs"]) > 0:
                    return str(d["outs"][0].get("md5", ""))
        except Exception:
            pass
    return ""


@dataclass
class LoadResult:
    """Result from loading/training candidate and champion models."""

    candidate_model: Any
    champion_model: Any | None
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    reference_df: pd.DataFrame | None
    target_column: str
    train_time_s: float
    lineage: LineageInfo
    is_first_run: bool = False


def detect_model_family(model: Any) -> dict[str, str]:
    """Identify the underlying estimator algorithm / model family of a model or pipeline.

    Unwraps scikit-learn ``Pipeline`` steps and common wrapper attributes
    (``base_model``, ``estimator``, ``classifier``, ``model``).
    """
    est = model
    seen: set[int] = set()
    while est is not None and id(est) not in seen:
        seen.add(id(est))
        steps = getattr(est, "steps", None)
        if (
            isinstance(steps, list)
            and steps
            and isinstance(steps[-1], tuple)
            and len(steps[-1]) == 2
        ):
            est = steps[-1][1]
            continue
        unwrapped = False
        for attr in ("base_model", "estimator", "classifier", "model"):
            inner = getattr(est, attr, None)
            if inner is not None and inner is not est and hasattr(inner, "predict"):
                est = inner
                unwrapped = True
                break
        if not unwrapped:
            break

    cls = type(est)
    top_pkg = cls.__module__.split(".")[0] if cls.__module__ else ""
    qualified = f"{top_pkg}.{cls.__name__}" if top_pkg else cls.__name__
    return {
        "name": cls.__name__,
        "module": cls.__module__,
        "qualified": qualified,
    }


def detect_model_family_change(
    champion_model: Any | None,
    candidate_model: Any | None,
) -> dict[str, Any] | None:
    """Compare champion and candidate model families and return change metadata."""
    if champion_model is None or candidate_model is None:
        return None
    champ = detect_model_family(champion_model)
    cand = detect_model_family(candidate_model)
    changed = champ["qualified"] != cand["qualified"]
    note = (
        f"Model family changed: Champion ({champ['name']}) -> Candidate ({cand['name']}). "
        "Algorithm swaps can shift single-row latency and subgroup selection rates even without data or label bugs."
        if changed
        else ""
    )
    return {
        "changed": changed,
        "champion_family": champ["name"],
        "candidate_family": cand["name"],
        "champion_qualified": champ["qualified"],
        "candidate_qualified": cand["qualified"],
        "note": note,
    }


def compute_file_hash(path: str | Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        path: Path to the file.

    Returns:
        Hex digest of the SHA-256 hash.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_sha() -> str:
    """Get the current git commit SHA.

    Returns:
        Short git SHA or empty string if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def get_package_versions() -> dict[str, str]:
    """Get installed package versions via pip freeze.

    Returns:
        Dictionary of package_name -> version.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        versions: dict[str, str] = {}
        for line in result.stdout.strip().split("\n"):
            if "==" in line:
                name, version = line.split("==", 1)
                versions[name.strip()] = version.strip()
        return versions
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}


def import_entrypoint(entrypoint: str) -> Any:
    """Import a function from a module:function entrypoint string.

    Args:
        entrypoint: String like 'demo/train.py:build_model'.

    Returns:
        The callable function.
    """
    module_path, func_name = entrypoint.rsplit(":", 1)

    # Convert file path to module import
    path = Path(module_path)
    if path.exists():
        spec = importlib.util.spec_from_file_location("_entrypoint", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)

    func = getattr(module, func_name)
    return func


def load_data(config: PredGateConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Load train, test, and optional reference data.

    Args:
        config: PredGate configuration.

    Returns:
        Tuple of (train_df, test_df, reference_df).
    """
    train_df = pd.read_csv(config.data.train)
    test_df = pd.read_csv(config.data.test)

    reference_df = None
    if config.data.reference:
        ref_path = Path(config.data.reference)
        if ref_path.exists():
            reference_df = pd.read_csv(ref_path)

    return train_df, test_df, reference_df


def load_candidate_and_champion(
    config: PredGateConfig,
) -> LoadResult:
    """Load/train the candidate model and load the champion from MLflow.

    Args:
        config: PredGate configuration.

    Returns:
        LoadResult with both models and metadata.
    """
    # Set the seed
    np.random.seed(config.seed)

    # Load data
    train_df, test_df, reference_df = load_data(config)

    # Compute data hashes
    data_hashes: dict[str, str] = {}
    for name, path in [
        ("train", config.data.train),
        ("test", config.data.test),
    ]:
        if Path(path).exists():
            data_hashes[name] = compute_file_hash(path)

    if config.data.reference and Path(config.data.reference).exists():
        data_hashes["reference"] = compute_file_hash(config.data.reference)

    dvc_hash = get_dvc_hash()
    if dvc_hash:
        data_hashes["dvc"] = dvc_hash

    # Import and train the candidate model
    build_fn = import_entrypoint(config.model.entrypoint)

    target = config.data.target
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    start = time.time()
    candidate_model = build_fn(X_train, y_train, seed=config.seed)
    train_time = time.time() - start

    # Try to load champion from MLflow, falling back to persisted champion/ baseline folder
    champion_model = None
    is_first_run = True
    try:
        import os

        import mlflow

        env_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
        active_uri = env_uri or config.mlflow.tracking_uri
        mlflow.set_tracking_uri(active_uri)
        client = mlflow.tracking.MlflowClient(active_uri)
        model_name = config.project
        try:
            client.get_model_version_by_alias(model_name, config.champion.mlflow_alias)
            champion_model = mlflow.sklearn.load_model(
                f"models:/{model_name}@{config.champion.mlflow_alias}"
            )
            is_first_run = False
        except Exception:
            is_first_run = True
    except ImportError:
        is_first_run = True

    if champion_model is None:
        baseline_dir = Path(getattr(config.champion, "baseline_dir", "champion"))
        baseline_model_path = baseline_dir / "model.pkl"
        if baseline_model_path.exists():
            from predgate.registry import verify_champion_integrity

            # Always fail closed: require manifest or pinned SHA-256 verification before unpickling
            verify_champion_integrity(baseline_dir=baseline_dir)
            try:
                import pickle

                with open(baseline_model_path, "rb") as f:
                    champion_model = pickle.load(f)
                champion_model.predict(test_df.drop(columns=[target]).iloc[:2])
                is_first_run = False
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load or run verified champion baseline pickle '{baseline_model_path}': {exc}. "
                    "Refusing to silently retrain champion from candidate entrypoint."
                ) from exc

    # Build lineage
    lineage = LineageInfo(
        git_sha=get_git_sha(),
        data_hashes=data_hashes,
        dvc_hash=dvc_hash,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        package_versions=get_package_versions(),
        seed=config.seed,
        hyperparameters=getattr(candidate_model, "get_params", lambda: {})(),
    )

    return LoadResult(
        candidate_model=candidate_model,
        champion_model=champion_model,
        train_df=train_df,
        test_df=test_df,
        reference_df=reference_df,
        target_column=target,
        train_time_s=train_time,
        lineage=lineage,
        is_first_run=is_first_run,
    )
