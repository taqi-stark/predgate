"""Candidate model training, champion loading, model-family detection, and lineage capture."""

from __future__ import annotations

import contextlib
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
    """Read the primary output MD5 hash from a `.dvc` file, or return `""`."""
    path = Path(dvc_path)
    if path.exists():
        with contextlib.suppress(Exception):
            import yaml

            with open(path) as f:
                d = yaml.safe_load(f)
                if d and "outs" in d and len(d["outs"]) > 0:
                    return str(d["outs"][0].get("md5", ""))
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


def _is_mock_object(obj: Any) -> bool:
    """Return True if obj is a unittest.mock proxy (which fabricates attributes dynamically)."""
    mod = getattr(type(obj), "__module__", "") or ""
    return mod.startswith(("unittest.mock", "mock."))


def _is_ensemble_estimator(obj: Any) -> bool:
    """Return True if obj is an ensemble whose `.estimator` is a template base learner, not a wrapper."""
    mod = getattr(type(obj), "__module__", "") or ""
    if "ensemble" in mod:
        return True
    obj_dict = getattr(obj, "__dict__", {})
    return any(k in obj_dict for k in ("n_estimators", "estimators_", "estimators"))


def detect_model_family(model: Any) -> dict[str, str]:
    """Identify the underlying estimator algorithm / model family of a model or pipeline.

    Unwraps scikit-learn ``Pipeline`` steps, search/target wrappers (``best_estimator_``,
    ``regressor_``), custom single-model wrappers (``base_model``, ``classifier``, ``model``),
    and non-ensemble single-model wrappers (e.g. ``CalibratedClassifierCV.estimator``), while
    never unwrapping ensemble template learners (e.g. ``RandomForestClassifier.estimator``)
    or dynamic mock objects.
    """
    est = model
    seen: set[int] = set()
    for _ in range(10):
        if est is None or id(est) in seen or _is_mock_object(est):
            break
        seen.add(id(est))

        obj_dict = getattr(est, "__dict__", {})
        steps = obj_dict.get("steps")
        if (
            isinstance(steps, list)
            and steps
            and isinstance(steps[-1], tuple)
            and len(steps[-1]) == 2
            and steps[-1][1] is not None
        ):
            est = steps[-1][1]
            continue

        unwrapped = False
        candidate_attrs: tuple[str, ...] = (
            "best_estimator_",
            "regressor_",
            "base_model",
            "classifier",
            "model",
        )
        if not _is_ensemble_estimator(est):
            candidate_attrs = (*candidate_attrs, "estimator")
        for attr in candidate_attrs:
            if attr not in obj_dict:
                continue
            inner = obj_dict[attr]
            if (
                inner is not None
                and inner is not est
                and not _is_mock_object(inner)
                and hasattr(inner, "predict")
            ):
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
    """Return the SHA-256 hex digest of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_sha() -> str:
    """Return the short git commit SHA, or `""` outside a git repository."""
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


_CORE_LINEAGE_PACKAGES: tuple[str, ...] = (
    "predgate",
    "scikit-learn",
    "numpy",
    "pandas",
    "scipy",
    "mlflow",
    "pandera",
    "pydantic",
    "fastapi",
    "dvc",
)


def get_package_versions() -> dict[str, str]:
    """Return core runtime package versions via importlib.metadata (avoids spawning `pip freeze`)."""
    import importlib.metadata as importlib_metadata

    versions: dict[str, str] = {}
    for pkg in _CORE_LINEAGE_PACKAGES:
        with contextlib.suppress(importlib_metadata.PackageNotFoundError):
            versions[pkg] = importlib_metadata.version(pkg)
    return versions


def import_entrypoint(entrypoint: str) -> Any:
    """Import a callable from a `path/to/file.py:func` or `module:func` string."""
    module_path, func_name = entrypoint.rsplit(":", 1)

    path = Path(module_path)
    if path.exists():
        spec = importlib.util.spec_from_file_location("_entrypoint", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)

    return getattr(module, func_name)


def load_data(config: PredGateConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Load `(train_df, test_df, reference_df)` CSVs configured in `config.data`."""
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
    """Train the candidate entrypoint and load the champion from MLflow or `champion/`."""
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

        env_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
        active_uri = env_uri or config.mlflow.tracking_uri

        # Skip initializing a fresh SQLite MLflow DB when the DB file does not exist yet
        # (it cannot contain a registered @champion and creating tables takes ~4-10s).
        skip_mlflow_lookup = False
        if active_uri.startswith("sqlite:///"):
            db_path = Path(active_uri[len("sqlite:///") :])
            if not db_path.exists() or db_path.stat().st_size == 0:
                skip_mlflow_lookup = True

        if not skip_mlflow_lookup:
            import mlflow

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
        hyperparameters=getattr(candidate_model, "get_params", dict)(),
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
