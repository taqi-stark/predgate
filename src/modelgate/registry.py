"""ModelGate registry and lineage.

Handles MLflow logging, champion promotion, and model reproduction.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from modelgate.verdict import Verdict

logger = logging.getLogger(__name__)


def _get_active_tracking_uri(tracking_uri: str | None = None) -> str:
    """Resolve active MLflow tracking URI (MLFLOW_TRACKING_URI env var overrides config)."""
    env_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if env_uri:
        return env_uri
    if tracking_uri:
        return tracking_uri
    return "sqlite:///mlflow.db"


def log_run_to_mlflow(
    project: str,
    verdict: Verdict,
    lineage_info: dict[str, Any],
    model: Any,
    seed: int,
    metrics: dict[str, float] | None = None,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
) -> str | None:
    """Log a ModelGate run to MLflow.

    Args:
        project: Project/experiment name.
        verdict: The run verdict.
        lineage_info: Lineage metadata dict.
        model: The trained model.
        seed: Random seed used.
        metrics: Additional metrics to log.
        tracking_uri: Configured MLflow tracking URI.
        artifact_location: Optional directory for artifacts.

    Returns:
        MLflow run ID, or None if MLflow is unavailable.
    """
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        logger.warning("MLflow not available; skipping run logging")
        return None

    active_uri = _get_active_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(active_uri)

    # Ensure experiment exists with specified artifact location if needed
    try:
        client = mlflow.tracking.MlflowClient(active_uri)
        exp = client.get_experiment_by_name(project)
        if exp is None:
            effective_artifact_loc = (
                None if active_uri.startswith(("http://", "https://")) else artifact_location
            )
            client.create_experiment(project, artifact_location=effective_artifact_loc)
    except Exception:
        pass

    mlflow.set_experiment(project)

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("seed", seed)
        mlflow.log_param("verdict", verdict.status)
        mlflow.log_param("git_sha", lineage_info.get("git_sha", ""))

        dvc_hash = lineage_info.get("dvc_hash", "")
        if dvc_hash:
            mlflow.log_param("dvc_hash", dvc_hash)
            mlflow.set_tag("dvc_hash", dvc_hash)

        # Log metrics
        if metrics:
            mlflow.log_metrics(metrics)

        # Log check results as metrics
        for cr in verdict.check_results:
            for key, val in cr.metrics.items():
                try:
                    mlflow.log_metric(f"{cr.name}_{key}", float(val))
                except (ValueError, TypeError):
                    pass

        # Log the model
        try:
            mlflow.sklearn.log_model(model, "model", serialization_format="pickle")
        except Exception as e:
            logger.warning(f"Failed to log model artifact: {e}")

        # Log verdict JSON as artifact
        verdict_json = json.dumps(verdict.to_dict(), indent=2)
        mlflow.log_text(verdict_json, "verdict.json")

        # Log lineage
        lineage_json = json.dumps(lineage_info, indent=2, default=str)
        mlflow.log_text(lineage_json, "lineage.json")

        # Tags
        mlflow.set_tag("verdict", verdict.status)
        mlflow.set_tag("git_sha", lineage_info.get("git_sha", ""))
        mlflow.set_tag("seed", str(seed))

        for dname, dhash in lineage_info.get("data_hashes", {}).items():
            mlflow.set_tag(f"data_hash_{dname}", dhash)

        return run.info.run_id


def promote_to_champion(
    project: str,
    run_id: str,
    alias: str = "champion",
    tracking_uri: str | None = None,
) -> bool:
    """Promote a model version to champion in the MLflow registry.

    Args:
        project: Project/model name in the registry.
        run_id: MLflow run ID to promote.
        alias: Alias to set (default: "champion").
        tracking_uri: Optional MLflow tracking URI.

    Returns:
        True if promotion succeeded.
    """
    try:
        import mlflow

        active_uri = _get_active_tracking_uri(tracking_uri)
        mlflow.set_tracking_uri(active_uri)
        client = mlflow.tracking.MlflowClient(active_uri)

        # Find the model version associated with this run
        model_name = project

        # Ensure the registered model exists
        try:
            client.get_registered_model(model_name)
        except Exception:
            client.create_registered_model(model_name)

        # Register the model from the run
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)

        # Set the alias
        client.set_registered_model_alias(model_name, alias, mv.version)

        logger.info(f"Promoted {model_name} version {mv.version} to '{alias}'")
        return True

    except ImportError:
        logger.error("MLflow not available; cannot promote model")
        return False
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        return False


def get_reproduce_command(run_id: str) -> str:
    """Generate the CLI command to reproduce a run.

    Args:
        run_id: MLflow run ID.

    Returns:
        CLI command string.
    """
    return f"modelgate reproduce {run_id}"


def export_champion_baseline(
    config: Any,
    baseline_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Export the champion model, metadata, and baseline test metrics to a champion/ folder.

    Also writes champion.dvc so the exported baseline is tracked deterministically in CI.
    If no champion exists in MLflow yet, trains the baseline model from config.model.entrypoint.
    """
    import hashlib
    import pickle
    import sys
    from datetime import datetime, timezone
    from pathlib import Path

    UTC = timezone.utc

    from sklearn.metrics import accuracy_score, f1_score

    from modelgate.loader import (
        compute_file_hash,
        get_dvc_hash,
        get_git_sha,
        import_entrypoint,
        load_data,
    )

    out_dir = Path(str(baseline_dir or getattr(config.champion, "baseline_dir", "champion")))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, _ = load_data(config)
    target = config.data.target
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target].values

    model = None
    source = "entrypoint"
    mlflow_version = ""

    try:
        import mlflow
        import mlflow.sklearn

        active_uri = _get_active_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_tracking_uri(active_uri)
        client = mlflow.tracking.MlflowClient(active_uri)
        mv = client.get_model_version_by_alias(config.project, config.champion.mlflow_alias)
        model = mlflow.sklearn.load_model(
            f"models:/{config.project}@{config.champion.mlflow_alias}"
        )
        source = "mlflow"
        mlflow_version = str(mv.version)
    except Exception:
        build_fn = import_entrypoint(config.model.entrypoint)
        model = build_fn(X_train, y_train, seed=config.seed)
        source = "entrypoint"

    model_path = out_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    model_sha256 = compute_file_hash(model_path)
    model_md5 = hashlib.md5(model_path.read_bytes()).hexdigest()

    preds = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4),
        "positive_rate": round(float(preds.mean()), 4),
        "n_test_rows": int(len(y_test)),
    }

    metadata = {
        "project": config.project,
        "alias": config.champion.mlflow_alias,
        "source": source,
        "mlflow_version": mlflow_version,
        "git_sha": get_git_sha(),
        "dvc_hash": get_dvc_hash(),
        "seed": config.seed,
        "model_sha256": model_sha256,
        "model_md5": model_md5,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "exported_at": datetime.now(UTC).isoformat(),
    }

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Write deterministic champion.dvc pointer file
    dvc_file = out_dir.parent / f"{out_dir.name}.dvc"
    dvc_content = (
        f"outs:\n"
        f"- md5: {model_md5}\n"
        f"  sha256: {model_sha256}\n"
        f"  size: {model_path.stat().st_size}\n"
        f"  path: {out_dir.name}\n"
    )
    dvc_file.write_text(dvc_content)

    return {
        "baseline_dir": str(out_dir),
        "model_path": str(model_path),
        "metadata": metadata,
        "metrics": metrics,
    }


def verify_champion_integrity(
    baseline_dir: str | os.PathLike[str] = "champion",
    dvc_path: str | os.PathLike[str] | None = None,
    pinned_sha256: str = "",
    changed_files: list[str] | None = None,
) -> dict[str, Any]:
    """Verify champion baseline integrity against metadata.json, champion.dvc, and a pinned SHA-256.

    Fails closed on every verification path:
    - Any PR changed_files touching champion/ or champion.dvc is rejected unconditionally
      (no custom approval bypass; govern baseline updates via branch protection + CODEOWNERS).
    - Missing model.pkl, missing/empty sha256 in metadata.json or champion.dvc, or any
      SHA-256 mismatch raises ValueError/FileNotFoundError.
    """
    from pathlib import Path

    import yaml

    from modelgate.loader import compute_file_hash

    in_dir = Path(baseline_dir)
    dvc_file = Path(dvc_path) if dvc_path is not None else (in_dir.parent / f"{in_dir.name}.dvc")

    if changed_files:
        prefix = f"{in_dir.name}/"
        dvc_name = dvc_file.name
        touched = [
            f
            for f in changed_files
            if f == in_dir.name
            or f.startswith(prefix)
            or f == dvc_name
            or f.endswith(f"/{dvc_name}")
        ]
        if touched:
            raise ValueError(
                f"Unapproved modification of champion baseline in PR ({', '.join(touched)}). "
                "No custom approval bypass is permitted; govern champion/ updates via GitHub branch protection and CODEOWNERS."
            )

    model_path = in_dir / "model.pkl"
    meta_path = in_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Champion baseline model not found: {model_path}")

    actual_sha = compute_file_hash(model_path)
    pinned = (pinned_sha256 or os.environ.get("CHAMPION_PINNED_SHA256", "")).strip()

    checked_any = False

    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        expected_meta_sha = str(metadata.get("model_sha256", "")).strip()
        if not expected_meta_sha or expected_meta_sha != actual_sha:
            raise ValueError(
                f"Champion model SHA-256 mismatch against metadata.json: "
                f"expected {expected_meta_sha or '<missing>'}, got {actual_sha}"
            )
        checked_any = True

    if dvc_file.exists():
        dvc_data = yaml.safe_load(dvc_file.read_text()) or {}
        outs = dvc_data.get("outs", [])
        expected_dvc_sha = str(outs[0].get("sha256", "")).strip() if outs else ""
        if not expected_dvc_sha or expected_dvc_sha != actual_sha:
            raise ValueError(
                f"Champion model SHA-256 mismatch against {dvc_file.name}: "
                f"expected {expected_dvc_sha or '<missing>'}, got {actual_sha}"
            )
        checked_any = True

    if pinned:
        if pinned != actual_sha:
            raise ValueError(
                f"Champion model SHA-256 mismatch against pinned base-branch value: "
                f"expected {pinned}, got {actual_sha}"
            )
        checked_any = True

    if not checked_any:
        raise ValueError(
            f"Cannot verify champion integrity for {model_path}: "
            "no metadata.json, champion.dvc sha256, or pinned_sha256 provided."
        )

    return {
        "verified": True,
        "baseline_dir": str(in_dir),
        "model_path": str(model_path),
        "sha256": actual_sha,
        "pinned_sha256": pinned or actual_sha,
    }


def import_champion_baseline(
    config: Any,
    baseline_dir: str | os.PathLike[str] | None = None,
    pinned_sha256: str = "",
) -> dict[str, Any]:
    """Import a persisted champion baseline from champion/ into MLflow and set the @champion alias."""
    import pickle
    from pathlib import Path

    from modelgate.verdict import CheckResult, Verdict

    in_dir = Path(str(baseline_dir or getattr(config.champion, "baseline_dir", "champion")))
    model_path = in_dir / "model.pkl"
    meta_path = in_dir / "metadata.json"
    metrics_path = in_dir / "metrics.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Champion baseline model not found: {model_path}")

    verify_res = verify_champion_integrity(
        baseline_dir=in_dir,
        pinned_sha256=pinned_sha256,
    )
    actual_sha = str(verify_res["sha256"])

    metadata: dict[str, Any] = (
        json.loads(meta_path.read_text()) if meta_path.exists() else {"project": config.project}
    )
    metrics: dict[str, float] = (
        json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    )

    from modelgate.loader import load_data

    _, test_df, _ = load_data(config)
    target = config.data.target
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        model.predict(test_df.drop(columns=[target]).iloc[:2])
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load or run verified champion baseline pickle '{model_path}': {exc}. "
            "Refusing to silently retrain champion from candidate entrypoint."
        ) from exc

    verdict = Verdict(
        status="pass",
        reasons=[],
        check_results=[
            CheckResult(
                name="baseline_import",
                status="pass",
                summary=f"Imported champion baseline ({actual_sha[:8]})",
                metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            )
        ],
        summary="Imported persisted champion baseline",
    )
    lineage_info = {
        "git_sha": metadata.get("git_sha", ""),
        "dvc_hash": metadata.get("dvc_hash", ""),
        "seed": int(metadata.get("seed", config.seed)),
        "data_hashes": {"champion_model": actual_sha},
    }

    run_id = log_run_to_mlflow(
        config.project,
        verdict,
        lineage_info,
        model,
        seed=int(metadata.get("seed", config.seed)),
        metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        tracking_uri=config.mlflow.tracking_uri,
        artifact_location=config.mlflow.artifact_location,
    )
    promoted = False
    if run_id:
        promoted = promote_to_champion(
            config.project,
            run_id,
            alias=config.champion.mlflow_alias,
            tracking_uri=config.mlflow.tracking_uri,
        )

    return {
        "baseline_dir": str(in_dir),
        "run_id": run_id,
        "promoted": promoted,
        "metadata": metadata,
        "metrics": metrics,
    }
