"""Lineage capture, persistence, and reproduction checks."""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

from predgate.loader import LineageInfo, compute_file_hash, get_git_sha, get_package_versions

logger = logging.getLogger(__name__)


def capture_lineage(
    data_paths: dict[str, str],
    seed: int,
    model: Any = None,
) -> LineageInfo:
    """Capture git, data hashes, Python/package versions, seed, and model hyperparameters."""
    data_hashes: dict[str, str] = {}
    for name, path in data_paths.items():
        p = Path(path)
        if p.exists():
            data_hashes[name] = compute_file_hash(p)

    hyperparams: dict[str, Any] = {}
    if model is not None and hasattr(model, "get_params"):
        with contextlib.suppress(Exception):
            hyperparams = model.get_params()

    package_versions = get_package_versions()

    return LineageInfo(
        git_sha=get_git_sha(),
        data_hashes=data_hashes,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        package_versions=package_versions,
        seed=seed,
        hyperparameters=hyperparams,
    )


def save_lineage(lineage: LineageInfo, path: str | Path) -> None:
    """Serialize a `LineageInfo` record to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "git_sha": lineage.git_sha,
        "data_hashes": lineage.data_hashes,
        "python_version": lineage.python_version,
        "seed": lineage.seed,
        "hyperparameters": lineage.hyperparameters,
        "package_versions": lineage.package_versions,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_lineage(path: str | Path) -> LineageInfo:
    """Load a `LineageInfo` record from JSON."""
    with open(path) as f:
        data = json.load(f)

    return LineageInfo(
        git_sha=data.get("git_sha", ""),
        data_hashes=data.get("data_hashes", {}),
        python_version=data.get("python_version", ""),
        package_versions=data.get("package_versions", {}),
        seed=data.get("seed", 0),
        hyperparameters=data.get("hyperparameters", {}),
    )


def verify_reproduction(
    original_lineage: LineageInfo,
    reproduced_metric: float,
    original_metric: float,
    tolerance: float = 0.001,
) -> tuple[bool, str]:
    """Return `(matched, message)` checking whether a reproduced metric is within `tolerance`."""
    diff = abs(reproduced_metric - original_metric)
    matched = diff <= tolerance

    if matched:
        msg = (
            f"✅ Reproduction verified: metric {reproduced_metric:.6f} "
            f"matches original {original_metric:.6f} (diff {diff:.6f} <= {tolerance})"
        )
    else:
        msg = (
            f"❌ Reproduction failed: metric {reproduced_metric:.6f} "
            f"differs from original {original_metric:.6f} by {diff:.6f} (> {tolerance})"
        )

    return matched, msg
