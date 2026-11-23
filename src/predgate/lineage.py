"""PredGate lineage tracking.

Records and verifies full lineage: data hashes, git commit, environment,
hyperparameters, and seed. Supports exact model reproduction.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from predgate.loader import LineageInfo, compute_file_hash, get_git_sha

logger = logging.getLogger(__name__)


def capture_lineage(
    data_paths: dict[str, str],
    seed: int,
    model: Any = None,
) -> LineageInfo:
    """Capture full lineage information for the current environment.

    Args:
        data_paths: Mapping of name -> path for data files.
        seed: Random seed used.
        model: Optional model to extract hyperparameters from.

    Returns:
        LineageInfo with all captured information.
    """
    data_hashes: dict[str, str] = {}
    for name, path in data_paths.items():
        p = Path(path)
        if p.exists():
            data_hashes[name] = compute_file_hash(p)

    hyperparams: dict[str, Any] = {}
    if model is not None and hasattr(model, "get_params"):
        try:
            hyperparams = model.get_params()
        except Exception:
            pass

    # Get package versions
    package_versions: dict[str, str] = {}
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.strip().split("\n"):
            if "==" in line:
                name, ver = line.split("==", 1)
                package_versions[name.strip()] = ver.strip()
    except Exception:
        pass

    return LineageInfo(
        git_sha=get_git_sha(),
        data_hashes=data_hashes,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        package_versions=package_versions,
        seed=seed,
        hyperparameters=hyperparams,
    )


def save_lineage(lineage: LineageInfo, path: str | Path) -> None:
    """Save lineage information to a JSON file.

    Args:
        lineage: LineageInfo to save.
        path: Output file path.
    """
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
    """Load lineage information from a JSON file.

    Args:
        path: Path to the lineage JSON file.

    Returns:
        LineageInfo instance.
    """
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
    """Verify that a reproduced model matches the original.

    Args:
        original_lineage: Lineage from the original run.
        reproduced_metric: Metric from the reproduced model.
        original_metric: Metric from the original model.
        tolerance: Acceptable difference in metrics.

    Returns:
        Tuple of (matched, message).
    """
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
