"""Data contract check.

Validates train, test, and inference DataFrames against a Pandera schema.
Reports column presence, dtypes, ranges, null rates, allowed categories, and uniqueness.
"""

from __future__ import annotations

import importlib
import time
from typing import Any

import pandas as pd

try:
    from pandera.pandas import DataFrameSchema
except ImportError:
    from pandera import DataFrameSchema  # type: ignore[no-redef]

from predgate.config import DataContractConfig
from predgate.verdict import CheckResult


def _import_schema(schema_path: str) -> Any:
    """Import a Pandera schema from a module:object path.

    Args:
        schema_path: e.g. 'contracts/schema.py:schema'

    Returns:
        The schema object.
    """
    from pathlib import Path

    module_path, obj_name = schema_path.rsplit(":", 1)
    path = Path(module_path)

    if path.exists():
        spec = importlib.util.spec_from_file_location("_schema_mod", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load schema from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)

    return getattr(module, obj_name)


def run_data_contract_check(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema_path: str,
    config: DataContractConfig,
) -> CheckResult:
    """Run the data contract check against a Pandera schema.

    Args:
        train_df: Training data.
        test_df: Test data.
        schema_path: Path to the Pandera schema (module:object).
        config: Data contract check config.

    Returns:
        CheckResult with violations listed.
    """
    start = time.time()

    if not config.enabled:
        return CheckResult(
            name="data_contract",
            status="skip",
            summary="Data contract check disabled",
            duration_s=time.time() - start,
        )

    if not schema_path:
        return CheckResult(
            name="data_contract",
            status="skip",
            summary="No schema configured",
            duration_s=time.time() - start,
        )

    if isinstance(schema_path, DataFrameSchema):
        schema = schema_path
    else:
        try:
            schema = _import_schema(schema_path)
        except (ImportError, AttributeError) as e:
            return CheckResult(
                name="data_contract",
                status="fail",
                summary=f"Failed to load schema: {e}",
                duration_s=time.time() - start,
            )

    violations: list[dict[str, Any]] = []

    for name, df in [("train", train_df), ("test", test_df)]:
        try:
            schema.validate(df, lazy=True)
        except Exception as e:
            # Pandera SchemaErrors contains failure_cases
            if hasattr(e, "failure_cases"):
                for _, row in e.failure_cases.iterrows():
                    violations.append(
                        {
                            "dataset": name,
                            "column": str(row.get("column", "")),
                            "check": str(row.get("check", "")),
                            "failure_case": str(row.get("failure_case", "")),
                        }
                    )
            else:
                violations.append(
                    {
                        "dataset": name,
                        "column": "unknown",
                        "check": "validation",
                        "failure_case": str(e),
                    }
                )

    n_violations = len(violations)
    duration = time.time() - start

    if n_violations == 0:
        return CheckResult(
            name="data_contract",
            status="pass",
            summary="0 violations",
            metrics={"violations": 0},
            details={"violations": []},
            duration_s=duration,
        )

    status = config.on_violation  # "fail" or "warn"
    return CheckResult(
        name="data_contract",
        status=status,
        summary=f"{n_violations} violation(s) found",
        metrics={"violations": float(n_violations)},
        details={"violations": violations[:50]},  # Cap detail output
        duration_s=duration,
    )
