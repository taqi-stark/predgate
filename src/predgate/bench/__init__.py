"""PredGate fault-injection benchmark package."""

from __future__ import annotations

from predgate.bench.faults import (
    FAULT_DESCRIPTIONS,
    GroupThresholdModel,
    _generate_data,
    _train_latency_model,
    _train_model,
    fixed_dp_limit_baseline,
    inject_fault,
    per_group_accuracy_drop_baseline,
)
from predgate.bench.render import (
    _generate_benchmark_markdown,
    generate_benchmark_markdown_from_json,
)
from predgate.bench.runner import (
    _get_git_commit,
    run_benchmark,
    run_single_trial,
    wilson_ci,
)

__all__ = [
    "FAULT_DESCRIPTIONS",
    "GroupThresholdModel",
    "_generate_benchmark_markdown",
    "_generate_data",
    "_get_git_commit",
    "_train_latency_model",
    "_train_model",
    "fixed_dp_limit_baseline",
    "generate_benchmark_markdown_from_json",
    "inject_fault",
    "per_group_accuracy_drop_baseline",
    "run_benchmark",
    "run_single_trial",
    "wilson_ci",
]
