"""Performance budget check.

Measures single-row latency via interleaved batched samples, gates on the ratio
of medians (candidate / champion) with a paired bootstrap confidence interval,
and optionally checks an absolute p95 latency ceiling, serialized model size,
and peak memory.
"""

from __future__ import annotations

import gc
import io
import pickle
import time
import tracemalloc
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd

from predgate.config import BudgetConfig
from predgate.stats import paired_bootstrap_median_ratio_ci
from predgate.verdict import CheckResult


def _measure_latency(
    model: Any,
    X_sample: pd.DataFrame,
    n_samples: int = 20,
    batch_size: int = 5,
    n_warmup: int = 5,
    clock: Callable[[], float] | None = None,
) -> np.ndarray:
    """Measure per-call latency (ms) over batched single-row prediction samples."""
    timer = clock or time.perf_counter
    single_row = X_sample.iloc[[0]]
    eff_batch = max(int(batch_size), 1)

    for _ in range(n_warmup):
        model.predict(single_row)

    gc.collect()

    samples = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        start = timer()
        for _ in range(eff_batch):
            model.predict(single_row)
        elapsed_ms = (timer() - start) * 1000.0
        samples[i] = elapsed_ms / eff_batch

    return samples


def _measure_paired_latency(
    candidate_model: Any,
    champion_model: Any,
    X_sample: pd.DataFrame,
    n_samples: int = 20,
    batch_size: int = 5,
    n_warmup: int = 5,
    clock: Callable[[], float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure candidate and champion latency over interleaved batched samples in the same job.

    Alternates batch execution order (candidate-first vs champion-first) on every paired sample
    so both models experience identical CPU scheduling and cache conditions.

    Returns:
        Tuple of (cand_samples_ms, champ_samples_ms), each an array of per-call ms per batch.
    """
    timer = clock or time.perf_counter
    single_row = X_sample.iloc[[0]]
    eff_batch = max(int(batch_size), 1)

    for _ in range(n_warmup):
        champion_model.predict(single_row)
        candidate_model.predict(single_row)

    gc.collect()

    cand_samples = np.empty(n_samples, dtype=np.float64)
    champ_samples = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        if i % 2 == 0:
            t0 = timer()
            for _ in range(eff_batch):
                candidate_model.predict(single_row)
            cand_samples[i] = ((timer() - t0) * 1000.0) / eff_batch

            t1 = timer()
            for _ in range(eff_batch):
                champion_model.predict(single_row)
            champ_samples[i] = ((timer() - t1) * 1000.0) / eff_batch
        else:
            t1 = timer()
            for _ in range(eff_batch):
                champion_model.predict(single_row)
            champ_samples[i] = ((timer() - t1) * 1000.0) / eff_batch

            t0 = timer()
            for _ in range(eff_batch):
                candidate_model.predict(single_row)
            cand_samples[i] = ((timer() - t0) * 1000.0) / eff_batch

    return cand_samples, champ_samples


def _measure_model_size(model: Any) -> float:
    """Measure the serialized model size in MB."""
    try:
        buf = io.BytesIO()
        pickle.dump(model, buf)
        size_bytes = buf.tell()
        return size_bytes / (1024 * 1024)
    except (pickle.PicklingError, TypeError, AttributeError):
        return 0.0


def _measure_peak_memory(model: Any, X_sample: pd.DataFrame) -> float:
    """Measure peak memory usage during prediction in MB."""
    tracemalloc.start()

    for _ in range(10):
        model.predict(X_sample.iloc[[0]])

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return peak / (1024 * 1024)


def run_budgets_check(
    test_df: pd.DataFrame,
    target_column: str,
    candidate_model: Any,
    champion_model: Any | None,
    config: BudgetConfig,
    seed: int = 42,
    clock: Callable[[], float] | None = None,
) -> CheckResult:
    """Run performance budget checks.

    When a champion model is available, measures champion and candidate interleaved over
    batched samples in the same job and gates on the ratio of medians with a paired
    bootstrap confidence interval. An absolute p95_latency_ms limit remains optional.
    """
    start = time.time()

    if not config.enabled:
        return CheckResult(
            name="budgets",
            status="skip",
            summary="Budget check disabled",
            duration_s=time.time() - start,
        )

    X_test = test_df.drop(columns=[target_column])

    champion_metrics: dict[str, float] = {}
    latency_bootstrap_detail: dict[str, float] | None = None
    latency_ratio: float | None = None
    ratio_ci_lower: float | None = None
    ratio_ci_upper: float | None = None
    champ_p50: float | None = None

    if champion_model is not None:
        cand_samples, champ_samples = _measure_paired_latency(
            candidate_model,
            champion_model,
            X_test,
            n_samples=config.latency_samples,
            batch_size=config.latency_batch_size,
            clock=clock,
        )
        p50 = float(np.median(cand_samples))
        p95 = float(np.percentile(cand_samples, 95))
        champ_p50 = float(np.median(champ_samples))
        champ_p95 = float(np.percentile(champ_samples, 95))

        ratio_boot = paired_bootstrap_median_ratio_ci(
            champ_samples,
            cand_samples,
            n_bootstrap=config.latency_n_bootstrap,
            confidence=config.latency_confidence,
            seed=seed,
        )
        latency_ratio = round(ratio_boot.point_estimate, 3)
        ratio_ci_lower = round(ratio_boot.ci_lower, 3)
        ratio_ci_upper = round(ratio_boot.ci_upper, 3)
        champ_size = _measure_model_size(champion_model)

        champion_metrics = {
            "champion_p50_ms": round(champ_p50, 2),
            "champion_p95_ms": round(champ_p95, 2),
            "latency_ratio": latency_ratio,
            "median_latency_ratio": latency_ratio,
            "latency_ratio_ci_lower": ratio_ci_lower,
            "latency_ratio_ci_upper": ratio_ci_upper,
            "latency_ratio_p_value": round(ratio_boot.p_value, 4),
            "champion_model_mb": round(champ_size, 2),
        }
        latency_bootstrap_detail = {
            "point_estimate": latency_ratio,
            "ci_lower": ratio_ci_lower,
            "ci_upper": ratio_ci_upper,
            "p_value": round(ratio_boot.p_value, 4),
            "n_samples": float(config.latency_samples),
            "batch_size": float(config.latency_batch_size),
        }
    else:
        cand_samples = _measure_latency(
            candidate_model,
            X_test,
            n_samples=config.latency_samples,
            batch_size=config.latency_batch_size,
            clock=clock,
        )
        p50 = float(np.median(cand_samples))
        p95 = float(np.percentile(cand_samples, 95))

    model_size_mb = _measure_model_size(candidate_model)
    peak_mem_mb = _measure_peak_memory(candidate_model, X_test)

    violations: list[str] = []
    warnings: list[str] = []

    if (
        latency_ratio is not None
        and ratio_ci_lower is not None
        and ratio_ci_upper is not None
        and champ_p50 is not None
        and config.max_latency_ratio is not None
    ):
        if ratio_ci_lower > config.max_latency_ratio:
            violations.append(
                f"median latency ratio {latency_ratio:.2f}x "
                f"(95% CI [{ratio_ci_lower:.2f}x, {ratio_ci_upper:.2f}x]; "
                f"{p50:.2f}ms vs champion {champ_p50:.2f}ms) > budget {config.max_latency_ratio:.2f}x"
            )
        elif latency_ratio > config.max_latency_ratio and ratio_ci_lower > 1.0:
            warnings.append(
                f"median latency ratio {latency_ratio:.2f}x "
                f"(95% CI [{ratio_ci_lower:.2f}x, {ratio_ci_upper:.2f}x]; "
                f"{p50:.2f}ms vs champion {champ_p50:.2f}ms) exceeds {config.max_latency_ratio:.2f}x"
            )

    if config.p95_latency_ms is not None and p95 > config.p95_latency_ms:
        violations.append(f"p95 {p95:.1f}ms > budget {config.p95_latency_ms}ms")

    if model_size_mb > config.max_model_mb:
        violations.append(f"model size {model_size_mb:.1f}MB > budget {config.max_model_mb}MB")

    if peak_mem_mb > config.max_peak_mem_mb:
        violations.append(f"peak memory {peak_mem_mb:.1f}MB > budget {config.max_peak_mem_mb}MB")

    duration = time.time() - start

    metrics = {
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "model_size_mb": round(model_size_mb, 2),
        "peak_mem_mb": round(peak_mem_mb, 2),
        **champion_metrics,
    }

    status: Literal["pass", "warn", "fail", "skip"]
    if violations:
        status = "fail"
        summary = "; ".join(violations)
    elif warnings:
        status = "warn"
        summary = "; ".join(warnings)
    else:
        status = "pass"
        if (
            latency_ratio is not None
            and ratio_ci_lower is not None
            and ratio_ci_upper is not None
            and champ_p50 is not None
        ):
            summary = (
                f"p50 {p50:.2f}ms ({latency_ratio:.2f}x [95% CI {ratio_ci_lower:.2f}x-{ratio_ci_upper:.2f}x] "
                f"vs champion {champ_p50:.2f}ms), size {model_size_mb:.1f}MB, mem {peak_mem_mb:.1f}MB"
            )
        else:
            summary = f"p50 {p50:.1f}ms, p95 {p95:.1f}ms, size {model_size_mb:.1f}MB, mem {peak_mem_mb:.1f}MB"

    details: dict[str, Any] = {"violations": violations, "warnings": warnings}
    if latency_bootstrap_detail is not None:
        details["latency_bootstrap"] = latency_bootstrap_detail

    return CheckResult(
        name="budgets",
        status=status,
        summary=summary,
        metrics=metrics,
        details=details,
        duration_s=duration,
    )
