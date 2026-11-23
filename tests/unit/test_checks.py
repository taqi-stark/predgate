"""Unit tests for all 6 PredGate checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandera.pandas import Check, Column, DataFrameSchema
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from predgate.checks.behavioral import run_behavioral_check
from predgate.checks.budgets import run_budgets_check
from predgate.checks.data_contract import run_data_contract_check
from predgate.checks.drift import run_drift_check
from predgate.checks.fairness import run_fairness_check
from predgate.checks.label_shift import run_label_shift_check
from predgate.checks.leakage import run_leakage_check
from predgate.checks.slices import run_slices_check
from predgate.config import (
    BehavioralConfig,
    BudgetConfig,
    DataContractConfig,
    DirectionalTest,
    DriftConfig,
    FairnessConfig,
    InvarianceTest,
    LabelShiftConfig,
    LeakageConfig,
    SliceConfig,
)


@pytest.fixture
def sample_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create sample train, test, and reference datasets."""
    np.random.RandomState(42)
    n = 200

    def make_df(seed_offset: int = 0) -> pd.DataFrame:
        r = np.random.RandomState(42 + seed_offset)
        return pd.DataFrame(
            {
                "applicant_id": np.arange(1, n + 1),
                "income": r.uniform(20000, 100000, n),
                "credit_score": r.randint(500, 850, n),
                "region": r.choice(["North", "South", "East", "West"], n),
                "gender": r.choice(["male", "female"], n),
                "default": r.choice([0, 1], n, p=[0.8, 0.2]),
            }
        )

    return make_df(0), make_df(1), make_df(2)


def test_data_contract_check_passes(
    sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, test_df, _ = sample_dataset
    schema = DataFrameSchema(
        {
            "applicant_id": Column(int, Check.greater_than(0)),
            "income": Column(float, Check.greater_than(0)),
            "credit_score": Column(int, Check.in_range(300, 850)),
            "default": Column(int, Check.isin([0, 1])),
        },
        strict=False,
    )

    cfg = DataContractConfig(enabled=True, on_violation="fail")
    res = run_data_contract_check(train_df, test_df, schema, cfg)
    assert res.status == "pass"
    assert res.metrics["violations"] == 0


def test_data_contract_check_violation(
    sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, test_df, _ = sample_dataset
    bad_train = train_df.copy()
    bad_train.loc[0, "income"] = -500  # violates > 0 check

    schema = DataFrameSchema(
        {
            "income": Column(float, Check.greater_than(0)),
        },
        strict=False,
    )

    cfg = DataContractConfig(enabled=True, on_violation="fail")
    res = run_data_contract_check(bad_train, test_df, schema, cfg)
    assert res.status == "fail"
    assert res.metrics["violations"] > 0


def test_drift_check_no_drift(
    sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, _, _ = sample_dataset
    ref_df = train_df.copy()
    cfg = DriftConfig(enabled=True, warn_at=0.1, fail_at=0.25)
    res = run_drift_check(train_df, ref_df, "default", cfg)
    assert res.status == "pass"
    assert res.metrics["psi_max"] < 0.1


def test_drift_check_detects_drift(
    sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, _, ref_df = sample_dataset
    drifted_train = train_df.copy()
    drifted_train["income"] = drifted_train["income"] * 10  # huge scale shift

    cfg = DriftConfig(enabled=True, warn_at=0.1, fail_at=0.25)
    res = run_drift_check(drifted_train, ref_df, "default", cfg)
    assert res.status in ("warn", "fail")
    assert res.metrics["psi_max"] > 0.1


def test_slices_check(sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df, _ = sample_dataset
    X_train = train_df[["income", "credit_score"]]
    y_train = train_df["default"]
    test_df[["income", "credit_score", "region"]]
    test_df["default"]

    # Champion and candidate identical
    champ = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    cand = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

    cfg = SliceConfig(
        enabled=True,
        metric="accuracy",
        by=["region"],
        min_slice_size=10,
        max_drop=0.05,
        confidence=0.95,
        n_bootstrap=100,
    )

    res = run_slices_check(test_df, "default", champ, cand, cfg, seed=42)
    assert res.status == "pass"


def test_behavioral_check_invariance() -> None:
    df = pd.DataFrame(
        {
            "applicant_id": [1, 2, 3, 4, 5],
            "income": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
            "default": [0, 0, 1, 0, 1],
        }
    )

    # Model that ignores applicant_id
    model = LogisticRegression().fit(df[["income"]], df["default"])

    class ModelWrapper:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return model.predict(X[["income"]])

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return model.predict_proba(X[["income"]])

    cfg = BehavioralConfig(
        enabled=True,
        invariance=[InvarianceTest(feature="applicant_id", perturb="random_id", max_flip_rate=0.0)],
        directional=[],
    )

    res = run_behavioral_check(df, "default", ModelWrapper(), cfg)
    assert res.status == "pass"


def test_budgets_check() -> None:
    df = pd.DataFrame(
        {
            "income": [50000.0, 60000.0],
            "credit_score": [650, 700],
            "default": [0, 1],
        }
    )
    model = LogisticRegression().fit(df[["income", "credit_score"]], df["default"])

    class BudgetModelWrapper:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return model.predict(X[["income", "credit_score"]])

    cfg = BudgetConfig(
        enabled=True,
        p95_latency_ms=500.0,
        max_model_mb=100.0,
        max_peak_mem_mb=1000.0,
    )

    res = run_budgets_check(df, "default", BudgetModelWrapper(), None, cfg)
    assert res.status == "pass"
    assert "p95_ms" in res.metrics


def test_fairness_check(sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df, _ = sample_dataset
    model = DummyClassifier(strategy="most_frequent").fit(train_df[["income"]], train_df["default"])

    class DummyWrapper:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return model.predict(X[["income"]])

    cfg = FairnessConfig(
        enabled=True,
        protected=["gender"],
        metric="demographic_parity_diff",
        warn_at=0.1,
        fail_at=0.2,
    )

    res = run_fairness_check(test_df, "default", DummyWrapper(), None, cfg)
    assert res.status == "pass"


def test_slices_check_fail(sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    _, test_df, _ = sample_dataset

    # Champion predicts ground truth perfectly
    class PerfectChampion:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return test_df.loc[X.index, "default"].values

    # Candidate always predicts 0
    class DegradedCandidate:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(X), dtype=int)

    cfg = SliceConfig(
        enabled=True,
        metric="accuracy",
        by=["region"],
        min_slice_size=5,
        max_drop=0.01,
        confidence=0.95,
        n_bootstrap=100,
    )

    res = run_slices_check(test_df, "default", PerfectChampion(), DegradedCandidate(), cfg, seed=42)
    assert res.status == "fail"
    assert res.metrics["n_failing_slices"] > 0


def test_slices_check_no_champion_skip(
    sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, test_df, _ = sample_dataset
    candidate = DummyClassifier().fit(test_df[["income"]], test_df["default"])
    cfg = SliceConfig(enabled=True, by=["region"])
    res = run_slices_check(test_df, "default", None, candidate, cfg)
    assert res.status == "skip"
    assert "No champion" in res.summary


def test_budgets_check_fail_latency() -> None:
    df = pd.DataFrame({"income": [50000.0], "default": [0]})

    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def advance(self, seconds: float) -> None:
            self.now += seconds

        def __call__(self) -> float:
            return self.now

    clock = FakeClock()

    class SlowModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            clock.advance(0.005)  # deterministic 5.0ms per call
            return np.array([0])

    # Optional absolute ms ceiling enforced when explicitly set
    cfg = BudgetConfig(enabled=True, p95_latency_ms=2.0, latency_samples=10, latency_batch_size=3)
    res = run_budgets_check(df, "default", SlowModel(), None, cfg, seed=42, clock=clock)
    assert res.status == "fail"
    assert "p95" in res.summary
    assert res.metrics["p95_ms"] == 5.0

    # When p95_latency_ms is None (default) and no champion is present, absolute ms does not fail
    cfg_opt = BudgetConfig(
        enabled=True, p95_latency_ms=None, latency_samples=10, latency_batch_size=3
    )
    res_opt = run_budgets_check(df, "default", SlowModel(), None, cfg_opt, seed=42, clock=clock)
    assert res_opt.status == "pass"


def test_budgets_check_ratio_gate_same_job() -> None:
    """Deterministic paired median-ratio gate with injected clock and paired bootstrap CI."""
    df = pd.DataFrame({"income": [50000.0], "default": [0]})

    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def advance(self, seconds: float) -> None:
            self.now += seconds

        def __call__(self) -> float:
            return self.now

    class ScriptedModel:
        def __init__(self, clock: FakeClock, durations_ms: list[float]) -> None:
            self.clock = clock
            self.durations_ms = durations_ms
            self.idx = 0

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            d = self.durations_ms[self.idx % len(self.durations_ms)]
            self.idx += 1
            self.clock.advance(d / 1000.0)
            return np.array([0])

    # 1. Clear regression: candidate 2.5ms vs champion 1.0ms (ratio 2.5x, CI [2.5x, 2.5x] > 1.5x) -> FAIL
    clock1 = FakeClock()
    champ_fast = ScriptedModel(clock1, [1.0])
    cand_slow = ScriptedModel(clock1, [2.5])
    cfg_fail = BudgetConfig(
        enabled=True,
        max_latency_ratio=1.5,
        p95_latency_ms=None,
        latency_samples=20,
        latency_batch_size=5,
        latency_n_bootstrap=500,
    )
    res_fail = run_budgets_check(
        df, "default", cand_slow, champ_fast, cfg_fail, seed=42, clock=clock1
    )
    assert res_fail.status == "fail"
    assert "median latency ratio" in res_fail.summary
    assert res_fail.metrics["median_latency_ratio"] == pytest.approx(2.5, abs=1e-3)
    assert res_fail.metrics["latency_ratio_ci_lower"] > 1.5
    assert "champion_p50_ms" in res_fail.metrics

    # 2. Equal speed models (ratio 1.0x <= 1.5x) -> PASS
    clock2 = FakeClock()
    res_pass = run_budgets_check(
        df,
        "default",
        ScriptedModel(clock2, [1.0]),
        ScriptedModel(clock2, [1.0]),
        cfg_fail,
        seed=42,
        clock=clock2,
    )
    assert res_pass.status == "pass"
    assert "vs champion" in res_pass.summary
    assert res_pass.metrics["median_latency_ratio"] == pytest.approx(1.0, abs=1e-3)

    # 3. Single outlier spike on champion (1 of 20 batches) must NOT break median ratio gate
    clock3 = FakeClock()
    champ_spike = ScriptedModel(clock3, [1.0] * 19 + [50.0])
    cand_slow3 = ScriptedModel(clock3, [2.5])
    res_robust = run_budgets_check(
        df,
        "default",
        cand_slow3,
        champ_spike,
        BudgetConfig(
            enabled=True,
            max_latency_ratio=1.5,
            p95_latency_ms=None,
            latency_samples=20,
            latency_batch_size=1,
            latency_n_bootstrap=500,
        ),
        seed=42,
        clock=clock3,
    )
    assert res_robust.status == "fail"
    assert res_robust.metrics["median_latency_ratio"] == pytest.approx(2.5, abs=1e-3)

    # 4. Borderline noisy slowdown (ratio > 1.5, CI lower > 1.0 but overlapping 1.5) -> WARN
    clock4 = FakeClock()
    champ_b = ScriptedModel(clock4, [1.0])
    # Warmup consumes 5 calls; then 20 batches of size 1: 8 at 1.2ms, 12 at 1.7ms -> median 1.7x, CI overlaps 1.5
    cand_b = ScriptedModel(clock4, [1.0] * 5 + [1.2] * 8 + [1.7] * 12)
    res_warn = run_budgets_check(
        df,
        "default",
        cand_b,
        champ_b,
        BudgetConfig(
            enabled=True,
            max_latency_ratio=1.5,
            p95_latency_ms=None,
            latency_samples=20,
            latency_batch_size=1,
            latency_n_bootstrap=500,
        ),
        seed=42,
        clock=clock4,
    )
    assert res_warn.status == "warn"
    assert res_warn.metrics["latency_ratio_ci_lower"] > 1.0
    assert res_warn.metrics["latency_ratio_ci_lower"] <= 1.5


class BigPicklableModel:
    def __init__(self) -> None:
        self.weights = np.zeros(200_000)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


def test_budgets_check_fail_size() -> None:
    df = pd.DataFrame({"income": [50000.0], "default": [0]})
    cfg = BudgetConfig(enabled=True, max_model_mb=0.01)
    res = run_budgets_check(df, "default", BigPicklableModel(), None, cfg)
    assert res.status == "fail"
    assert "model size" in res.summary


def test_fairness_check_fail() -> None:
    # Dataset with protected attribute
    df = pd.DataFrame(
        {
            "gender": ["male"] * 50 + ["female"] * 50,
            "income": [50000.0] * 100,
            "default": [0] * 100,
        }
    )

    # Disparate model: predicts 1 for males and 0 for females
    class BiasedModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return (X["gender"] == "male").astype(int).values

    cfg = FairnessConfig(
        enabled=True,
        protected=["gender"],
        metric="demographic_parity_diff",
        warn_at=0.1,
        fail_at=0.2,
    )
    res = run_fairness_check(df, "default", BiasedModel(), None, cfg)
    assert res.status == "fail"
    assert "demographic_parity_diff" in res.summary


def test_behavioral_check_fail_invariance() -> None:
    df = pd.DataFrame(
        {
            "applicant_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "income": [50000.0] * 8,
            "default": [0] * 8,
        }
    )

    # Model sensitive to applicant_id (should be invariant)
    class IDSensitiveModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return (X["applicant_id"] % 2 == 0).astype(int).values

    cfg = BehavioralConfig(
        enabled=True,
        invariance=[InvarianceTest(feature="applicant_id", perturb="random_id", max_flip_rate=0.0)],
        directional=[],
    )
    res = run_behavioral_check(df, "default", IDSensitiveModel(), cfg)
    assert res.status == "fail"
    assert res.metrics["n_fail"] > 0


def test_behavioral_check_directional() -> None:
    df = pd.DataFrame(
        {
            "debt": [0.1, 0.2, 0.5, 0.8],
            "default": [0, 0, 1, 1],
        }
    )

    class MonotonicModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return (X["debt"] > 0.4).astype(int).values

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            p = X["debt"].values
            return np.column_stack([1 - p, p])

    cfg_pass = BehavioralConfig(
        enabled=True,
        directional=[DirectionalTest(feature="debt", direction="increase", expect="prob_increase")],
    )
    res_pass = run_behavioral_check(df, "default", MonotonicModel(), cfg_pass)
    assert res_pass.status == "pass"

    # Expect probability to decrease when debt increases -> fail
    cfg_fail = BehavioralConfig(
        enabled=True,
        directional=[DirectionalTest(feature="debt", direction="increase", expect="prob_decrease")],
    )
    res_fail = run_behavioral_check(df, "default", MonotonicModel(), cfg_fail)
    assert res_fail.status == "fail"


def test_data_contract_edge_cases(
    sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, test_df, _ = sample_dataset

    # Disabled
    res_disabled = run_data_contract_check(
        train_df, test_df, "dummy:schema", DataContractConfig(enabled=False)
    )
    assert res_disabled.status == "skip"

    # No schema
    res_no_schema = run_data_contract_check(train_df, test_df, "", DataContractConfig(enabled=True))
    assert res_no_schema.status == "skip"

    # Schema import error
    res_bad_import = run_data_contract_check(
        train_df, test_df, "nonexistent.mod:schema", DataContractConfig(enabled=True)
    )
    assert res_bad_import.status == "fail"


def test_fairness_equal_opportunity(
    sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, test_df, _ = sample_dataset
    model = DummyClassifier(strategy="most_frequent").fit(train_df[["income"]], train_df["default"])

    class ModelWrapper:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return model.predict(X[["income"]])

    cfg = FairnessConfig(
        enabled=True,
        protected=["gender"],
        metric="equal_opportunity_diff",
        warn_at=0.1,
        fail_at=0.2,
    )
    res = run_fairness_check(test_df, "default", ModelWrapper(), None, cfg)
    assert res.status == "pass"

    # Missing protected column -> skipped
    cfg_missing = FairnessConfig(enabled=True, protected=["nonexistent_col"])
    res_missing = run_fairness_check(test_df, "default", ModelWrapper(), None, cfg_missing)
    assert res_missing.status == "pass"
    assert res_missing.details["groups"][0]["status"] == "skip"

    # Disabled
    res_disabled = run_fairness_check(
        test_df, "default", ModelWrapper(), None, FairnessConfig(enabled=False)
    )
    assert res_disabled.status == "skip"


def test_fairness_paired_bootstrap_champion_comparison() -> None:
    """Champion vs candidate paired bootstrap CI flags candidate that widens calibration/parity gap."""
    rng = np.random.RandomState(123)
    n = 300
    df = pd.DataFrame(
        {
            "gender": ["male"] * (n // 2) + ["female"] * (n // 2),
            "income": rng.uniform(20000, 100000, n),
            "default": rng.binomial(1, 0.1, size=n),
        }
    )

    class FairChampion:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.column_stack([np.full(len(X), 0.9), np.full(len(X), 0.1)])

    class MisCalibratedCandidate:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            p1 = np.where(X["gender"] == "female", 0.38, 0.10)
            return np.column_stack([1.0 - p1, p1])

    cfg = FairnessConfig(
        enabled=True,
        protected=["gender"],
        metric="demographic_parity_diff",
        warn_at=0.05,
        fail_at=0.10,
    )
    res = run_fairness_check(df, "default", MisCalibratedCandidate(), FairChampion(), cfg, seed=42)
    assert res.status == "fail"
    assert "calibration_gap_diff" in res.summary
    group_detail = res.details["groups"][0]
    assert "all_gap_bootstraps" in group_detail
    assert group_detail["all_gap_bootstraps"]["calibration_gap_diff"]["ci_lower"] > 0.10


def test_fairness_default_relative_gate_and_power_note() -> None:
    """Default champion-vs-candidate gate does not fail on pre-existing DP if candidate does not widen it;
    opt-in enforce_absolute evaluates excess relative to champion's gap; small groups emit MDE/power note.
    """
    from predgate.report import build_html_report, build_pr_comment
    from predgate.verdict import compute_verdict

    # Small group dataset: 15 males (all predicted 1), 15 females (all predicted 0) -> DP = 1.0
    df = pd.DataFrame(
        {
            "gender": ["male"] * 15 + ["female"] * 15,
            "income": [50000.0] * 30,
            "default": [1, 0, 1, 0, 1] * 6,
        }
    )

    class StaticDisparateModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return (X["gender"] == "male").astype(int).values

    # Both champion and candidate have identical DP=1.0 -> Δ_gap = 0.0 -> PASS by default
    cfg_default = FairnessConfig(
        enabled=True,
        protected=["gender"],
        warn_at=0.05,
        fail_at=0.10,
        enforce_absolute=False,
    )
    res_pass = run_fairness_check(
        df, "default", StaticDisparateModel(), StaticDisparateModel(), cfg_default, seed=42
    )
    assert res_pass.status == "pass"
    assert res_pass.metrics["insufficient_power"] == 1.0
    assert "minimum detectable gap / insufficient power" in res_pass.summary
    assert len(res_pass.details["power_notes"]) == 1

    verdict = compute_verdict([res_pass])
    pr_md = build_pr_comment(verdict)
    html_doc = build_html_report(verdict)
    assert "minimum detectable gap / insufficient power" in pr_md
    assert "Statistical Power Notes" in html_doc


def test_fairness_enforce_absolute_opt_in_relative_to_champion() -> None:
    """Opt-in enforce_absolute=True gates on excess above champion's own gap."""
    n_per_group = 200
    df = pd.DataFrame(
        {
            "gender": ["male"] * n_per_group + ["female"] * n_per_group,
            "idx": list(range(n_per_group)) + list(range(n_per_group)),
            "default": [0, 1] * n_per_group,
        }
    )

    # Champion: male pos rate = 0.40, female pos rate = 0.10 -> DP = 0.30
    class ChampModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            is_m = X["gender"].values == "male"
            idx = X["idx"].values
            return np.where(is_m, idx < 80, idx < 20).astype(int)

    # Candidate A: identical to champion (DP = 0.30, excess over champion = 0.0)
    # Candidate B: male pos rate = 0.52, female pos rate = 0.10 -> DP = 0.42 (excess +0.12 over champion)
    class CandWiderDP:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            is_m = X["gender"].values == "male"
            idx = X["idx"].values
            return np.where(is_m, idx < 104, idx < 20).astype(int)

    # Candidate C: male pos rate = 0.46, female pos rate = 0.10 -> DP = 0.36 (excess +0.06 over champion)
    class CandSlightlyWiderDP:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            is_m = X["gender"].values == "male"
            idx = X["idx"].values
            return np.where(is_m, idx < 92, idx < 20).astype(int)

    # Set high bootstrap thresholds so we isolate enforce_absolute behavior
    cfg_abs = FairnessConfig(
        enabled=True,
        protected=["gender"],
        metric="demographic_parity_diff",
        enforce_absolute=True,
        warn_at=0.05,
        fail_at=0.10,
        max_gap_increase=0.50,
        fail_gap_increase=0.50,
        selection_rate_shift_warn=0.50,
    )
    # 1. Pre-existing champion DP=0.30 with 0 excess -> PASS even with enforce_absolute=True
    res_same = run_fairness_check(df, "default", ChampModel(), ChampModel(), cfg_abs, seed=42)
    assert res_same.status == "pass"
    assert res_same.details["enforce_absolute"] is True
    assert res_same.details["groups"][0]["relative_excess_over_champion"] == 0.0

    # 2. Excess +0.06 >= warn_at (0.05) -> WARN
    res_warn = run_fairness_check(
        df, "default", CandSlightlyWiderDP(), ChampModel(), cfg_abs, seed=42
    )
    assert res_warn.status == "warn"
    assert res_warn.details["groups"][0]["relative_excess_over_champion"] == pytest.approx(
        0.06, abs=1e-3
    )

    # 3. Excess +0.12 >= fail_at (0.10) -> FAIL
    res_fail = run_fairness_check(df, "default", CandWiderDP(), ChampModel(), cfg_abs, seed=42)
    assert res_fail.status == "fail"
    assert res_fail.details["groups"][0]["relative_excess_over_champion"] == pytest.approx(
        0.12, abs=1e-3
    )

    # 4. With enforce_absolute=False (default), same config does NOT trigger static limit
    cfg_no_abs = cfg_abs.model_copy(update={"enforce_absolute": False})
    res_opt_out = run_fairness_check(
        df, "default", CandWiderDP(), ChampModel(), cfg_no_abs, seed=42
    )
    assert res_opt_out.status == "pass"


def test_fairness_per_group_selection_rate_shift_two_sided_warn() -> None:
    """Two-sided paired bootstrap per-group selection-rate shift emits WARN even when DP gap narrows."""
    n_per_group = 300
    df = pd.DataFrame(
        {
            "gender": ["male"] * n_per_group + ["female"] * n_per_group,
            "idx": list(range(n_per_group)) + list(range(n_per_group)),
            "default": ([1] * 60 + [0] * 240) + ([1] * 60 + [0] * 240),
        }
    )

    # Champion: male pos rate = 0.25 (75/300), female pos rate = 0.05 (15/300) -> DP = 0.20
    class ChampModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            is_m = X["gender"].values == "male"
            idx = X["idx"].values
            return np.where(is_m, idx < 75, idx < 15).astype(int)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            p = np.full(len(X), 0.20)
            return np.column_stack([1.0 - p, p])

    # Candidate shifts female selection rate UP (0.05 -> 0.15, +0.10), NARROWING DP (0.20 -> 0.10),
    # while keeping probabilities unchanged and TPR/FPR equalized odds narrower (female TPR 0.25->0.75 vs male 1.0).
    class FemaleUpCandidate:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            is_m = X["gender"].values == "male"
            idx = X["idx"].values
            return np.where(is_m, idx < 75, idx < 45).astype(int)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            p = np.full(len(X), 0.20)
            return np.column_stack([1.0 - p, p])

    # Candidate shifts male selection rate DOWN (0.25 -> 0.15, -0.10), ALSO narrowing DP (0.20 -> 0.10)
    class MaleDownCandidate:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            is_m = X["gender"].values == "male"
            idx = X["idx"].values
            return np.where(is_m, idx < 45, idx < 15).astype(int)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            p = np.full(len(X), 0.20)
            return np.column_stack([1.0 - p, p])

    cfg = FairnessConfig(
        enabled=True,
        protected=["gender"],
        metric="demographic_parity_diff",
        selection_rate_shift_warn=0.02,
    )

    # Positive selection-rate shift on female -> WARN (not FAIL)
    res_up = run_fairness_check(df, "default", FemaleUpCandidate(), ChampModel(), cfg, seed=42)
    assert res_up.status == "warn"
    assert "Selection-rate shift" in res_up.summary
    assert "gender=female" in res_up.summary
    assert res_up.metrics["max_selection_rate_shift"] == pytest.approx(0.10, abs=1e-3)

    # Negative selection-rate shift on male -> WARN (two-sided!)
    res_down = run_fairness_check(df, "default", MaleDownCandidate(), ChampModel(), cfg, seed=42)
    assert res_down.status == "warn"
    assert "Selection-rate shift" in res_down.summary
    assert "gender=male" in res_down.summary
    assert res_down.metrics["max_selection_rate_shift"] == pytest.approx(0.10, abs=1e-3)

    # Identical candidate -> PASS
    res_noop = run_fairness_check(df, "default", ChampModel(), ChampModel(), cfg, seed=42)
    assert res_noop.status == "pass"
    assert res_noop.metrics["max_selection_rate_shift"] == 0.0


def test_behavioral_min_functionality(tmp_path: Path) -> None:
    csv_file = tmp_path / "mf.csv"
    csv_file.write_text("income,default,expected\n50000.0,0,0\n60000.0,1,1\n")

    class SimpleModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.array([0, 0])  # second case fails

    cfg = BehavioralConfig(enabled=True, min_functionality_csv=str(csv_file))
    df = pd.DataFrame({"income": [50000.0], "default": [0]})
    res = run_behavioral_check(df, "default", SimpleModel(), cfg)
    assert res.status == "fail"
    assert res.metrics["n_fail"] > 0


def test_leakage_check_pass_and_fail(
    sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, test_df, ref_df = sample_dataset
    cfg = LeakageConfig(enabled=True)

    # Clean data passes
    res_clean = run_leakage_check(train_df, test_df, "default", cfg, reference_df=ref_df)
    assert res_clean.status == "pass"
    assert res_clean.metrics["n_leaky_features"] == 0.0

    # Inject target-derived column -> fails
    rng = np.random.RandomState(42)
    leaky_train = train_df.copy()
    leaky_train["leaked_target"] = leaky_train["default"] * 0.9 + rng.normal(
        0, 0.1, len(leaky_train)
    )
    res_leaky = run_leakage_check(leaky_train, test_df, "default", cfg, reference_df=ref_df)
    assert res_leaky.status == "fail"
    assert "leaked_target" in res_leaky.summary
    assert res_leaky.metrics["max_abs_corr"] >= 0.75


def test_label_shift_check_pass_and_fail(
    sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, _, _ = sample_dataset
    ref_df = train_df.copy()
    cfg = LabelShiftConfig(enabled=True, by=["region", "gender"], min_slice_size=20)

    # Identical label distribution passes
    res_clean = run_label_shift_check(train_df, ref_df, "default", cfg)
    assert res_clean.status == "pass"

    # Flip 60% of labels in North region -> fails
    shifted = train_df.copy()
    north_mask = shifted["region"] == "North"
    shifted.loc[north_mask, "default"] = 1
    res_shifted = run_label_shift_check(shifted, ref_df, "default", cfg)
    assert res_shifted.status == "fail"
    assert "region=North" in res_shifted.summary
    assert res_shifted.metrics["max_slice_label_delta"] >= 0.10
