"""Tests for the PredGate configuration loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from predgate.config import (
    PredGateConfig,
    load_config,
)

VALID_YAML = textwrap.dedent("""\
    project: credit-risk-demo
    seed: 42
    model:
      entrypoint: demo/train.py:build_model
      task: classification
    data:
      train: data/train.csv
      test: data/test.csv
      reference: data/reference.csv
      target: default
    champion:
      mlflow_alias: champion
    checks:
      data_contract:
        enabled: true
        on_violation: fail
      drift:
        enabled: true
        method: psi
        warn_at: 0.1
        fail_at: 0.25
      slices:
        enabled: true
        metric: f1
        by: [region, age_bucket]
        min_slice_size: 100
        max_drop: 0.03
        confidence: 0.95
        n_bootstrap: 1000
      behavioral:
        enabled: true
        invariance:
          - feature: applicant_id
            perturb: random_id
            max_flip_rate: 0.0
        directional:
          - feature: income
            direction: increase
            expect: prob_not_decrease
      budgets:
        p95_latency_ms: 25
        max_model_mb: 50
        max_peak_mem_mb: 500
      fairness:
        enabled: true
        protected: [gender]
        metric: demographic_parity_diff
        warn_at: 0.05
        fail_at: 0.10
    verdict:
      fail_if_any_fail: true
      warn_requires_approval: true
""")

MINIMAL_YAML = textwrap.dedent("""\
    project: test
    seed: 1
    model:
      entrypoint: train.py:build
    data:
      train: train.csv
      test: test.csv
""")


class TestLoadConfig:
    """Tests for load_config."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Test loading a complete, valid config file."""
        config_file = tmp_path / "predgate.yaml"
        config_file.write_text(VALID_YAML)

        config = load_config(config_file)

        assert config.project == "credit-risk-demo"
        assert config.seed == 42
        assert config.model.entrypoint == "demo/train.py:build_model"
        assert config.model.task == "classification"
        assert config.data.train == "data/train.csv"
        assert config.data.target == "default"
        assert config.champion.mlflow_alias == "champion"

    def test_checks_loaded(self, tmp_path: Path) -> None:
        """Test that all check configs are properly loaded."""
        config_file = tmp_path / "predgate.yaml"
        config_file.write_text(VALID_YAML)

        config = load_config(config_file)

        assert config.checks.data_contract.enabled is True
        assert config.checks.data_contract.on_violation == "fail"
        assert config.checks.drift.method == "psi"
        assert config.checks.drift.warn_at == 0.1
        assert config.checks.drift.fail_at == 0.25
        assert config.checks.slices.metric == "f1"
        assert config.checks.slices.by == ["region", "age_bucket"]
        assert config.checks.slices.n_bootstrap == 1000
        assert len(config.checks.behavioral.invariance) == 1
        assert config.checks.behavioral.invariance[0].feature == "applicant_id"
        assert len(config.checks.behavioral.directional) == 1
        assert config.checks.budgets.p95_latency_ms == 25
        assert config.checks.fairness.protected == ["gender"]
        assert config.checks.fairness.metric == "demographic_parity_diff"

    def test_verdict_config(self, tmp_path: Path) -> None:
        """Test verdict configuration."""
        config_file = tmp_path / "predgate.yaml"
        config_file.write_text(VALID_YAML)

        config = load_config(config_file)

        assert config.verdict.fail_if_any_fail is True
        assert config.verdict.warn_requires_approval is True

    def test_minimal_config(self, tmp_path: Path) -> None:
        """Test loading a minimal config with all defaults."""
        config_file = tmp_path / "predgate.yaml"
        config_file.write_text(MINIMAL_YAML)

        config = load_config(config_file)

        assert config.project == "test"
        assert config.seed == 1
        assert config.model.entrypoint == "train.py:build"
        assert config.data.train == "train.csv"
        # Defaults
        assert config.model.task == "classification"
        assert config.checks.data_contract.enabled is True
        assert config.verdict.fail_if_any_fail is True

    def test_file_not_found(self) -> None:
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/predgate.yaml")

    def test_invalid_seed(self, tmp_path: Path) -> None:
        """Test that a negative seed is rejected."""
        config_file = tmp_path / "predgate.yaml"
        config_file.write_text(
            textwrap.dedent("""\
            project: test
            seed: -1
            model:
              entrypoint: train.py:build
            data:
              train: train.csv
              test: test.csv
        """)
        )

        with pytest.raises(ValueError, match="seed must be non-negative"):
            load_config(config_file)

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        """Test that missing required fields raise validation errors."""
        from pydantic import ValidationError

        config_file = tmp_path / "predgate.yaml"
        config_file.write_text("project: test\n")

        with pytest.raises(ValidationError):
            load_config(config_file)

    def test_invalid_task_type(self, tmp_path: Path) -> None:
        """Test that invalid task type is rejected."""
        from pydantic import ValidationError

        config_file = tmp_path / "predgate.yaml"
        config_file.write_text(
            textwrap.dedent("""\
            project: test
            seed: 42
            model:
              entrypoint: train.py:build
              task: unsupervised
            data:
              train: train.csv
              test: test.csv
        """)
        )

        with pytest.raises(ValidationError):
            load_config(config_file)


class TestPredGateConfig:
    """Direct model tests."""

    def test_roundtrip_dict(self) -> None:
        """Test config can be serialized and deserialized."""
        config = PredGateConfig(
            project="test",
            seed=42,
            model={"entrypoint": "train.py:build"},
            data={"train": "train.csv", "test": "test.csv"},
        )

        d = config.model_dump()
        config2 = PredGateConfig(**d)

        assert config.project == config2.project
        assert config.seed == config2.seed
        assert config.model.entrypoint == config2.model.entrypoint
