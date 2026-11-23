"""ModelGate configuration loader.

Parses and validates modelgate.yaml with Pydantic v2 models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelConfig(BaseModel):
    """Model training configuration."""

    entrypoint: str = Field(
        description="Module path to the model builder, e.g. 'demo/train.py:build_model'"
    )
    task: Literal["classification", "regression"] = "classification"


class DataConfig(BaseModel):
    """Data paths and schema configuration."""

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    train: str
    test: str
    reference: str = ""
    target: str = "default"
    schema_: str = Field(default="", alias="schema")

    @property
    def schema(self) -> str:  # type: ignore[override]
        return self.schema_


class ChampionConfig(BaseModel):
    """Champion model configuration."""

    mlflow_alias: str = "champion"
    baseline_dir: str = "champion"


class MLflowConfig(BaseModel):
    """MLflow tracking and registry configuration."""

    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: str = "./mlruns"


class DataContractConfig(BaseModel):
    """Data contract check settings."""

    enabled: bool = True
    on_violation: Literal["fail", "warn"] = "fail"


class DriftConfig(BaseModel):
    """Drift detection check settings."""

    enabled: bool = True
    method: Literal["psi", "ks"] = "psi"
    warn_at: float = 0.1
    fail_at: float = 0.25


class SliceConfig(BaseModel):
    """Slice regression check settings."""

    enabled: bool = True
    metric: str = "f1"
    by: list[str] = Field(default_factory=list)
    min_slice_size: int = 100
    max_drop: float = 0.03
    confidence: float = 0.95
    n_bootstrap: int = 1000


class InvarianceTest(BaseModel):
    """Invariance behavioral test definition."""

    feature: str
    perturb: str
    max_flip_rate: float = 0.0


class DirectionalTest(BaseModel):
    """Directional behavioral test definition."""

    feature: str
    direction: Literal["increase", "decrease"]
    expect: str  # e.g. 'prob_not_decrease'


class BehavioralConfig(BaseModel):
    """Behavioral check settings."""

    enabled: bool = True
    invariance: list[InvarianceTest] = Field(default_factory=list)
    directional: list[DirectionalTest] = Field(default_factory=list)
    min_functionality_csv: str = ""


class BudgetConfig(BaseModel):
    """Performance budget check settings."""

    enabled: bool = True
    max_latency_ratio: float | None = 1.5
    latency_samples: int = 20
    latency_batch_size: int = 5
    latency_confidence: float = 0.95
    latency_n_bootstrap: int = 1000
    p95_latency_ms: float | None = None
    max_model_mb: float = 50.0
    max_peak_mem_mb: float = 500.0


class FairnessConfig(BaseModel):
    """Fairness check settings."""

    enabled: bool = True
    protected: list[str] = Field(default_factory=list)
    metric: str = "demographic_parity_diff"
    warn_at: float = 0.05
    fail_at: float = 0.10
    enforce_absolute: bool = False
    max_gap_increase: float = 0.02
    fail_gap_increase: float = 0.05
    selection_rate_shift_warn: float = 0.02
    selection_rate_n_bootstrap: int = 1000
    min_group_size: int = 100
    min_group_positives: int = 30
    confidence: float = 0.95
    n_bootstrap: int = 200


class LeakageConfig(BaseModel):
    """Target leakage check settings."""

    enabled: bool = True
    warn_corr: float = 0.50
    fail_corr: float = 0.75
    warn_auc: float = 0.85
    fail_auc: float = 0.92


class LabelShiftConfig(BaseModel):
    """Label-shift check settings."""

    enabled: bool = True
    by: list[str] = Field(default_factory=lambda: ["region", "gender", "age_bucket"])
    min_slice_size: int = 30
    warn_delta: float = 0.05
    fail_delta: float = 0.10
    alpha: float = 0.01


class ChecksConfig(BaseModel):
    """All check configurations."""

    data_contract: DataContractConfig = Field(default_factory=DataContractConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    leakage: LeakageConfig = Field(default_factory=LeakageConfig)
    label_shift: LabelShiftConfig = Field(default_factory=LabelShiftConfig)
    slices: SliceConfig = Field(default_factory=SliceConfig)
    behavioral: BehavioralConfig = Field(default_factory=BehavioralConfig)
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    fairness: FairnessConfig = Field(default_factory=FairnessConfig)


class VerdictConfig(BaseModel):
    """Verdict engine configuration."""

    fail_if_any_fail: bool = True
    warn_requires_approval: bool = True


class ModelGateConfig(BaseModel):
    """Root configuration for ModelGate."""

    project: str = "default"
    seed: int = 42
    model: ModelConfig
    data: DataConfig
    champion: ChampionConfig = Field(default_factory=ChampionConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    checks: ChecksConfig = Field(default_factory=ChecksConfig)
    verdict: VerdictConfig = Field(default_factory=VerdictConfig)

    @field_validator("seed")
    @classmethod
    def seed_must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("seed must be non-negative")
        return v


def load_config(path: str | Path) -> ModelGateConfig:
    """Load and validate a ModelGate configuration from a YAML file.

    Args:
        path: Path to the modelgate.yaml file.

    Returns:
        Validated ModelGateConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return ModelGateConfig(**raw)
