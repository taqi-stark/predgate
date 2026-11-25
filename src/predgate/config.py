"""PredGate configuration loader.

Parses and validates predgate.yaml with Pydantic v2 models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PresetName = Literal["strict", "balanced", "quiet"]

PRESET_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "strict": {
        "drift": {
            "warn_at": 0.08,
            "fail_at": 0.20,
            "on_new_feature": "warn",
            "point_mass_warn_delta": 0.08,
            "point_mass_fail_delta": 0.15,
        },
        "leakage": {
            "warn_corr": 0.40,
            "fail_corr": 0.65,
            "warn_auc": 0.80,
            "fail_auc": 0.88,
        },
        "label_shift": {
            "warn_delta": 0.03,
            "fail_delta": 0.07,
            "alpha": 0.01,
        },
        "slices": {
            "max_drop": 0.02,
            "max_ci_width": 0.08,
            "on_inconclusive": "warn",
        },
        "budgets": {
            "max_latency_ratio": 1.25,
        },
        "fairness": {
            "max_gap_increase": 0.015,
            "fail_gap_increase": 0.03,
            "selection_rate_shift_warn": 0.015,
        },
    },
    "balanced": {
        "drift": {
            "warn_at": 0.1,
            "fail_at": 0.25,
            "on_new_feature": "info",
            "point_mass_warn_delta": 0.10,
            "point_mass_fail_delta": 0.20,
        },
        "leakage": {
            "warn_corr": 0.50,
            "fail_corr": 0.75,
            "warn_auc": 0.85,
            "fail_auc": 0.92,
        },
        "label_shift": {
            "warn_delta": 0.05,
            "fail_delta": 0.10,
            "alpha": 0.01,
        },
        "slices": {
            "max_drop": 0.03,
            "max_ci_width": 0.10,
            "on_inconclusive": "warn",
        },
        "budgets": {
            "max_latency_ratio": 1.5,
        },
        "fairness": {
            "max_gap_increase": 0.02,
            "fail_gap_increase": 0.05,
            "selection_rate_shift_warn": 0.02,
        },
    },
    "quiet": {
        "drift": {
            "warn_at": 0.15,
            "fail_at": 0.30,
            "on_new_feature": "info",
            "point_mass_warn_delta": 0.15,
            "point_mass_fail_delta": 0.25,
        },
        "leakage": {
            "warn_corr": 0.55,
            "fail_corr": 0.80,
            "warn_auc": 0.88,
            "fail_auc": 0.94,
        },
        "label_shift": {
            "warn_delta": 0.07,
            "fail_delta": 0.12,
            "alpha": 0.01,
        },
        "slices": {
            "max_drop": 0.05,
            "max_ci_width": 0.15,
            "on_inconclusive": "info",
        },
        "budgets": {
            "max_latency_ratio": 2.0,
        },
        "fairness": {
            "max_gap_increase": 0.04,
            "fail_gap_increase": 0.08,
            "selection_rate_shift_warn": 0.05,
        },
    },
}


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
    on_new_feature: Literal["info", "warn"] = "info"
    point_mass_warn_delta: float = 0.10
    point_mass_fail_delta: float = 0.20


class SliceConfig(BaseModel):
    """Slice regression check settings."""

    enabled: bool = True
    metric: str = "f1"
    by: list[str] = Field(default_factory=list)
    min_slice_size: int = 100
    max_drop: float = 0.03
    max_ci_width: float = 0.10
    on_inconclusive: Literal["warn", "info"] = "warn"
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


class PredGateConfig(BaseModel):
    """Root configuration for PredGate."""

    project: str = "default"
    seed: int = 42
    preset: PresetName | None = None
    model: ModelConfig
    data: DataConfig
    champion: ChampionConfig = Field(default_factory=ChampionConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    checks: ChecksConfig = Field(default_factory=ChecksConfig)
    verdict: VerdictConfig = Field(default_factory=VerdictConfig)

    @model_validator(mode="before")
    @classmethod
    def apply_preset_defaults(cls, data: Any) -> Any:
        """Apply preset threshold defaults underneath any explicitly configured check fields."""
        if not isinstance(data, dict):
            return data
        preset_name = data.get("preset")
        if preset_name is None:
            return data
        if preset_name not in PRESET_CONFIGS:
            return data  # Let Pydantic field validation emit a clear Literal error

        merged = dict(data)
        raw_checks = merged.get("checks")
        if isinstance(raw_checks, ChecksConfig):
            checks_dict: dict[str, Any] = raw_checks.model_dump()
        elif isinstance(raw_checks, dict):
            checks_dict = {
                k: (v.model_dump() if isinstance(v, BaseModel) else dict(v))
                if isinstance(v, (dict, BaseModel))
                else v
                for k, v in raw_checks.items()
            }
        else:
            checks_dict = {}

        for check_name, preset_fields in PRESET_CONFIGS[preset_name].items():
            existing = checks_dict.get(check_name)
            if isinstance(existing, dict):
                combined = dict(preset_fields)
                combined.update(existing)
                checks_dict[check_name] = combined
            elif existing is None:
                checks_dict[check_name] = dict(preset_fields)

        merged["checks"] = checks_dict
        return merged

    @field_validator("seed")
    @classmethod
    def seed_must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("seed must be non-negative")
        return v


def get_preset_checks_config(preset: PresetName = "balanced") -> ChecksConfig:
    """Return a ChecksConfig populated with the named preset thresholds."""
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset '{preset}'. Expected one of: {list(PRESET_CONFIGS)}")
    return ChecksConfig.model_validate(PRESET_CONFIGS[preset])


def load_config(path: str | Path, preset_override: PresetName | None = None) -> PredGateConfig:
    """Load and validate a PredGate configuration from a YAML file.

    Args:
        path: Path to the predgate.yaml file.
        preset_override: Optional CLI preset override ('strict', 'balanced', or 'quiet').

    Returns:
        Validated PredGateConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    if preset_override is not None:
        raw["preset"] = preset_override

    return PredGateConfig(**raw)
