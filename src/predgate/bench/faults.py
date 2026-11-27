"""Synthetic dataset generation, fault injection, and baseline comparators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from predgate.stats import demographic_parity_diff

FAULT_DESCRIPTIONS = {
    "feature_leakage": "Add a column derived from the target",
    "label_shift": "Corrupt labels in one subgroup of the train set",
    "degraded_slice": "Remove most of one subgroup from training data",
    "schema_drift": "Change a column's scale or unit",
    "latency_regression": "Swap in a much larger model",
    "fairness_regression": "Bias sampling against one protected group",
    "fairness_threshold_shift": "Shift decision threshold for one protected group (probabilities unchanged)",
    "fairness_proxy_feature": "Drop protected attribute and bias training via a correlated proxy feature",
    "noop": "Retrain with a different seed (should PASS)",
}


def per_group_accuracy_drop_baseline(
    y_true: np.ndarray,
    champ_preds: np.ndarray,
    cand_preds: np.ndarray,
    test_df: pd.DataFrame,
    group_cols: list[str],
    max_drop: float = 0.02,
    min_group_size: int = 30,
) -> bool:
    """Flag if any subgroup's accuracy drops by more than max_drop."""
    y_arr = np.asarray(y_true)
    ch_arr = np.asarray(champ_preds)
    ca_arr = np.asarray(cand_preds)
    for col in group_cols:
        if col not in test_df.columns:
            continue
        vals = test_df[col].astype(str).values
        for g in np.unique(vals):
            mask = vals == g
            if int(np.sum(mask)) < min_group_size:
                continue
            acc_ch = float(np.mean(y_arr[mask] == ch_arr[mask]))
            acc_ca = float(np.mean(y_arr[mask] == ca_arr[mask]))
            if (acc_ch - acc_ca) > max_drop:
                return True
    return False


def fixed_dp_limit_baseline(
    cand_preds: np.ndarray,
    protected_values: np.ndarray,
    dp_limit: float = 0.10,
) -> bool:
    """Flag if candidate demographic parity difference meets or exceeds dp_limit."""
    return bool(
        demographic_parity_diff(np.asarray(cand_preds, dtype=int), protected_values) >= dp_limit
    )


def _generate_data(seed: int = 42, n: int = 3000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic credit train and test DataFrames."""
    rng = np.random.RandomState(seed)

    def _make_df(n: int, rng: np.random.RandomState) -> pd.DataFrame:
        income = rng.lognormal(10.5, 0.5, n)
        age = rng.randint(18, 70, n)
        credit = rng.normal(680, 80, n).clip(300, 850).astype(int)
        debt = rng.beta(2, 5, n)
        emp = rng.exponential(5, n).clip(0, 40).astype(int)
        region = rng.choice(["North", "South", "East", "West"], n)
        gender = rng.choice(["male", "female"], n, p=[0.55, 0.45])
        age_bucket = np.where(age < 30, "young", np.where(age < 50, "middle", "senior"))

        logit = -2 + 0.3 * (debt - 0.3) - 0.001 * (income - 30000) / 10000 - 0.005 * (credit - 600)
        logit += np.where(region == "North", 0.1, 0)
        prob = 1 / (1 + np.exp(-logit))
        default = (rng.random(n) < prob).astype(int)

        return pd.DataFrame(
            {
                "income": income,
                "age": age,
                "credit_score": credit,
                "debt_ratio": debt,
                "employment_years": emp,
                "region": region,
                "age_bucket": age_bucket,
                "gender": gender,
                "default": default,
            }
        )

    train = _make_df(n, rng)
    test_rng = np.random.RandomState(seed + 1000)
    test = _make_df(n // 2, test_rng)
    return train, test


def _train_model(X: pd.DataFrame, y: pd.Series, seed: int) -> Pipeline:
    """Train a gradient boosting pipeline."""
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=seed),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def _train_latency_model(X: pd.DataFrame, y: pd.Series, seed: int) -> Pipeline:
    """Train a large random forest pipeline with high inference latency."""
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=300, max_depth=15, random_state=seed),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


class GroupThresholdModel:
    """Wrapper shifting the binary decision threshold for one protected group."""

    def __init__(
        self,
        base_model: Any,
        protected_col: str,
        target_group: str,
        group_threshold: float = 0.25,
        default_threshold: float = 0.50,
    ) -> None:
        self.base_model = base_model
        self.protected_col = protected_col
        self.target_group = target_group
        self.group_threshold = group_threshold
        self.default_threshold = default_threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.base_model.predict_proba(X)[:, 1]
        if self.protected_col in X.columns:
            thresh = np.where(
                X[self.protected_col].astype(str).values == self.target_group,
                self.group_threshold,
                self.default_threshold,
            )
        else:
            thresh = np.full(len(X), self.default_threshold)
        return (probs >= thresh).astype(int)


def inject_fault(
    train_df: pd.DataFrame,
    fault_type: str,
    seed: int,
) -> tuple[pd.DataFrame, Any]:
    """Inject a named fault and return `(modified_train_df, model_override_or_None)`."""
    rng = np.random.RandomState(seed)
    df = train_df.copy()
    model_override = None

    if fault_type == "feature_leakage":
        df["leaked"] = df["default"] * 0.9 + rng.normal(0, 0.1, len(df))

    elif fault_type == "label_shift":
        mask = df["region"] == "North"
        flip_idx = df[mask].sample(frac=0.4, random_state=seed).index
        df.loc[flip_idx, "default"] = 1 - df.loc[flip_idx, "default"]

    elif fault_type == "degraded_slice":
        north = df[df["region"] == "North"]
        drop_idx = north.sample(frac=0.9, random_state=seed).index
        df = df.drop(drop_idx)

    elif fault_type == "schema_drift":
        df["income"] = df["income"] / 1000

    elif fault_type == "latency_regression":
        model_override = "large_model"

    elif fault_type == "fairness_regression":
        female_non_default = df[(df["gender"] == "female") & (df["default"] == 0)]
        flip_idx = female_non_default.sample(frac=0.3, random_state=seed).index
        df.loc[flip_idx, "default"] = 1

    elif fault_type == "fairness_threshold_shift":
        model_override = "group_threshold_shift"

    elif fault_type == "fairness_proxy_feature":
        is_female = (df["gender"] == "female").astype(float)
        df["proxy_score"] = is_female * 0.85 + rng.normal(0.0, 0.25, len(df))
        high_proxy_neg = df[(df["proxy_score"] > 0.6) & (df["default"] == 0)]
        flip_idx = high_proxy_neg.sample(frac=0.35, random_state=seed).index
        df.loc[flip_idx, "default"] = 1
        model_override = "proxy_blind_model"

    elif fault_type == "noop":
        pass

    return df, model_override
