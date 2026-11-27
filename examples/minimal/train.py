"""Minimal self-contained PredGate example using the public Diabetes dataset (built into scikit-learn)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_public_dataset() -> pd.DataFrame:
    """Load scikit-learn's built-in public Diabetes dataset as a binary classification task."""
    bunch = load_diabetes(as_frame=True)
    df = bunch.frame.copy()
    median_y = float(df["target"].median())
    df["high_risk"] = (df["target"] >= median_y).astype(int)
    df = df.drop(columns=["target"])
    df["patient_id"] = np.arange(1, len(df) + 1)
    df["sex_group"] = np.where(df["sex"] > 0, "group_a", "group_b")
    df["age_group"] = np.where(df["age"] >= 0, "older", "younger")
    return df


def build_model(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> Pipeline:
    """Train a GradientBoostingClassifier pipeline (ignores patient_id)."""
    cat_cols = [c for c in ["sex_group", "age_group"] if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols and c != "patient_id"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=40,
                    max_depth=3,
                    learning_rate=0.08,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


if __name__ == "__main__":
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    full_df = load_public_dataset().sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_train = int(len(full_df) * 0.65)

    train_df = full_df.iloc[:n_train]
    test_df = full_df.iloc[n_train:]

    train_df.to_csv(data_dir / "train.csv", index=False)
    # reference.csv is the baseline training snapshot the champion was trained on
    train_df.to_csv(data_dir / "reference.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)
    print(
        f"Saved train={len(train_df)}, reference={len(train_df)}, test={len(test_df)} -> {data_dir}/"
    )
