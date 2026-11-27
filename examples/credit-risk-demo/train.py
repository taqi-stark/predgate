"""Demo project: synthetic credit dataset generator and model trainer.

Generates a synthetic credit risk dataset with meaningful slicing columns
(region, age_bucket, gender) for demonstrating PredGate checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


def generate_dataset(
    n_samples: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic credit risk dataset.

    Features:
        - income: Annual income (numeric)
        - age: Age in years (numeric)
        - credit_score: Credit score 300-850 (numeric)
        - debt_ratio: Debt-to-income ratio (numeric)
        - employment_years: Years of employment (numeric)
        - region: Geographic region (categorical: North, South, East, West)
        - age_bucket: Age group (categorical: young, middle, senior)
        - gender: Gender (categorical: male, female)
        - applicant_id: Unique ID (should be irrelevant to prediction)
        - default: Target variable (0 = no default, 1 = default)

    Args:
        n_samples: Number of samples to generate.
        seed: Random seed.

    Returns:
        DataFrame with all features and target.
    """
    rng = np.random.RandomState(seed)

    income = rng.lognormal(mean=10.5, sigma=0.5, size=n_samples)
    age = rng.randint(18, 70, size=n_samples)
    credit_score = rng.normal(680, 80, size=n_samples).clip(300, 850).astype(int)
    debt_ratio = rng.beta(2, 5, size=n_samples)
    employment_years = rng.exponential(5, size=n_samples).clip(0, 40).astype(int)

    regions = rng.choice(
        ["North", "South", "East", "West"], size=n_samples, p=[0.25, 0.30, 0.25, 0.20]
    )
    genders = rng.choice(["male", "female"], size=n_samples, p=[0.55, 0.45])
    applicant_ids = np.arange(1, n_samples + 1)

    # Age buckets
    age_bucket = np.where(age < 30, "young", np.where(age < 50, "middle", "senior"))

    # Generate target with realistic correlations
    logit = (
        -2.0
        + 0.3 * (debt_ratio - 0.3)
        - 0.3 * (income - 30000) / 10000
        - 0.005 * (credit_score - 600)
        + 0.02 * (40 - employment_years) / 10
        + rng.normal(0, 0.3, size=n_samples)
    )

    # Add slight regional effect
    region_effect = np.where(regions == "North", 0.1, 0)
    logit += region_effect

    prob = 1 / (1 + np.exp(-logit))
    default = (rng.random(n_samples) < prob).astype(int)

    df = pd.DataFrame(
        {
            "applicant_id": applicant_ids,
            "income": np.round(income, 2),
            "age": age,
            "credit_score": credit_score,
            "debt_ratio": np.round(debt_ratio, 4),
            "employment_years": employment_years,
            "region": regions,
            "age_bucket": age_bucket,
            "gender": genders,
            "default": default,
        }
    )

    return df


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42,
) -> Pipeline:
    """Build and train an sklearn Pipeline with preprocessing and gradient boosting classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        seed: Random seed.

    Returns:
        Trained Pipeline.
    """
    cat_cols = [c for c in ["region", "age_bucket", "gender"] if c in X_train.columns]
    num_cols = [
        c
        for c in ["income", "age", "credit_score", "debt_ratio", "employment_years"]
        if c in X_train.columns
    ]

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
                GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


if __name__ == "__main__":
    from pathlib import Path

    # Generate and save datasets
    print("Generating synthetic credit dataset...")

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Full dataset
    full_df = generate_dataset(n_samples=5000, seed=42)

    # Split into train/test/reference
    train_df = full_df.sample(frac=0.6, random_state=42)
    remaining = full_df.drop(train_df.index)
    test_df = remaining.sample(frac=0.5, random_state=42)
    reference_df = remaining.drop(test_df.index)

    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)
    reference_df.to_csv(data_dir / "reference.csv", index=False)

    print(f"  Train: {len(train_df)} rows")
    print(f"  Test: {len(test_df)} rows")
    print(f"  Reference: {len(reference_df)} rows")
    print(f"  Default rate: {full_df['default'].mean():.1%}")
    print(f"  Saved to {data_dir}/")
