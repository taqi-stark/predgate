"""Script to create a deliberately bad model branch for testing PredGate.

Injects various faults to demonstrate that PredGate catches them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_bad_branch_data(fault_type: str = "degraded_slice", seed: int = 42) -> None:
    """Create deliberately faulty training data.

    Args:
        fault_type: Type of fault to inject. One of:
            - feature_leakage: Add a column derived from target
            - label_shift: Corrupt labels in one subgroup
            - degraded_slice: Remove most of one subgroup
            - schema_drift: Change a column's scale
            - fairness_regression: Bias against one group
        seed: Random seed.
    """
    data_dir = Path("data")
    train_df = pd.read_csv(data_dir / "train.csv")
    rng = np.random.RandomState(seed)

    if fault_type == "feature_leakage":
        # Add a column derived from the target
        train_df["leaked_feature"] = train_df["default"] * 0.9 + rng.normal(0, 0.1, len(train_df))
        print("Injected feature leakage: added 'leaked_feature' correlated with target")

    elif fault_type == "label_shift":
        # Corrupt labels in the North region
        mask = train_df["region"] == "North"
        flip_idx = train_df[mask].sample(frac=0.4, random_state=seed).index
        train_df.loc[flip_idx, "default"] = 1 - train_df.loc[flip_idx, "default"]
        print(f"Injected label shift: flipped {len(flip_idx)} labels in region=North")

    elif fault_type == "degraded_slice":
        # Remove 90% of rows from region=North
        north_mask = train_df["region"] == "North"
        north_rows = train_df[north_mask]
        drop_idx = north_rows.sample(frac=0.9, random_state=seed).index
        train_df = train_df.drop(drop_idx)
        print(f"Injected degraded slice: removed {len(drop_idx)} rows from region=North")

    elif fault_type == "schema_drift":
        # Change income scale (divide by 1000)
        train_df["income"] = train_df["income"] / 1000
        print("Injected schema drift: scaled income by 1/1000")

    elif fault_type == "fairness_regression":
        # Bias: increase default rate for females
        female_mask = train_df["gender"] == "female"
        non_default_females = train_df[female_mask & (train_df["default"] == 0)]
        flip_idx = non_default_females.sample(frac=0.3, random_state=seed).index
        train_df.loc[flip_idx, "default"] = 1
        print(f"Injected fairness regression: biased {len(flip_idx)} female records")

    else:
        print(f"Unknown fault type: {fault_type}")
        return

    # Save the bad data
    bad_dir = data_dir / "bad"
    bad_dir.mkdir(exist_ok=True)
    train_df.to_csv(bad_dir / "train.csv", index=False)
    print(f"Saved faulty data to {bad_dir}/train.csv")


if __name__ == "__main__":
    fault = sys.argv[1] if len(sys.argv) > 1 else "degraded_slice"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    create_bad_branch_data(fault, seed)
