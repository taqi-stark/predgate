"""Pandera schema for the credit risk demo dataset."""

from __future__ import annotations

try:
    from pandera.pandas import Check, Column, DataFrameSchema
except ImportError:
    from pandera import Check, Column, DataFrameSchema  # type: ignore[no-redef]

schema = DataFrameSchema(
    {
        "applicant_id": Column(int, Check.greater_than(0), nullable=False),
        "income": Column(float, Check.greater_than(0), nullable=False),
        "age": Column(int, Check.in_range(18, 100), nullable=False),
        "credit_score": Column(int, Check.in_range(300, 850), nullable=False),
        "debt_ratio": Column(float, Check.in_range(0, 1), nullable=False),
        "employment_years": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "region": Column(str, Check.isin(["North", "South", "East", "West"]), nullable=False),
        "age_bucket": Column(str, Check.isin(["young", "middle", "senior"]), nullable=False),
        "gender": Column(str, Check.isin(["male", "female"]), nullable=False),
        "default": Column(int, Check.isin([0, 1]), nullable=False),
    },
    strict=False,  # Allow extra columns
    coerce=True,
)
