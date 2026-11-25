"""Pandera data contract schema for the minimal Diabetes example."""

from __future__ import annotations

try:
    from pandera.pandas import Check, Column, DataFrameSchema
except ImportError:
    from pandera import Check, Column, DataFrameSchema  # type: ignore[no-redef]

schema = DataFrameSchema(
    {
        "patient_id": Column(int, Check.greater_than(0), nullable=False),
        "age": Column(float, nullable=False),
        "bmi": Column(float, nullable=False),
        "bp": Column(float, nullable=False),
        "sex_group": Column(str, Check.isin(["group_a", "group_b"]), nullable=False),
        "age_group": Column(str, Check.isin(["younger", "older"]), nullable=False),
        "high_risk": Column(int, Check.isin([0, 1]), nullable=False),
    },
    strict=False,
    coerce=True,
)
