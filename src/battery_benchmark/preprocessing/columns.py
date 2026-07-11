"""Column standardisation utilities.

Maps dataset-specific column names to benchmark schema names.
"""

from __future__ import annotations

import pandas as pd


def rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename columns according to *mapping* (raw_name → schema_name).

    Columns not present in *mapping* are left unchanged.
    """
    present = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=present)


def drop_unknown_columns(
    df: pd.DataFrame, keep: list[str]
) -> pd.DataFrame:
    """Drop any columns not in *keep*."""
    return df[[c for c in keep if c in df.columns]]


def cast_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cast *cols* to float64, coercing errors to NaN."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
