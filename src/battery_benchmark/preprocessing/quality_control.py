"""Data quality checks for battery cycle DataFrames."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from battery_benchmark.data.schema import COLUMNS


def check_cycle_quality(
    df: pd.DataFrame,
    min_samples: int = 10,
    max_nan_fraction: float = 0.05,
    voltage_range: tuple[float, float] = (2.0, 4.5),
) -> dict[str, Any]:
    """Run quality checks on a single cycle DataFrame.

    Returns
    -------
    dict
        ``{"pass": bool, "issues": list[str]}``
    """
    issues: list[str] = []

    if len(df) < min_samples:
        issues.append(f"Too few samples: {len(df)} < {min_samples}")

    for col in [COLUMNS.VOLTAGE, COLUMNS.CURRENT, COLUMNS.TEMPERATURE]:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")
            continue
        nan_frac = df[col].isna().mean()
        if nan_frac > max_nan_fraction:
            issues.append(f"High NaN fraction in '{col}': {nan_frac:.2%}")

    if COLUMNS.VOLTAGE in df.columns:
        v = df[COLUMNS.VOLTAGE].dropna()
        if len(v) > 0:
            if v.min() < voltage_range[0]:
                issues.append(f"Voltage below physical minimum: {v.min():.3f} V")
            if v.max() > voltage_range[1]:
                issues.append(f"Voltage above physical maximum: {v.max():.3f} V")

    return {"pass": len(issues) == 0, "issues": issues}


def filter_passing_cycles(
    cycles: list[pd.DataFrame], **qc_kwargs: Any
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    """Return only cycles passing quality checks, with per-cycle QC results."""
    passing, reports = [], []
    for df in cycles:
        report = check_cycle_quality(df, **qc_kwargs)
        reports.append(report)
        if report["pass"]:
            passing.append(df)
    return passing, reports
