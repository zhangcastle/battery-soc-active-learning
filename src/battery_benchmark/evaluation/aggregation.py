"""Cross-experiment result aggregation utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_metrics_files(results_dir: str | Path) -> pd.DataFrame:
    """Load all metrics.json files under *results_dir* into a DataFrame."""
    import json

    results_dir = Path(results_dir)
    rows = []
    for f in sorted(results_dir.rglob("metrics.json")):
        with f.open() as fp:
            data = json.load(fp)
        data["source_file"] = str(f.relative_to(results_dir))
        rows.append(data)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def summarise_experiments(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy summary table sorted by test MAE."""
    if df.empty:
        return df
    cols = [c for c in ["experiment_name", "test_mae", "test_rmse", "test_max_error"] if c in df.columns]
    if not cols:
        return df
    return df[cols].sort_values("test_mae") if "test_mae" in df.columns else df[cols]
