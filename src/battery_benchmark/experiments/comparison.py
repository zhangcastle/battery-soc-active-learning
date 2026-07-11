"""Cross-experiment comparison utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from battery_benchmark.evaluation.aggregation import load_metrics_files, summarise_experiments


def compare_experiments(results_dir: str | Path) -> pd.DataFrame:
    """Load all experiment results and return a comparison DataFrame."""
    df = load_metrics_files(results_dir)
    return summarise_experiments(df)


def print_comparison_table(df: pd.DataFrame) -> None:
    """Print a formatted comparison table to stdout."""
    if df.empty:
        print("No experiment results found.")
        return
    print(df.to_string())
