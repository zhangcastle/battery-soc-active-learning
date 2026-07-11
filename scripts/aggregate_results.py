#!/usr/bin/env python
"""Aggregate all experiment results into a comparison table.

Usage
-----
python scripts/aggregate_results.py --results-dir outputs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from battery_benchmark.experiments.comparison import compare_experiments, print_comparison_table
from battery_benchmark.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--results-dir", default="outputs")
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    df = compare_experiments(args.results_dir)
    print_comparison_table(df)
    if args.output_csv:
        df.to_csv(args.output_csv)
        print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()
