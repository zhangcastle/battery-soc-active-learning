#!/usr/bin/env python
"""Prepare (preprocess and standardise) a raw dataset.

Usage
-----
python scripts/prepare_dataset.py \\
    --dataset panasonic_18650pf \\
    --paths   configs/paths/local.yaml

Runs the dataset adapter's discover_files / build_manifest / load_cycle /
standardize_columns pipeline and saves processed Parquet files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from battery_benchmark.utils.config import load_yaml
from battery_benchmark.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a battery dataset")
    parser.add_argument("--dataset", required=True,
                        choices=["panasonic_18650pf", "lg_hg2", "samsung_48g", "samsung_30t", "cmu_evtol"])
    parser.add_argument("--paths", default="configs/paths/local.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    paths = load_yaml(args.paths)
    raw_dir = Path(paths.get("raw_data_root", "data/raw")) / args.dataset
    print(f"Dataset:     {args.dataset}")
    print(f"Raw data:    {raw_dir}")
    print("Note: Dataset adapters are placeholders. See notebooks/0{2..5}_prepare_*.ipynb.")


if __name__ == "__main__":
    main()
