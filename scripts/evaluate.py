#!/usr/bin/env python
"""Evaluate a saved checkpoint on a specified dataset split.

Usage
-----
python scripts/evaluate.py \\
    --checkpoint outputs/smoke_test/checkpoints/best.pt \\
    --config     configs/experiments/smoke_test.yaml \\
    --paths      configs/paths/local.yaml \\
    --split      test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from battery_benchmark.utils.config import load_config
from battery_benchmark.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None, help="Output JSON file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config, paths_path=args.paths)
    print(f"Evaluating checkpoint: {args.checkpoint}")
    print("Note: Full evaluation script is a scaffold. Use ExperimentRunner for complete evaluation.")


if __name__ == "__main__":
    main()
