#!/usr/bin/env python
"""Generic experiment runner (thin wrapper around ExperimentRunner).

Usage
-----
python scripts/run_experiment.py \\
    --config configs/reproductions/vidal_2019/vidal_case1.yaml \\
    --paths  configs/paths/local.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from battery_benchmark.experiments.runner import ExperimentRunner
from battery_benchmark.utils.config import load_config
from battery_benchmark.utils.logging import setup_logging
from battery_benchmark.utils.paths import find_repo_root, resolve_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a benchmark experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config, paths_path=args.paths)
    if args.resume:
        cfg["checkpoint_path"] = args.resume
        cfg["resume"] = True
    repo_root = find_repo_root(_REPO_ROOT)
    output_dir = resolve_output_path(cfg, repo_root=repo_root)
    runner = ExperimentRunner(cfg, output_dir=output_dir, repo_root=repo_root)
    runner.run()


if __name__ == "__main__":
    main()
