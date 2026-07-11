#!/usr/bin/env python
"""Command-line training entry point.

Usage
-----
python scripts/train.py \\
    --config configs/experiments/smoke_test.yaml \\
    --paths  configs/paths/local.yaml

To resume after a Colab disconnection:

python scripts/train.py \\
    --config configs/experiments/smoke_test.yaml \\
    --paths  configs/paths/local.yaml \\
    --resume outputs/smoke_test/checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the Python path when running directly without pip install
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from battery_benchmark.experiments.runner import ExperimentRunner
from battery_benchmark.utils.config import load_config
from battery_benchmark.utils.logging import setup_logging
from battery_benchmark.utils.paths import find_repo_root, resolve_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Battery SOC benchmark training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--paths", default=None,
        help="Path to paths YAML config (optional)",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger = setup_logging(level=args.log_level)
    logger.info("Battery SOC Benchmark — training script")
    logger.info("Config: %s", args.config)
    logger.info("Paths:  %s", args.paths or "(none)")

    # Load config
    cfg = load_config(args.config, paths_path=args.paths)

    # Override resume path from CLI
    if args.resume:
        cfg["checkpoint_path"] = args.resume
        cfg["resume"] = True

    # Resolve output directory
    repo_root = find_repo_root(_REPO_ROOT)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = resolve_output_path(cfg, repo_root=repo_root)

    logger.info("Output directory: %s", output_dir)

    # Run experiment
    runner = ExperimentRunner(cfg, output_dir=output_dir, repo_root=repo_root)
    result = runner.run()

    # Print summary
    print("\n" + "=" * 60)
    print(f"Experiment: {result.experiment_name}")
    print(f"Source → Target: {result.source_domain} → {result.target_domain}")
    print(f"Transfer method: {result.transfer_method}")
    print("-" * 60)
    print(f"Source test  MAE:       {result.test_mae:.4f}")
    print(f"Source test  RMSE:      {result.test_rmse:.4f}")
    print(f"Source test  MAX error: {result.test_max_error:.4f}")
    print(f"Outputs saved to:       {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
