#!/usr/bin/env python3
"""Run the minimum all-vs-random-vs-Regression-CLUE confirmation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.reproductions.vidal_2019.case2 import Case2Config, run_case2  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--committee-checkpoint", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.1)
    parser.add_argument("--run-mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    common = {
        **base,
        "source_dataset": args.source,
        "target_dataset": args.target,
        "source_checkpoint": args.source_checkpoint,
        "run_mode": args.run_mode,
    }
    specifications = (
        ("all", 1.0, ()),
        ("random", args.budget, ()),
        ("regression_clue", args.budget, (args.committee_checkpoint,)),
    )
    rows = []
    for strategy, budget, committee in specifications:
        cfg = Case2Config(
            **common,
            selection_strategy=strategy,
            target_budget_fraction=budget,
            committee_checkpoints=committee,
        )
        result = run_case2(
            cfg,
            args.input_root,
            args.output_root / strategy,
            REPO_ROOT,
            args.device,
        )
        rows.append({
            "strategy": strategy,
            "budget_fraction": budget,
            "selected_windows": result["metadata"]["actual_train_window_count"],
            "pool_windows": result["metadata"]["selection_pool_window_count"],
            "rmse_percent": result["metrics"]["rmse_percent"],
            "mae_percent": result["metrics"]["mae_percent"],
            "max_error_percent": result["metrics"]["max_error_percent"],
            "run_dir": result["run_dir"],
        })
    summary = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "confirmation_summary.csv"
    summary.to_csv(path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
