#!/usr/bin/env python3
"""CLI for Vidal Case 4 Stage B target full-train fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.experiments.vidal_case4 import Case4FinetuneConfig, run_case4_finetune  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("panasonic_18650pf", "lg_hg2", "samsung_30t"), default="samsung_30t")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--case1-output-root", type=Path)
    parser.add_argument("--target-case1-run", type=Path)
    parser.add_argument("--run-mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-windows", type=int)
    args = parser.parse_args()
    values = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    values.update({"target_dataset": args.target, "run_mode": args.run_mode, "source_checkpoint": args.source_checkpoint})
    if args.seed is not None:
        values["random_seed"] = args.seed
    if args.max_train_windows is not None:
        values["max_train_windows"] = args.max_train_windows
    cfg = Case4FinetuneConfig(**values)
    result = run_case4_finetune(
        cfg,
        args.input_root,
        args.output_root,
        REPO_ROOT,
        args.device,
        case1_output_root=args.case1_output_root,
        target_case1_run=args.target_case1_run,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
