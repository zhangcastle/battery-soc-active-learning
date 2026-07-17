#!/usr/bin/env python3
"""CLI for Vidal Case 3 partial-target transfer runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.experiments.vidal_case1 import cache_case1_inputs  # noqa: E402
from battery_benchmark.experiments.vidal_case2 import (  # noqa: E402
    Case2Config, all_case2_pairs, find_target_case1_run, run_case2,
)


def _checkpoint_for(source_checkpoint_root: Path | None, source: str) -> Path | None:
    if source_checkpoint_root is None:
        return None
    return find_target_case1_run(source_checkpoint_root, source) / "checkpoints" / "epoch_515.pt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("panasonic_18650pf", "lg_hg2", "samsung_30t"))
    parser.add_argument("--target", choices=("panasonic_18650pf", "lg_hg2", "samsung_30t"))
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case1-output-root", type=Path)
    parser.add_argument("--case2-output-root", type=Path)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--source-checkpoint-root", type=Path)
    parser.add_argument("--target-case1-run", type=Path)
    parser.add_argument("--target-case2-metrics", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--run-mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-windows", type=int)
    args = parser.parse_args()

    if args.all_pairs:
        pairs = all_case2_pairs()
    else:
        if args.source is None or args.target is None:
            raise ValueError("--source and --target are required unless --all-pairs is set")
        pairs = [(args.source, args.target)]

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    results = []
    for source, target in pairs:
        values = {
            **raw,
            "source_dataset": source,
            "target_dataset": target,
            "run_mode": args.run_mode,
            "case_number": 3,
            "output_case_name": "vidal_case3",
            "target_train_split": "partial_train",
        }
        if args.seed is not None:
            values["random_seed"] = args.seed
        if args.max_train_windows is not None:
            values["max_train_windows"] = args.max_train_windows
        if args.target_case2_metrics is not None:
            values["target_case2_metrics_path"] = args.target_case2_metrics
        checkpoint = args.source_checkpoint or _checkpoint_for(args.source_checkpoint_root, source)
        if checkpoint is not None:
            values["source_checkpoint"] = checkpoint
        cfg = Case2Config(**values)
        input_root = args.input_root
        if args.cache_root:
            cache_case1_inputs(args.input_root, args.cache_root, target, splits=("full_train", "partial_train", "test"))
            input_root = args.cache_root
        results.append(run_case2(
            cfg,
            input_root,
            args.output_root,
            REPO_ROOT,
            args.device,
            case1_output_root=args.case1_output_root,
            target_case1_run=args.target_case1_run,
            case2_output_root=args.case2_output_root,
        ))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
