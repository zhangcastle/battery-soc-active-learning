#!/usr/bin/env python3
"""CLI for Vidal Case 1 smoke/formal runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.experiments.vidal_case1 import (  # noqa: E402
    DATASETS, Case1Config, cache_case1_inputs, run_case1, write_case1_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--run-mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.resume and args.dataset == "all":
        raise ValueError("--resume requires one explicit dataset")
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    results = []
    for dataset in datasets:
        values = {**raw, "dataset": dataset, "run_mode": args.run_mode}
        if args.seed is not None: values["random_seed"] = args.seed
        if args.max_train_windows is not None: values["max_train_windows"] = args.max_train_windows
        cfg = Case1Config(**values)
        input_root = args.input_root
        if args.cache_root:
            cache_case1_inputs(args.input_root, args.cache_root, dataset)
            input_root = args.cache_root
        results.append(run_case1(cfg, input_root, args.output_root, REPO_ROOT, args.device,
                                 resume_path=args.resume))
    if args.run_mode == "formal" and args.dataset == "all":
        write_case1_summary(results, args.output_root)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
