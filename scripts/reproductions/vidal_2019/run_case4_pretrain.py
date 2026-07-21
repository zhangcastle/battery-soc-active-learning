#!/usr/bin/env python3
"""CLI for Vidal Case 4 Stage A multi-source partial pretraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.reproductions.vidal_2019.case4 import Case4PretrainConfig, run_case4_pretrain  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-windows", type=int)
    args = parser.parse_args()
    values = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    values["run_mode"] = args.run_mode
    if args.seed is not None:
        values["random_seed"] = args.seed
    if args.max_train_windows is not None:
        values["max_train_windows"] = args.max_train_windows
    cfg = Case4PretrainConfig(**values)
    result = run_case4_pretrain(cfg, args.input_root, args.output_root, REPO_ROOT, args.device)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
