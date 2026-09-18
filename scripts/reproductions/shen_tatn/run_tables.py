#!/usr/bin/env python3
"""Run the Shen et al. (2023) TATN Table II/IV/V/VI/VII reproduction."""

from __future__ import annotations

import argparse
from pathlib import Path

from battery_benchmark.reproductions.shen_2023.data import PROFILES
from battery_benchmark.reproductions.shen_tatn.suite import ShenTableSuite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--transfer-epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    args = parser.parse_args()

    suite = ShenTableSuite(
        args.raw_root,
        args.output_dir,
        epochs=args.epochs,
        transfer_epochs=args.transfer_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
        min_epochs=args.min_epochs,
    )
    profiles = tuple(args.profiles)
    suite.run_all_transfers(profiles=profiles)
    for name, table in suite.tables(profiles=profiles).items():
        print(f"\nTable {name}\n{table.round(3).to_string()}")


if __name__ == "__main__":
    main()
