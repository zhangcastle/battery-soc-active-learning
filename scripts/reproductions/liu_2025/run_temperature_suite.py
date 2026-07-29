#!/usr/bin/env python3
"""Run Liu 2025 across every public Panasonic target temperature."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

TARGETS = {
    "10c": ("10 °C", "panasonic_25c_to_10c.csv"),
    "0c": ("0 °C", "panasonic_25c_to_0c.csv"),
    "n10c": ("-10 °C", "panasonic_25c_to_n10c.csv"),
    "n20c": ("-20 °C", "panasonic_25c_to_n20c.csv"),
    "s_n20c": ("S_-20 °C", "panasonic_25c_to_s_n20c.csv"),
    "s_10c": ("S_10 °C", "panasonic_25c_to_s_10c.csv"),
}
PROFILES = ("LA92", "UDDS", "US06", "HWFET", "NN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--transfer-epochs", type=int, default=100)
    parser.add_argument("--grl-strength", type=float, default=0.1)
    parser.add_argument("--targets", nargs="+", choices=tuple(TARGETS), default=list(TARGETS))
    parser.add_argument(
        "--reuse-result",
        action="append",
        default=[],
        metavar="TARGET=RESULTS_JSON",
    )
    return parser.parse_args()


def summarize(results: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for target, result in results.items():
        methods = [
            ("Source estimator", "target_source_only"),
            ("Fine-tuning", "target_fine_tuning"),
            ("DARM", "target_darm"),
        ]
        for method, key in methods:
            if not result.get(key):
                continue
            metrics = result[key]
            row = {"target_temperature": TARGETS[target][0], "method": method}
            for metric in ("rmse_percent", "mae_percent"):
                values = []
                for profile in PROFILES:
                    value = metrics.get(profile, {}).get(metric)
                    row[f"{metric}_{profile}"] = value
                    if value is not None:
                        values.append(value)
                row[f"average_{metric}"] = sum(values) / len(values)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[3]
    manifests = repo / "data/manifests/liu_2025"
    runner = repo / "scripts/reproductions/liu_2025/run_reproduction.py"
    reused = {}
    for item in args.reuse_result:
        target, separator, path = item.partition("=")
        if not separator or target not in TARGETS:
            raise ValueError(f"Invalid --reuse-result: {item}")
        reused[target] = Path(path)

    results = {}
    for target in args.targets:
        output = args.output_root / target
        result_path = reused.get(target, output / "results.json")
        if target in reused and not result_path.is_file():
            raise FileNotFoundError(f"Reused result not found: {result_path}")
        if not result_path.is_file():
            command = [
                sys.executable,
                str(runner),
                "--manifest",
                str(manifests / TARGETS[target][1]),
                "--data-root",
                str(args.data_root),
                "--output",
                str(output),
                "--source-artifact",
                str(args.source_artifact),
                "--feature-order",
                "paper_ivt",
                "--source-normalization",
                "source_domain",
                "--include-fine-tuning",
                "--transfer-epochs",
                str(args.transfer_epochs),
                "--transfer-lr",
                "0.0001",
                "--batch-size",
                "256",
                "--patience",
                "20",
                "--alpha",
                "10",
                "--beta",
                "1",
                "--adversarial-weight",
                "1",
                "--adversarial-update",
                "grl_joint",
                "--grl-strength",
                str(args.grl_strength),
                "--target-bn-mode",
                "frozen_stats",
                "--seed",
                "42",
                "--device",
                args.device,
                "--verify-hashes",
                "--resume",
            ]
            subprocess.run(command, check=True)
        results[target] = json.loads(result_path.read_text(encoding="utf-8"))

    table = summarize(results)
    args.output_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_root / "table_v_reproduction.csv", index=False)
    (args.output_root / "suite_metadata.json").write_text(
        json.dumps(
            {
                "targets": args.targets,
                "source_artifact": str(args.source_artifact),
                "transfer_epochs": args.transfer_epochs,
                "grl_strength": args.grl_strength,
                "paper_declared_parameters": {
                    "source_epochs": 500,
                    "transfer_epochs": 100,
                    "source_learning_rate": 0.001,
                    "transfer_learning_rate": 0.0001,
                    "batch_size": 256,
                    "alpha": 10,
                    "beta": 1,
                    "patience": 20,
                },
                "reconstruction_assumptions": {
                    "target_bn_mode": "frozen_stats",
                    "adversarial_update": "grl_joint",
                    "grl_strength": args.grl_strength,
                    "s_10_missing": ["US06", "NN"],
                },
                "methods": {
                    "source_estimator": "zero-shot target evaluation",
                    "fine_tuning": "supervised target Cycle 1",
                    "darm": "unlabelled target Cycle 1",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    print(f"\nSaved: {args.output_root / 'table_v_reproduction.csv'}")


if __name__ == "__main__":
    main()
