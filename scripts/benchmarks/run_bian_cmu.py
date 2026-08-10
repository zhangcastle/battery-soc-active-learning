#!/usr/bin/env python3
"""Run paper-aligned supervised Bian DTNN adaptation on CMU development SOC."""

# ruff: noqa: E402 -- executable from a source checkout without installation

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from battery_benchmark.datasets.cmu_evtol.vidal_adapter import load_cmu_dynamic_soc_for_vidal
from battery_benchmark.evaluation.metrics import compute_metrics
from battery_benchmark.reproductions.bian_2021 import DTNN, adaptive_multiscale_mmd
from battery_benchmark.reproductions.liu_2025.cmu_protocol import select_duration_matched_missions
from battery_benchmark.reproductions.shen_2023.data import (
    TEMPERATURES,
    load_temperature_cycles_and_range,
)
from battery_benchmark.utils.device import get_device
from battery_benchmark.utils.hashing import sha256_file

CELLS = ("VAH01", "VAH17", "VAH27", "VAH30", "VAH15", "VAH12", "VAH11")
BASELINE_CELLS = {"VAH01", "VAH17", "VAH27"}
SOURCE_PROFILES = ("Cycle_1", "Cycle_2", "Cycle_3", "Cycle_4", "NN", "US06", "HWFET")
FEATURES = ("current", "voltage")


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _windows(parts: list[tuple[np.ndarray, np.ndarray, str]], length: int):
    xs, ys, missions = [], [], []
    for x, y, mission in parts:
        usable = min(len(x), len(y)) // length * length
        if usable:
            count = usable // length
            xs.append(x[:usable].reshape(count, length, x.shape[-1]))
            ys.append(y[:usable].reshape(count, length))
            missions.append((mission, count))
    if not xs:
        raise ValueError("No complete non-overlapping windows")
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32), missions


def _source_data(root: Path, length: int):
    raw_parts = []
    for temperature in TEMPERATURES:
        cycles, feature_min, feature_max = load_temperature_cycles_and_range(root, temperature)
        scale = np.where(feature_max == feature_min, 1.0, feature_max - feature_min)
        raw_parts.extend(
            (
                cycles[name].features[:, :2] * scale[:2] + feature_min[:2],
                cycles[name].soc,
                f"{temperature}C/{name}",
            )
            for name in SOURCE_PROFILES
        )
    all_features = np.concatenate([part[0] for part in raw_parts])
    feature_min, feature_max = all_features.min(0), all_features.max(0)
    scale = np.where(feature_max == feature_min, 1.0, feature_max - feature_min)
    parts = [((x - feature_min) / scale, y, name) for x, y, name in raw_parts]
    x, y, _ = _windows(parts, length)
    return x, y, feature_min, feature_max


def _target_parts(records, mission_ids, initial_soc, fraction, portion):
    if portion not in {"adapt", "test"}:
        raise ValueError("portion must be adapt or test")
    parts = []
    for record in records:
        mission = str(record.metadata["mission_cycle_id"])
        if mission not in mission_ids:
            continue
        frame = record.data
        y = frame["soc"].to_numpy(dtype=np.float64)
        starts = np.flatnonzero(y <= initial_soc)
        if not len(starts):
            continue
        raw = frame.loc[:, FEATURES].to_numpy(dtype=np.float64)[starts[0] :]
        y = y[starts[0] :]
        boundary = int(len(y) * fraction)
        selection = slice(0, boundary) if portion == "adapt" else slice(boundary, None)
        parts.append((raw[selection], y[selection], mission))
    return parts


def _target_data(parts, length, feature_min, feature_max):
    scale = np.where(feature_max == feature_min, 1.0, feature_max - feature_min)
    parts = [((x - feature_min) / scale, y, mission) for x, y, mission in parts]
    return _windows(parts, length)


def _loader(x, y, batch_size, shuffle):
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=min(batch_size, len(x)),
        shuffle=shuffle,
    )


def _train_source(model, x, y, epochs, batch_size, lr, device):
    loader = _loader(x, y, batch_size, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(xb)
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % max(epochs // 10, 1) == 0:
            print(f"  source epoch {epoch + 1}/{epochs}: mse={total / len(x):.6f}")


def _adapt(model, source_x, source_y, target_x, target_y, args, device):
    source_loader = _loader(source_x, source_y, args.transfer_batch_size, True)
    target_loader = _loader(target_x, target_y, args.transfer_batch_size, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.transfer_lr)
    criterion = nn.MSELoss()
    final = {}
    for _ in range(args.transfer_epochs):
        model.train()
        source_batches = cycle(source_loader)
        for target_batch in target_loader:
            source_batch = next(source_batches)
            sx, sy = (item.to(device) for item in source_batch)
            tx, ty = (item.to(device) for item in target_batch)
            optimizer.zero_grad(set_to_none=True)
            sp, sf = model(sx, return_features=True)
            tp, tf = model(tx, return_features=True)
            source_loss = criterion(sp, sy)
            target_loss = criterion(tp, ty)
            mmd, layer_mmd, weights = adaptive_multiscale_mmd(sf, tf)
            loss = source_loss + args.target_weight * target_loss + args.mmd_weight * mmd
            loss.backward()
            optimizer.step()
            final = {
                "source_mse": float(source_loss.detach()),
                "target_mse": float(target_loss.detach()),
                "adaptive_mmd": float(mmd.detach()),
                "layer_mmd": [float(v) for v in layer_mmd.detach().cpu()],
                "layer_weights": [float(v) for v in weights.cpu()],
            }
    final["optimizer_steps"] = args.transfer_epochs * len(target_loader)
    return final


@torch.no_grad()
def _evaluate(model, x, y, missions, batch_size, device):
    model.eval()
    prediction = torch.cat(
        [model(xb.to(device)).cpu() for xb, _ in _loader(x, y, batch_size, False)]
    ).numpy()
    prediction = np.clip(prediction, 0.0, 1.0)
    metrics = compute_metrics(y, prediction)
    overall = {
        "overall_mae_percent": metrics["mae"] * 100,
        "overall_rmse_percent": metrics["rmse"] * 100,
        "overall_max_error_percent": metrics["max_error"] * 100,
        "overall_bias_percent": metrics["bias"] * 100,
        "overall_r2": metrics["r2"],
    }
    start, per_mission = 0, []
    for _, count in missions:
        stop = start + count
        per_mission.append(compute_metrics(y[start:stop], prediction[start:stop]))
        start = stop
    overall.update(
        {
            "mission_macro_rmse_percent": float(np.mean([m["rmse"] for m in per_mission]) * 100),
            "mission_macro_mae_percent": float(np.mean([m["mae"] for m in per_mission]) * 100),
            "test_missions": len(missions),
            "test_windows": len(x),
        }
    )
    return overall


def _cell_path(cell, base_root, condition_root):
    root = base_root if cell in BASELINE_CELLS else condition_root
    return root / f"{cell}_dynamic_soc.parquet"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panasonic-data-root", type=Path, default=REPO_ROOT / "data/raw/Panasonic_18650PF"
    )
    parser.add_argument(
        "--cmu-base-root",
        type=Path,
        default=REPO_ROOT / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1",
    )
    parser.add_argument(
        "--cmu-condition-root",
        type=Path,
        default=REPO_ROOT
        / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1_extension4",
    )
    parser.add_argument(
        "--cmu-manifest",
        type=Path,
        default=REPO_ROOT
        / "data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1.csv",
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "outputs/benchmarks/bian_2021_cmu"
    )
    parser.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS))
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--adapt-fraction", type=float, default=0.30)
    parser.add_argument("--initial-soc", type=float, default=0.70)
    parser.add_argument("--source-epochs", type=int, default=2000)
    parser.add_argument("--transfer-epochs", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--transfer-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--transfer-lr", type=float, default=1e-4)
    parser.add_argument("--target-weight", type=float, default=5.0)
    parser.add_argument("--mmd-weight", type=float, default=0.1)
    parser.add_argument("--budget-seconds", type=float, default=8778.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lightweight", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        not 0 < args.adapt_fraction < 1
        or not 0 < args.initial_soc <= 1
        or min(args.source_epochs, args.transfer_epochs, args.repeats) <= 0
    ):
        raise ValueError("fractions/SOC and epoch/repeat counts are invalid")
    device = get_device(args.device)
    source_x, source_y, source_feature_min, source_feature_max = _source_data(
        args.panasonic_data_root, args.sequence_length
    )
    manifest = pd.read_csv(args.cmu_manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows, runs = [], []
    widths = (64, 32, 64) if args.lightweight else (256, 64, 256)

    for repeat in range(args.repeats):
        seed = args.seed + repeat
        _seed(seed)
        print(f"repeat {repeat + 1}/{args.repeats}: pretraining on {len(source_x)} source windows")
        source_model = DTNN(2, *widths).to(device)
        _train_source(
            source_model, source_x, source_y, args.source_epochs, args.batch_size, args.lr, device
        )
        source_state = copy.deepcopy(source_model.state_dict())

        for cell in args.cells:
            path = _cell_path(cell, args.cmu_base_root, args.cmu_condition_root)
            expected = set(
                manifest.loc[manifest["cell_id"] == cell, "source_artifact_sha256"].astype(str)
            )
            actual = sha256_file(path)
            if expected != {actual}:
                raise ValueError(f"{cell}: CMU input hash does not match manifest")
            adapt_ids, validation_ids, test_ids, budget = select_duration_matched_missions(
                manifest, cell, budget_seconds=args.budget_seconds
            )
            records = load_cmu_dynamic_soc_for_vidal([path], target_hz=1.0).records
            adapt_parts = _target_parts(
                records, adapt_ids, args.initial_soc, args.adapt_fraction, "adapt"
            )
            adapt_x, adapt_y, adapt_missions = _target_data(
                adapt_parts, args.sequence_length, source_feature_min, source_feature_max
            )
            test_parts = _target_parts(
                records, test_ids, args.initial_soc, args.adapt_fraction, "test"
            )
            test_x, test_y, test_missions = _target_data(
                test_parts, args.sequence_length, source_feature_min, source_feature_max
            )
            model = DTNN(2, *widths).to(device)
            model.load_state_dict(source_state)
            source_only = _evaluate(model, test_x, test_y, test_missions, args.batch_size, device)
            training = _adapt(model, source_x, source_y, adapt_x, adapt_y, args, device)
            adapted = _evaluate(model, test_x, test_y, test_missions, args.batch_size, device)
            for method, metrics in (("Source only", source_only), ("Bian MDA", adapted)):
                rows.append(
                    {
                        "repeat": repeat + 1,
                        "seed": seed,
                        "cell_id": cell,
                        "initial_soc_percent": int(round(args.initial_soc * 100)),
                        "method": method,
                        **metrics,
                    }
                )
            runs.append(
                {
                    "repeat": repeat + 1,
                    "seed": seed,
                    "cell_id": cell,
                    "initial_soc_percent": int(round(args.initial_soc * 100)),
                    "cmu_input_sha256": actual,
                    "adaptation_budget": budget,
                    "adaptation_fraction_within_mission": args.adapt_fraction,
                    "adaptation_missions": len(adapt_missions),
                    "adaptation_windows": len(adapt_x),
                    "validation_missions_unused": len(validation_ids),
                    "training_final_batch": training,
                }
            )
            print(
                f"  {cell}: source={source_only['overall_rmse_percent']:.3f}% "
                f"MDA={adapted['overall_rmse_percent']:.3f}%"
            )

    table = pd.DataFrame(rows)
    table.to_csv(args.output_root / "metrics.csv", index=False)
    summary = (
        table.groupby(["cell_id", "initial_soc_percent", "method"], sort=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    summary.to_csv(args.output_root / "summary.csv", index=False)
    metadata = {
        "benchmark": "bian_2021_panasonic_multitemperature_to_cmu_supervised_development",
        "architecture": "CNN1->CNN2->two-layer-BiLSTM->FC->sequence-regression",
        "mmd_layers": ["CNN2", "BiLSTM2", "FC"],
        "objective": "source_mse + 5*target_mse + 0.1*adaptive_multiscale_mmd",
        "source_profiles": list(SOURCE_PROFILES),
        "source_temperatures_c": list(TEMPERATURES),
        "feature_order": list(FEATURES),
        "normalization": "source-global min-max applied unchanged to target",
        "target_protocol": "start at initial SOC; first 30% adapt; remaining 70% test",
        "target_soc_labels_used_for_adaptation": True,
        "target_soc_labels_used_for_checkpoint_selection": False,
        "target_soc_reference_used_for_posthoc_evaluation": True,
        "label_is_ground_truth": False,
        "formal_benchmark_eligible": False,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "runs": runs,
    }
    (args.output_root / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
