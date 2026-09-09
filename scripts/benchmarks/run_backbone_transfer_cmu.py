#!/usr/bin/env python3
"""Run CNN-LSTM, TCN, and Transformer-encoder backbones through a Vidal-style
supervised transfer protocol on the CMU eVTOL core-seven cells.

This script deliberately does NOT import from
``battery_benchmark.reproductions.vidal_2019`` (case1.py / case2.py /
windows.py / resampling.py / sequence_v2.py). Those modules produced the
already-published, formal-status results in
``docs/benchmarks/vidal_panasonic_to_cmu_evtol_results_v0_1.md`` and must not
be touched by this experiment. Instead this script reimplements the same
*protocol* -- Panasonic 25 C, 1 Hz, non-overlapping fixed-length windows,
source-fitted min-max normalisation, nested chronological target-label
budgets, full-parameter fine-tuning restarted from the frozen source
checkpoint for every budget -- using only shared dataset adapters
(``datasets.cmu_evtol``, ``reproductions.shen_2023.data``) and the generic
model registry (``models.cnn_lstm`` / ``models.tcn`` /
``models.transformer_encoder``).

Budgets reuse the nested mission-budget flags already frozen in
``data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1.csv``
(``budget_1pct``, ``budget_5pct``, ``budget_10pct``, ``vidal_partial``,
``vidal_full``), so the resulting budget points line up exactly with the
already-published Vidal LSTM sweep.

CMU ``soc_dynamic`` is a development dynamic-capacity reference, not
measured ground truth. Every run written by this script is
``label_is_ground_truth=false`` and ``formal_benchmark_eligible=false``.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
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
from battery_benchmark.models.registry import ModelRegistry
import battery_benchmark.models.cnn_lstm  # noqa: F401 -- registers "cnn_lstm"
import battery_benchmark.models.tcn  # noqa: F401 -- registers "tcn"
import battery_benchmark.models.transformer_encoder  # noqa: F401 -- registers "transformer_encoder"
from battery_benchmark.reproductions.shen_2023.data import PROFILES, load_temperature_cycles_and_range
from battery_benchmark.utils.device import get_device
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty
from battery_benchmark.utils.hashing import sha256_file

CELLS = ("VAH01", "VAH17", "VAH27", "VAH30", "VAH15", "VAH12", "VAH11")
BASELINE_CELLS = {"VAH01", "VAH17", "VAH27"}
FEATURES = ("current", "voltage", "temperature")
MODEL_NAMES = ("cnn_lstm", "tcn", "transformer_encoder")
BUDGETS = ("zero_shot", "1pct", "5pct", "10pct", "partial", "full")
BUDGET_COLUMNS = {
    "1pct": "budget_1pct",
    "5pct": "budget_5pct",
    "10pct": "budget_10pct",
    "partial": "vidal_partial",
    "full": "vidal_full",
}


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _windows(parts: list[tuple[np.ndarray, np.ndarray, str]], length: int):
    """Reshape a list of (features, soc, id) time series into non-overlapping windows."""
    xs, ys, ids = [], [], []
    for x, y, item_id in parts:
        usable = min(len(x), len(y)) // length * length
        if usable:
            count = usable // length
            xs.append(x[:usable].reshape(count, length, x.shape[-1]))
            ys.append(y[:usable].reshape(count, length))
            ids.append((item_id, count))
    if not xs:
        raise ValueError("No complete non-overlapping windows")
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32), ids


def _source_data(root: Path, length: int):
    """Panasonic 25 C, all nine profiles, already min-max normalised to [0, 1]."""
    cycles, feature_min, feature_max = load_temperature_cycles_and_range(root, 25)
    parts = [(cycles[name].features, cycles[name].soc, name) for name in PROFILES]
    x, y, _ = _windows(parts, length)
    return x, y, feature_min, feature_max


def _cell_path(cell: str, base_root: Path, condition_root: Path) -> Path:
    root = base_root if cell in BASELINE_CELLS else condition_root
    return root / f"{cell}_dynamic_soc.parquet"


def _budget_mission_ids(manifest: pd.DataFrame, cell_id: str, budget: str) -> tuple[set[str], set[str]]:
    rows = manifest.loc[manifest["cell_id"] == cell_id]
    if rows.empty:
        raise ValueError(f"{cell_id}: no manifest rows")
    test_ids = set(rows.loc[rows["split"] == "test", "mission_cycle_id"].astype(str))
    if budget == "zero_shot":
        return set(), test_ids
    column = BUDGET_COLUMNS[budget]
    train = rows.loc[rows["split"] == "train"]
    adapt_ids = set(train.loc[train[column].astype(bool), "mission_cycle_id"].astype(str))
    if not adapt_ids:
        raise ValueError(f"{cell_id}/{budget}: empty adaptation mission set")
    return adapt_ids, test_ids


def _target_windows(records, mission_ids: set[str], feature_min, feature_max, length: int):
    if not mission_ids:
        raise ValueError("mission_ids is empty")
    scale = np.where(feature_max == feature_min, 1.0, feature_max - feature_min)
    parts = []
    for record in records:
        mission = str(record.metadata["mission_cycle_id"])
        if mission not in mission_ids:
            continue
        frame = record.data
        x = (frame.loc[:, list(FEATURES)].to_numpy(dtype=np.float64) - feature_min) / scale
        y = frame["soc"].to_numpy(dtype=np.float64)
        parts.append((x, y, mission))
    if not parts:
        raise ValueError("No CMU records matched the requested mission_ids")
    return _windows(parts, length)


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=min(batch_size, len(x)),
        shuffle=shuffle,
    )


def _train(model: nn.Module, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int,
           lr: float, device: torch.device, label: str) -> None:
    """Full-parameter supervised MSE training (used for both source pretraining
    and target fine-tuning -- Vidal Case 2 unlocks every trainable parameter
    for target fine-tuning rather than freezing any layer)."""
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
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % max(epochs // 5, 1) == 0:
            print(f"    {label} epoch {epoch + 1}/{epochs}: mse={total / len(x):.6f}")


@torch.no_grad()
def _evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, missions, batch_size: int,
              device: torch.device) -> dict[str, float]:
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
            "mission_macro_mae_percent": float(np.mean([m["mae"] for m in per_mission]) * 100),
            "mission_macro_rmse_percent": float(np.mean([m["rmse"] for m in per_mission]) * 100),
            "missions": len(missions),
            "windows": len(x),
        }
    )
    return overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panasonic-data-root", type=Path, default=REPO_ROOT / "data/raw/Panasonic_18650PF"
    )
    parser.add_argument(
        "--cmu-base-root", type=Path,
        default=REPO_ROOT / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1",
    )
    parser.add_argument(
        "--cmu-condition-root", type=Path,
        default=REPO_ROOT / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1_extension4",
    )
    parser.add_argument(
        "--cmu-manifest", type=Path,
        default=REPO_ROOT / "data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1.csv",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO_ROOT / "outputs/benchmarks/panasonic_to_cmu_evtol/backbone_transfer",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS))
    parser.add_argument("--budgets", nargs="+", choices=BUDGETS, default=list(BUDGETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--source-epochs", type=int, default=200)
    parser.add_argument("--transfer-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--transfer-lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="skip (model, cell, seed) triples with an existing result file")
    return parser.parse_args()


def _rebuild_cell_metrics(output_root: Path) -> pd.DataFrame:
    """Aggregate every per-(model, seed, cell) result.json under output_root.

    cell_metrics.csv is shared across models, and each backbone is normally
    launched as its own subprocess/notebook cell (one ``--models`` value per
    invocation). Writing it from the current process's in-memory rows alone
    would make the last model to run silently overwrite the others' rows in
    the CSV, even though their result.json files are untouched on disk. This
    rebuilds the CSV from every result.json found, so the file always
    reflects everything completed so far, regardless of which invocation ran
    last.
    """
    all_rows: list[dict] = []
    for result_path in sorted(output_root.glob("*/seed_*/*/result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        all_rows.extend(payload["metrics"])
    table = pd.DataFrame(all_rows)
    table.to_csv(output_root / "cell_metrics.csv", index=False)
    return table


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    print("loading Panasonic 25C source windows ...")
    source_x, source_y, feature_min, feature_max = _source_data(
        args.panasonic_data_root, args.sequence_length
    )
    manifest = pd.read_csv(args.cmu_manifest)

    rows: list[dict] = []
    param_counts: dict[str, int] = {}

    for model_name in args.models:
        for seed in args.seeds:
            _seed(seed)
            print(f"=== {model_name} / seed {seed}: pretraining on {len(source_x)} source windows ===")
            model = ModelRegistry.build(model_name, {"input_size": len(FEATURES)}).to(device)
            param_counts[model_name] = model.count_parameters()
            _train(model, source_x, source_y, args.source_epochs, args.batch_size, args.lr, device, "source")
            source_state = copy.deepcopy(model.state_dict())

            for cell in args.cells:
                result_path = args.output_root / model_name / f"seed_{seed}" / cell / "result.json"
                if args.resume and result_path.is_file():
                    rows.extend(json.loads(result_path.read_text(encoding="utf-8"))["metrics"])
                    print(f"  {cell}: resumed from {result_path}")
                    continue

                path = _cell_path(cell, args.cmu_base_root, args.cmu_condition_root)
                expected = set(manifest.loc[manifest["cell_id"] == cell, "source_artifact_sha256"].astype(str))
                actual = sha256_file(path)
                if expected != {actual}:
                    raise ValueError(f"{cell}: CMU input hash does not match manifest")
                records = load_cmu_dynamic_soc_for_vidal([path], target_hz=1.0).records

                cell_metrics = []
                for budget in args.budgets:
                    adapt_ids, test_ids = _budget_mission_ids(manifest, cell, budget)
                    test_x, test_y, test_missions = _target_windows(
                        records, test_ids, feature_min, feature_max, args.sequence_length
                    )
                    eval_model = ModelRegistry.build(model_name, {"input_size": len(FEATURES)}).to(device)
                    eval_model.load_state_dict(source_state)
                    adapt_missions = 0
                    if budget != "zero_shot":
                        adapt_x, adapt_y, adapt_mission_ids = _target_windows(
                            records, adapt_ids, feature_min, feature_max, args.sequence_length
                        )
                        adapt_missions = len(adapt_mission_ids)
                        _seed(seed)
                        _train(
                            eval_model, adapt_x, adapt_y, args.transfer_epochs,
                            args.batch_size, args.transfer_lr, device, budget,
                        )
                    metrics = _evaluate(eval_model, test_x, test_y, test_missions, args.batch_size, device)
                    row = {
                        "model": model_name,
                        "cell_id": cell,
                        "budget": budget,
                        "seed": seed,
                        "adaptation_missions": adapt_missions,
                        "n_parameters": param_counts[model_name],
                        **metrics,
                    }
                    cell_metrics.append(row)
                    print(
                        f"  {cell}/{budget}: MAE={metrics['overall_mae_percent']:.3f}% "
                        f"RMSE={metrics['overall_rmse_percent']:.3f}%"
                    )

                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(
                    json.dumps(
                        {
                            "model": model_name,
                            "cell_id": cell,
                            "seed": seed,
                            "cmu_input_sha256": actual,
                            "n_parameters": param_counts[model_name],
                            "metrics": cell_metrics,
                            "label_is_ground_truth": False,
                            "formal_benchmark_eligible": False,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                rows.extend(cell_metrics)
                _rebuild_cell_metrics(args.output_root)

    table = _rebuild_cell_metrics(args.output_root)
    metadata = {
        "benchmark": "backbone_transfer_panasonic_25c_to_cmu_core7",
        "note": (
            "Reimplements the Vidal transfer protocol (source-fitted 1Hz "
            "windows, nested chronological target-label budgets, "
            "restart-from-source full-parameter fine-tuning) for CNN-LSTM, "
            "TCN, and Transformer-encoder backbones. Does not import "
            "battery_benchmark.reproductions.vidal_2019; the published Vidal "
            "LSTM CMU results are untouched by this script."
        ),
        "models": sorted(set(table["model"]) | set(args.models)) if not table.empty else list(args.models),
        "n_parameters": param_counts,
        "cells": list(args.cells),
        "budgets": list(args.budgets),
        "budget_columns": BUDGET_COLUMNS,
        "seeds": list(args.seeds),
        "feature_order": list(FEATURES),
        "sequence_length": args.sequence_length,
        "source_profiles": list(PROFILES),
        "source_temperature_c": 25,
        "source_feature_min": feature_min.tolist(),
        "source_feature_max": feature_max.tolist(),
        "normalization": "Panasonic 25C source-fitted min-max applied unchanged to CMU target",
        "source_epochs": args.source_epochs,
        "transfer_epochs": args.transfer_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "transfer_lr": args.transfer_lr,
        "target_soc_labels_used_for_adaptation": True,
        "target_soc_labels_used_for_checkpoint_selection": False,
        "target_soc_reference_used_for_posthoc_evaluation": True,
        "label_is_ground_truth": False,
        "formal_benchmark_eligible": False,
        "git_commit": get_full_commit_hash(REPO_ROOT),
        "git_dirty": is_dirty(REPO_ROOT),
    }
    metadata_path = args.output_root / "run_metadata.json"
    if metadata_path.is_file():
        # run_metadata.json is shared across models the same way cell_metrics.csv
        # is (each backbone is normally launched as its own invocation): merge
        # into the existing file instead of overwriting other models' entries.
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["models"] = sorted(set(previous.get("models", [])) | set(metadata["models"]))
        metadata["n_parameters"] = {**previous.get("n_parameters", {}), **metadata["n_parameters"]}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
