#!/usr/bin/env python3
"""Run the Shen et al. (2023) MED source-free method on the CMU eVTOL target
domain (Panasonic 25C source -> CMU eVTOL core-seven cells).

This is the natural "zero_shot" comparison point for
``run_backbone_transfer_cmu.py``: MED never sees any CMU labels during
adaptation (no real target SOC, no source replay), so there is no notion of
a label budget here the way the backbone-transfer sweep has one. Per cell,
per seed, this script reports exactly two (optionally three) rows:

* ``zero_shot``    -- the frozen Panasonic source model applied directly to
                       the CMU target domain with no adaptation at all.
* ``med``          -- Algorithm 1's source-free transfer: the target feature
                       extractor is adapted using only pseudo-labels drawn
                       from the frozen source model's own averaged-head
                       output on CMU's *unlabeled* train-split missions.
                       Real CMU labels are used only afterwards, to score
                       the held-out test-split missions.
* ``dann_mmd``     -- optional (``--include-dann-mmd-baseline``): the
                       paper's own domain-adversarial + MMD comparison
                       baseline, run the same way.

Like ``run_backbone_transfer_cmu.py``, this script does not import
``battery_benchmark.reproductions.vidal_2019`` case code directly and does
not touch the already-published, formal-status Vidal LSTM results. CMU
``soc_dynamic`` is a development dynamic-capacity reference, not measured
ground truth, so every row this script writes is
``label_is_ground_truth=false`` / ``formal_benchmark_eligible=false`` --
identical convention to the backbone-transfer benchmark.

Pseudo-label confidence filtering (``build_pseudo_target_dataset``) runs
per CMU *mission cycle*, exactly like it runs per Panasonic *profile* in the
in-paper suite -- this avoids the cross-file-boundary artifact identified in
the collaborator's delivered reference code, where the same filter is
applied to samples flattened across unrelated drive-cycle files first.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from battery_benchmark.datasets.cmu_evtol.vidal_adapter import load_cmu_dynamic_soc_for_vidal
from battery_benchmark.evaluation.metrics import compute_metrics
from battery_benchmark.reproductions.shen_2023.data import (
    PROFILES,
    ShenCycle,
    load_temperature_cycles_and_range,
    windows,
)
from battery_benchmark.reproductions.shen_2023.model import MEDNet, DANNBaselineNet
from battery_benchmark.reproductions.shen_2023.train import (
    adapt_dann_mmd as _adapt_dann_mmd,
    build_pseudo_target_dataset,
    load_med_source,
    load_source as load_dann_source,
    med_adapt,
    pretrain_med,
    predict,
    train_source as train_dann_source,
)
from battery_benchmark.utils.device import get_device
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty
from battery_benchmark.utils.hashing import sha256_file

CELLS = ("VAH01", "VAH17", "VAH27", "VAH30", "VAH15", "VAH12", "VAH11")
BASELINE_CELLS = {"VAH01", "VAH17", "VAH27"}
FEATURES = ("current", "voltage", "temperature")
WINDOW = 90  # fixed by the MED/TATN architecture (Conv2d/LSTM shapes in model.py)
METHOD_CHOICES = ("zero_shot", "med", "dann_mmd")


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


class _Timer:
    """Wall-clock stopwatch. CUDA ops are async, so any timed block that ran
    on GPU is synced before stopping -- otherwise the measured duration would
    just be launch-overhead, not actual compute time."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def __enter__(self) -> "_Timer":
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self._start


def _cell_path(cell: str, base_root: Path, condition_root: Path) -> Path:
    root = base_root if cell in BASELINE_CELLS else condition_root
    return root / f"{cell}_dynamic_soc.parquet"


def _mission_cycles(
    records, mission_ids: set[str], feature_min: np.ndarray, feature_max: np.ndarray
) -> list[ShenCycle]:
    """Build one ShenCycle per CMU mission, normalised with the Panasonic
    source-fitted min-max range (source-fitted normalisation applied
    unchanged to the target domain, same convention as the backbone-transfer
    benchmark)."""

    if not mission_ids:
        raise ValueError("mission_ids is empty")
    scale = np.where(feature_max == feature_min, 1.0, feature_max - feature_min)
    cycles = []
    for record in records:
        mission = str(record.metadata["mission_cycle_id"])
        if mission not in mission_ids:
            continue
        frame = record.data
        x = ((frame.loc[:, list(FEATURES)].to_numpy(dtype=np.float64) - feature_min) / scale).astype(
            np.float32
        )
        y = frame["soc"].to_numpy(dtype=np.float64).astype(np.float32)
        cycles.append(ShenCycle(profile=mission, temperature=None, features=x, soc=y, paths=()))
    if not cycles:
        raise ValueError("No CMU records matched the requested mission_ids")
    return cycles


def _budget_mission_ids(manifest: pd.DataFrame, cell_id: str) -> tuple[set[str], set[str]]:
    """Every train-split mission is unlabeled adaptation data for MED (no
    label budget applies -- MED never uses target labels for training)."""

    rows = manifest.loc[manifest["cell_id"] == cell_id]
    if rows.empty:
        raise ValueError(f"{cell_id}: no manifest rows")
    train_ids = set(rows.loc[rows["split"] == "train", "mission_cycle_id"].astype(str))
    test_ids = set(rows.loc[rows["split"] == "test", "mission_cycle_id"].astype(str))
    if not train_ids:
        raise ValueError(f"{cell_id}: empty train-split mission set")
    return train_ids, test_ids


@torch.no_grad()
def _evaluate(model, cycles: list[ShenCycle], device: torch.device, branch: str = "target") -> dict[str, float]:
    truth, estimate = predict(model, cycles, device, branch=branch)
    estimate = np.clip(estimate, 0.0, 1.0)
    metrics = compute_metrics(truth, estimate)
    per_mission = [
        compute_metrics(*predict(model, [cycle], device, branch=branch)) for cycle in cycles
    ]
    return {
        "overall_mae_percent": metrics["mae"] * 100,
        "overall_rmse_percent": metrics["rmse"] * 100,
        "overall_max_error_percent": metrics["max_error"] * 100,
        "overall_bias_percent": metrics["bias"] * 100,
        "overall_r2": metrics["r2"],
        "mission_macro_mae_percent": float(np.mean([m["mae"] for m in per_mission]) * 100),
        "mission_macro_rmse_percent": float(np.mean([m["rmse"] for m in per_mission]) * 100),
        "missions": len(cycles),
        "windows": sum(len(windows([c])[0]) for c in cycles),
    }


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
        default=REPO_ROOT / "outputs/benchmarks/panasonic_to_cmu_evtol/sftnn_med_transfer",
    )
    parser.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS))
    parser.add_argument("--methods", nargs="+", choices=METHOD_CHOICES, default=["zero_shot", "med"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--source-epochs", type=int, default=2000)
    parser.add_argument("--transfer-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--transfer-lr", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--pseudo-threshold", type=float, default=0.08)
    parser.add_argument("--pseudo-lookahead", type=int, default=200)
    parser.add_argument("--med-lambda-soc", type=float, default=1.0)
    parser.add_argument("--med-lambda-dis", type=float, default=0.1)
    parser.add_argument(
        "--dann-lambdas", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        metavar=("ESTIMATE", "ADVERSARIAL", "MMD"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume", action="store_true",
        help="skip source pretraining and (method, seed, cell) evaluations that "
        "already have a result file on disk",
    )
    parser.add_argument(
        "--no-save-checkpoints", action="store_true",
        help="do not write model checkpoints; result JSON rows are always written",
    )
    return parser.parse_args()


def _save_timing(seed_root: Path, name: str, *, seconds: float | None, epochs: int | None, device: str) -> None:
    """Persist a pretraining timing record so it survives --resume: if this
    run resumed from a checkpoint instead of retraining, we still want the
    original run's timing available for the rows written this time."""

    seed_root.mkdir(parents=True, exist_ok=True)
    path = seed_root / f"{name}_timing.json"
    if seconds is None:
        return  # nothing new measured this run; leave any existing file alone
    path.write_text(
        json.dumps({"seconds": seconds, "epochs": epochs, "device": device}, indent=2),
        encoding="utf-8",
    )


def _load_timing(seed_root: Path, name: str) -> dict:
    path = seed_root / f"{name}_timing.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"seconds": None, "epochs": None, "device": None}


def _rebuild_cell_metrics(output_root: Path) -> pd.DataFrame:
    """Aggregate every per-(method, seed, cell) result JSON under output_root,
    the same merge-not-overwrite convention run_backbone_transfer_cmu.py
    uses for cell_metrics.csv."""

    all_rows: list[dict] = []
    for result_path in sorted(output_root.glob("seed_*/*/*.json")):
        all_rows.append(json.loads(result_path.read_text(encoding="utf-8")))
    table = pd.DataFrame(all_rows)
    table.to_csv(output_root / "cell_metrics.csv", index=False)
    return table


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    device = get_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    print("loading Panasonic 25C source cycles (MED source-free pretraining) ...")
    source_cycles, feature_min, feature_max = load_temperature_cycles_and_range(
        args.panasonic_data_root, 25
    )
    manifest = pd.read_csv(args.cmu_manifest)

    for seed in args.seeds:
        seed_root = args.output_root / f"seed_{seed}"
        med_checkpoint = seed_root / "med_source.pt"
        dann_checkpoint = seed_root / "dann_source.pt"

        med_pretrain_seconds = None
        med_pretrain_epochs = None
        _seed(seed)
        if args.resume and med_checkpoint.is_file():
            med_source = load_med_source(med_checkpoint, device)
            print(f"=== seed {seed}: resumed MED source checkpoint from {med_checkpoint} ===")
        else:
            print(f"=== seed {seed}: MED source pretraining, max {args.source_epochs} epochs ===")
            with _Timer(device) as timer:
                med_source, med_pretrain_history = pretrain_med(
                    source_cycles,
                    epochs=args.source_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    device=device,
                    seed=seed,
                    checkpoint=None if args.no_save_checkpoints else med_checkpoint,
                    early_stopping_patience=args.early_stopping_patience,
                )
            med_pretrain_seconds = timer.seconds
            med_pretrain_epochs = len(med_pretrain_history)
            print(f"    done in {med_pretrain_seconds:.1f}s ({med_pretrain_epochs} epochs)")
        _save_timing(
            seed_root, "med_source_pretrain",
            seconds=med_pretrain_seconds, epochs=med_pretrain_epochs, device=str(device),
        )
        med_pretrain_timing = _load_timing(seed_root, "med_source_pretrain")

        dann_source = None
        dann_pretrain_seconds = None
        dann_pretrain_epochs = None
        if "dann_mmd" in args.methods:
            if args.resume and dann_checkpoint.is_file():
                dann_source = load_dann_source(dann_checkpoint, device)
                print(f"=== seed {seed}: resumed DANN+MMD source checkpoint from {dann_checkpoint} ===")
            else:
                print(f"=== seed {seed}: DANN+MMD source pretraining, max {args.source_epochs} epochs ===")
                with _Timer(device) as timer:
                    dann_source, dann_pretrain_history = train_dann_source(
                        source_cycles,
                        epochs=args.source_epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.lr,
                        device=device,
                        seed=seed,
                        checkpoint=None if args.no_save_checkpoints else dann_checkpoint,
                        early_stopping_patience=args.early_stopping_patience,
                    )
                dann_pretrain_seconds = timer.seconds
                dann_pretrain_epochs = len(dann_pretrain_history)
                print(f"    done in {dann_pretrain_seconds:.1f}s ({dann_pretrain_epochs} epochs)")
            _save_timing(
                seed_root, "dann_source_pretrain",
                seconds=dann_pretrain_seconds, epochs=dann_pretrain_epochs, device=str(device),
            )
        dann_pretrain_timing = _load_timing(seed_root, "dann_source_pretrain")

        for cell in args.cells:
            cell_root = seed_root / cell
            path = _cell_path(cell, args.cmu_base_root, args.cmu_condition_root)
            expected = set(
                manifest.loc[manifest["cell_id"] == cell, "source_artifact_sha256"].astype(str)
            )
            actual = sha256_file(path)
            if expected != {actual}:
                raise ValueError(f"{cell}: CMU input hash does not match manifest")
            records = load_cmu_dynamic_soc_for_vidal([path], target_hz=1.0).records
            train_ids, test_ids = _budget_mission_ids(manifest, cell)
            test_cycles = _mission_cycles(records, test_ids, feature_min, feature_max)
            train_cycles = _mission_cycles(records, train_ids, feature_min, feature_max)

            for method in args.methods:
                result_path = cell_root / f"{method}.json"
                if args.resume and result_path.is_file():
                    print(f"  {cell}/{method}: resumed from {result_path}")
                    continue

                extra = {}
                timing = {"pseudo_label_seconds": None, "adapt_seconds": None}
                if method == "zero_shot":
                    with _Timer(device) as eval_timer:
                        metrics = _evaluate(med_source, test_cycles, device, branch="source")
                elif method == "med":
                    with _Timer(device) as pseudo_timer:
                        dataset, info = build_pseudo_target_dataset(
                            med_source, train_cycles, device,
                            threshold=args.pseudo_threshold, lookahead=args.pseudo_lookahead,
                        )
                    if len(dataset) == 0:
                        raise RuntimeError(
                            f"{cell}: no pseudo-labels survived confidence filtering; "
                            "try a higher --pseudo-threshold"
                        )
                    with _Timer(device) as adapt_timer:
                        adapted, history = med_adapt(
                            med_source, dataset,
                            epochs=args.transfer_epochs, batch_size=args.batch_size,
                            learning_rate=args.transfer_lr, device=device, seed=seed,
                            lambdas=(args.med_lambda_soc, args.med_lambda_dis),
                            early_stopping_patience=args.early_stopping_patience,
                        )
                    with _Timer(device) as eval_timer:
                        metrics = _evaluate(adapted, test_cycles, device, branch="target")
                    timing["pseudo_label_seconds"] = pseudo_timer.seconds
                    timing["adapt_seconds"] = adapt_timer.seconds
                    extra = {"pseudo_label_info": info, "epochs_completed": len(history)}
                    if not args.no_save_checkpoints:
                        cell_root.mkdir(parents=True, exist_ok=True)
                        torch.save(adapted.target_extractor.state_dict(), cell_root / "med.pt")
                elif method == "dann_mmd":
                    assert dann_source is not None
                    with _Timer(device) as adapt_timer:
                        adapted, history, info = _adapt_dann_mmd(
                            dann_source, source_cycles, train_cycles,
                            test_profile=None, scenario="unsupervised",
                            epochs=args.transfer_epochs, batch_size=args.batch_size,
                            learning_rate=args.transfer_lr, lambdas=tuple(args.dann_lambdas),
                            device=device, seed=seed,
                        )
                    with _Timer(device) as eval_timer:
                        metrics = _evaluate(adapted, test_cycles, device, branch="target")
                    timing["adapt_seconds"] = adapt_timer.seconds
                    extra = {"pseudo_label_info": info, "epochs_completed": len(history)}
                    if not args.no_save_checkpoints:
                        cell_root.mkdir(parents=True, exist_ok=True)
                        torch.save(adapted.target_extractor.state_dict(), cell_root / "dann_mmd.pt")
                else:
                    raise ValueError(method)

                row = {
                    "method": method,
                    "cell_id": cell,
                    "seed": seed,
                    "device": str(device),
                    "med_source_pretrain_seconds": med_pretrain_timing["seconds"],
                    "med_source_pretrain_epochs": med_pretrain_timing["epochs"],
                    "dann_source_pretrain_seconds": dann_pretrain_timing["seconds"],
                    "dann_source_pretrain_epochs": dann_pretrain_timing["epochs"],
                    "pseudo_label_seconds": timing["pseudo_label_seconds"],
                    "adapt_seconds": timing["adapt_seconds"],
                    "eval_seconds": eval_timer.seconds,
                    "cmu_input_sha256": actual,
                    "n_train_missions": len(train_cycles),
                    "n_test_missions": len(test_cycles),
                    "label_is_ground_truth": False,
                    "formal_benchmark_eligible": False,
                    **extra,
                    **metrics,
                }
                row_seconds = sum(
                    s for s in (timing["pseudo_label_seconds"], timing["adapt_seconds"], eval_timer.seconds)
                    if s is not None
                )
                print(
                    f"  {cell}/{method}: MAE={metrics['overall_mae_percent']:.3f}% "
                    f"RMSE={metrics['overall_rmse_percent']:.3f}% ({row_seconds:.1f}s)"
                )
                cell_root.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
                _rebuild_cell_metrics(args.output_root)

    table = _rebuild_cell_metrics(args.output_root)
    metadata = {
        "benchmark": "sftnn_med_transfer_panasonic_25c_to_cmu_core7",
        "note": (
            "MED (Shen et al. 2023 Algorithm 1) source-free transfer from "
            "Panasonic 25C to the CMU eVTOL core-seven cells, plus an "
            "optional zero-shot baseline and an optional DANN+MMD "
            "comparison baseline. MED's transfer stage uses no CMU labels "
            "and no Panasonic source data/labels -- only pseudo-labels from "
            "the frozen source model's averaged-head output on CMU's "
            "unlabeled train-split missions. Real CMU labels are used only "
            "for post-hoc scoring of the held-out test-split missions."
        ),
        "methods": list(args.methods),
        "cells": list(args.cells),
        "seeds": list(args.seeds),
        "feature_order": list(FEATURES),
        "window": WINDOW,
        "source_profiles": list(PROFILES),
        "source_temperature_c": 25,
        "source_feature_min": feature_min.tolist(),
        "source_feature_max": feature_max.tolist(),
        "normalization": "Panasonic 25C source-fitted min-max applied unchanged to CMU target",
        "pseudo_threshold": args.pseudo_threshold,
        "pseudo_lookahead": args.pseudo_lookahead,
        "med_lambdas": [args.med_lambda_soc, args.med_lambda_dis],
        "dann_lambdas": args.dann_lambdas,
        "source_epochs": args.source_epochs,
        "transfer_epochs": args.transfer_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "transfer_lr": args.transfer_lr,
        "target_soc_labels_used_for_adaptation": False,
        "target_soc_labels_used_for_checkpoint_selection": False,
        "target_soc_reference_used_for_posthoc_evaluation": True,
        "label_is_ground_truth": False,
        "formal_benchmark_eligible": False,
        "git_commit": get_full_commit_hash(REPO_ROOT),
        "git_dirty": is_dirty(REPO_ROOT),
        # Wall-clock time spent by *this invocation* only -- with --resume,
        # anything skipped because a result file already existed is not
        # counted here. Per-row timing breakdowns (pretrain/pseudo-label/
        # adapt/eval seconds, plus the device they ran on) live in
        # cell_metrics.csv, which is cumulative across every invocation.
        "this_invocation_wall_seconds": time.perf_counter() - run_start,
        "device": str(device),
    }
    metadata_path = args.output_root / "run_metadata.json"
    if metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["methods"] = sorted(set(previous.get("methods", [])) | set(metadata["methods"]))
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nthis invocation took {metadata['this_invocation_wall_seconds']:.1f}s wall-clock")


if __name__ == "__main__":
    main()
