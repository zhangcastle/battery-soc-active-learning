#!/usr/bin/env python3
"""Run Vidal-style Panasonic-to-CMU zero-shot or supervised adaptation."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.datasets.cmu_evtol.vidal_adapter import (
    build_cmu_vidal_windows,
    load_cmu_dynamic_soc_for_vidal,
    transform_cmu_records,
)
from battery_benchmark.evaluation.metrics import compute_metrics
from battery_benchmark.preprocessing.normalization import MinMaxNormalizer
from battery_benchmark.reproductions.vidal_2019.case1 import masked_mse
from battery_benchmark.reproductions.vidal_2019.case2 import (
    load_case1_weights_for_transfer,
)
from battery_benchmark.reproductions.vidal_2019.model import VidalLSTM
from battery_benchmark.reproductions.vidal_2019.windows import (
    FEATURES,
    Case1WindowDataset,
)
from battery_benchmark.training.checkpointing import load_checkpoint, save_checkpoint
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty
from battery_benchmark.utils.hashing import sha256_file


CELLS = ("VAH01", "VAH17", "VAH27", "VAH30", "VAH15", "VAH12", "VAH11")
SETTINGS = (
    "zero_shot",
    "budget_1pct",
    "budget_5pct",
    "budget_10pct",
    "vidal_partial",
    "vidal_full",
)
TRAIN_COLUMNS = {
    "budget_1pct": "budget_1pct",
    "budget_5pct": "budget_5pct",
    "budget_10pct": "budget_10pct",
    "vidal_partial": "vidal_partial",
    "vidal_full": "vidal_full",
}


@dataclass
class RunConfig:
    setting: str
    random_seed: int = 42
    seq_len: int = 50
    stride: int = 50
    epochs: int = 515
    batch_size: int = 512
    learning_rate: float = 0.01
    scheduler_step_size: int = 100
    scheduler_gamma: float = 0.5
    num_workers: int = 0


def _cell_path(cell_id: str, base_root: Path, extension_root: Path) -> Path:
    root = base_root if cell_id in {"VAH01", "VAH17", "VAH27"} else extension_root
    return root / f"{cell_id}_dynamic_soc.parquet"


def _records_for_ids(records, mission_ids: set[str]):
    return [
        record
        for record in records
        if str(record.metadata["mission_cycle_id"]) in mission_ids
    ]


def select_mission_ids(
    manifest: pd.DataFrame, cell_id: str, setting: str
) -> tuple[set[str], set[str], set[str]]:
    rows = manifest.loc[manifest["cell_id"] == cell_id].copy()
    if rows.empty or rows["mission_cycle_id"].duplicated().any():
        raise ValueError(f"{cell_id}: missing or duplicate mission manifest rows")
    train_column = TRAIN_COLUMNS.get(setting)
    train = (
        set(rows.loc[rows[train_column].astype(bool), "mission_cycle_id"].astype(str))
        if train_column
        else set()
    )
    validation = set(
        rows.loc[rows["split"] == "validation", "mission_cycle_id"].astype(str)
    )
    test = set(rows.loc[rows["split"] == "test", "mission_cycle_id"].astype(str))
    if train & validation or train & test or validation & test:
        raise ValueError(f"{cell_id}: mission split leakage")
    return train, validation, test


def _fit_target_scaler(records) -> MinMaxNormalizer:
    if not records:
        raise ValueError("Cannot fit target scaler without training records")
    values = np.concatenate(
        [record.data.loc[:, list(FEATURES)].to_numpy(np.float32) for record in records]
    )
    return MinMaxNormalizer().fit(values)


def _load_source_scaler(
    checkpoint_path: Path, scaler_path: Path, model: VidalLSTM, device: torch.device
) -> MinMaxNormalizer:
    if scaler_path.parent != checkpoint_path.parent.parent:
        raise ValueError("Source checkpoint and scaler must come from the same Case 1 run")
    scaler = MinMaxNormalizer()
    load_checkpoint(checkpoint_path, model, normalizer=scaler, device=device)
    saved = json.loads(scaler_path.read_text(encoding="utf-8"))
    expected_min = saved.get("feature_min", saved.get("min"))
    expected_max = saved.get("feature_max", saved.get("max"))
    state = scaler.state_dict()
    if not (
        expected_min is not None
        and expected_max is not None
        and np.array_equal(np.asarray(expected_min), np.asarray(state["min"]))
        and np.array_equal(np.asarray(expected_max), np.asarray(state["max"]))
    ):
        raise ValueError("Source scaler.json does not match checkpoint normalizer state")
    return scaler


def _train(
    model: VidalLSTM,
    windows,
    cfg: RunConfig,
    device: torch.device,
) -> tuple[pd.DataFrame, torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR]:
    dataset = Case1WindowDataset(windows)
    generator = torch.Generator().manual_seed(cfg.random_seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=cfg.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, cfg.scheduler_step_size, cfg.scheduler_gamma
    )
    if sum(p.numel() for group in optimizer.param_groups for p in group["params"]) != 571:
        raise ValueError("Optimizer must contain exactly 571 trainable parameters")

    history = []
    model.to(device)
    for epoch in range(cfg.epochs):
        started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        valid_sum = 0
        for x, y, mask, _ in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(x), y, mask)
            loss.backward()
            optimizer.step()
            valid = int(mask.sum().item())
            loss_sum += float(loss.item()) * valid
            valid_sum += valid
        history.append(
            {
                "epoch": epoch + 1,
                "train_mse": loss_sum / max(valid_sum, 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        scheduler.step()
    return pd.DataFrame(history), optimizer, scheduler


@torch.no_grad()
def evaluate_target_last(
    model: VidalLSTM,
    windows,
    phase_by_source_row: dict[int, str],
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, float]]:
    loader = DataLoader(
        Case1WindowDataset(windows), batch_size=batch_size, shuffle=False
    )
    model.eval()
    rows = []
    for x, y, mask, indices in loader:
        predictions = model(x.to(device)).cpu().numpy()
        for batch_index, window_index in enumerate(indices.tolist()):
            metadata = windows.metadata[window_index]
            target_index = int(mask[batch_index].sum().item()) - 1
            reference = float(y[batch_index, target_index])
            prediction = float(predictions[batch_index, target_index])
            source_row = int(metadata["source_row_right"])
            rows.append(
                {
                    "cell_id": metadata["cell_id"],
                    "mission_cycle_id": metadata["mission_cycle_id"],
                    "window_index": window_index,
                    "source_row_right": source_row,
                    "development_phase_role": phase_by_source_row.get(
                        source_row, "NOT_AVAILABLE"
                    ),
                    "soc_reference": reference,
                    "soc_prediction": prediction,
                    "absolute_error": abs(prediction - reference),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Evaluation produced no complete windows")
    metric = compute_metrics(frame["soc_reference"], frame["soc_prediction"])
    mission_metrics = [
        compute_metrics(group["soc_reference"], group["soc_prediction"])
        for _, group in frame.groupby("mission_cycle_id", sort=False)
    ]
    result = {
        "overall_mae": float(metric["mae"]),
        "overall_rmse": float(metric["rmse"]),
        "mission_macro_mae": float(
            np.mean([metric["mae"] for metric in mission_metrics])
        ),
        "mission_macro_rmse": float(
            np.mean([metric["rmse"] for metric in mission_metrics])
        ),
    }
    for role in ("high_1", "lower", "high_2"):
        group = frame.loc[frame["development_phase_role"] == role]
        role_metric = (
            compute_metrics(group["soc_reference"], group["soc_prediction"])
            if not group.empty
            else {"mae": math.nan, "rmse": math.nan}
        )
        result[f"{role}_mae"] = float(role_metric["mae"])
        result[f"{role}_rmse"] = float(role_metric["rmse"])
    return frame, result


def _latest_zero_shot_metrics(output_root: Path, seed: int) -> pd.DataFrame:
    candidates = sorted(
        (output_root / "zero_shot" / f"seed_{seed}").glob("*/cell_metrics.csv")
    )
    return pd.read_csv(candidates[-1]) if candidates else pd.DataFrame()


def run_setting(
    cfg: RunConfig,
    manifest_path: Path,
    base_root: Path,
    extension_root: Path,
    output_root: Path,
    checkpoint_path: Path,
    scaler_path: Path,
    repo_root: Path,
    device_name: str = "auto",
) -> Path:
    if cfg.setting not in SETTINGS:
        raise ValueError(f"Unsupported setting: {cfg.setting}")
    if cfg.stride != cfg.seq_len:
        raise ValueError("Vidal windows require stride == seq_len")
    manifest = pd.read_csv(manifest_path)
    if set(manifest["cell_id"]) != set(CELLS):
        raise ValueError("Split manifest must contain exactly the core7 cells")
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    run_id = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    run_root = output_root / cfg.setting / f"seed_{cfg.random_seed}" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    campaign_started = time.perf_counter()
    cell_rows = []

    for cell_id in CELLS:
        started = time.perf_counter()
        source_path = _cell_path(cell_id, base_root, extension_root)
        raw_phase = pd.read_parquet(
            source_path,
            columns=["source_row", "development_phase_role"],
        )
        phase_by_source_row = dict(
            zip(
                raw_phase["source_row"].astype(int),
                raw_phase["development_phase_role"].astype(str),
            )
        )
        adapted = load_cmu_dynamic_soc_for_vidal([source_path], target_hz=1.0)
        train_ids, validation_ids, test_ids = select_mission_ids(
            manifest, cell_id, cfg.setting
        )
        all_ids = {
            str(record.metadata["mission_cycle_id"]) for record in adapted.records
        }
        manifest_ids = set(
            manifest.loc[
                manifest["cell_id"] == cell_id, "mission_cycle_id"
            ].astype(str)
        )
        if all_ids != manifest_ids:
            raise ValueError(f"{cell_id}: adapter and split manifest mission sets differ")
        train_records = _records_for_ids(adapted.records, train_ids)
        validation_records = _records_for_ids(adapted.records, validation_ids)
        test_records = _records_for_ids(adapted.records, test_ids)

        torch.manual_seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)
        model = VidalLSTM()
        source_metadata = load_case1_weights_for_transfer(
            model,
            checkpoint_path,
            "panasonic_18650pf",
            require_formal=True,
            device=device,
        )
        if cfg.setting == "zero_shot":
            scaler = _load_source_scaler(
                checkpoint_path, scaler_path, model, device
            )
            history = pd.DataFrame()
            optimizer = scheduler = None
            train_windows = None
        else:
            scaler = _fit_target_scaler(train_records)
            train_windows = build_cmu_vidal_windows(
                transform_cmu_records(train_records, scaler),
                seq_len=cfg.seq_len,
                stride=cfg.stride,
                keep_incomplete=False,
            )
            history, optimizer, scheduler = _train(
                model, train_windows, cfg, device
            )

        validation_windows = build_cmu_vidal_windows(
            transform_cmu_records(validation_records, scaler),
            seq_len=cfg.seq_len,
            stride=cfg.stride,
            keep_incomplete=False,
        )
        test_windows = build_cmu_vidal_windows(
            transform_cmu_records(test_records, scaler),
            seq_len=cfg.seq_len,
            stride=cfg.stride,
            keep_incomplete=False,
        )
        validation_predictions, validation_metrics = evaluate_target_last(
            model,
            validation_windows,
            phase_by_source_row,
            cfg.batch_size,
            device,
        )
        test_predictions, test_metrics = evaluate_target_last(
            model, test_windows, phase_by_source_row, cfg.batch_size, device
        )

        cell_dir = run_root / cell_id
        cell_dir.mkdir()
        history.to_csv(cell_dir / "training_history.csv", index=False)
        validation_predictions.to_parquet(
            cell_dir / "validation_predictions.parquet", index=False
        )
        test_predictions.to_parquet(
            cell_dir / "test_predictions.parquet", index=False
        )
        (cell_dir / "scaler.json").write_text(
            json.dumps(
                {
                    **scaler.state_dict(),
                    "fit_scope": (
                        "panasonic_case1_full_train"
                        if cfg.setting == "zero_shot"
                        else f"{cell_id}_{cfg.setting}_train_missions_only"
                    ),
                    "feature_names": list(FEATURES),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if optimizer is not None and scheduler is not None:
            save_checkpoint(
                cell_dir / "checkpoint_epoch_515.pt",
                model,
                optimizer,
                cfg.epochs - 1,
                float(history.iloc[-1]["train_mse"]),
                asdict(cfg),
                scheduler=scheduler,
                normalizer=scaler,
                extra=source_metadata,
            )
        runtime = time.perf_counter() - started
        cell_row = {
            "setting": cfg.setting,
            "cell_id": cell_id,
            "condition": manifest.loc[
                manifest["cell_id"] == cell_id, "condition"
            ].iloc[0],
            "train_missions": len(train_ids),
            "validation_missions": len(validation_ids),
            "test_missions": len(test_ids),
            "train_fraction_of_all_missions": len(train_ids) / len(all_ids),
            "train_fraction_of_train_split": (
                len(train_ids)
                / int(
                    (manifest.loc[manifest["cell_id"] == cell_id, "split"] == "train").sum()
                )
            ),
            "train_windows": 0 if train_windows is None else len(train_windows.X),
            "validation_windows": len(validation_windows.X),
            "test_windows": len(test_windows.X),
            **test_metrics,
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
            "runtime_seconds": runtime,
        }
        cell_rows.append(cell_row)
        pd.DataFrame(cell_rows).to_csv(run_root / "cell_metrics.csv", index=False)
        print(
            f"{cfg.setting} {cell_id}: "
            f"MAE={test_metrics['overall_mae']:.6f}, "
            f"RMSE={test_metrics['overall_rmse']:.6f}, "
            f"runtime={runtime:.1f}s"
        )

    metrics = pd.DataFrame(cell_rows)
    zero_shot = (
        metrics if cfg.setting == "zero_shot" else _latest_zero_shot_metrics(output_root, cfg.random_seed)
    )
    if not zero_shot.empty:
        baseline = zero_shot.set_index("cell_id")
        metrics["gain_vs_zero_shot_mae"] = metrics.apply(
            lambda row: baseline.loc[row["cell_id"], "overall_mae"]
            - row["overall_mae"],
            axis=1,
        )
        metrics["gain_vs_zero_shot_rmse"] = metrics.apply(
            lambda row: baseline.loc[row["cell_id"], "overall_rmse"]
            - row["overall_rmse"],
            axis=1,
        )
    metrics.to_csv(run_root / "cell_metrics.csv", index=False)
    (run_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "benchmark": "vidal_style_panasonic_to_cmu_supervised_development",
                "setting": cfg.setting,
                "config": asdict(cfg),
                "cells": list(CELLS),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "source_checkpoint": str(checkpoint_path),
                "source_checkpoint_sha256": sha256_file(checkpoint_path),
                "source_scaler": str(scaler_path),
                "source_scaler_sha256": sha256_file(scaler_path),
                "git_commit": get_full_commit_hash(repo_root),
                "git_dirty": is_dirty(repo_root),
                "device": str(device),
                "runtime_seconds": time.perf_counter() - campaign_started,
                "target_labels_used": cfg.setting != "zero_shot",
                "pseudo_labels_used": False,
                "label_is_ground_truth": False,
                "formal_benchmark_eligible": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"RESULT_DIR={run_root}")
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True, choices=SETTINGS)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-scaler", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=515)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    run_setting(
        RunConfig(
            setting=args.setting,
            epochs=args.epochs,
            batch_size=args.batch_size,
        ),
        args.manifest,
        args.base_root,
        args.extension_root,
        args.output_root,
        args.source_checkpoint,
        args.source_scaler,
        REPO_ROOT,
        args.device,
    )


if __name__ == "__main__":
    main()
