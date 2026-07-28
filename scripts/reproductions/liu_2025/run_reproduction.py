#!/usr/bin/env python3
"""Manifest-driven Liu et al. (2025) DARM reproduction."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from battery_benchmark.data.mat_loader import load_mat_file
from battery_benchmark.labels.soc_calculator import compute_vidal_drive_soc
from battery_benchmark.preprocessing.normalization import MinMaxNormalizer
from battery_benchmark.reproductions.liu_2025.darm import (
    ADVERSARIAL_UPDATES,
    DARM,
    TARGET_BN_MODES,
    audit_target_bn_forward,
    target_bn_buffer_drift,
    target_bn_buffer_snapshot,
)
from battery_benchmark.reproductions.liu_2025.protocol import load_experiment_manifest
from battery_benchmark.reproductions.liu_2025.source_artifact import (
    load_source_artifact,
    save_source_artifact,
)
from battery_benchmark.reproductions.liu_2025.windows import build_liu_windows
from battery_benchmark.reproductions.vidal_2019.resampling import resample_vidal_record
from battery_benchmark.reproductions.vidal_2019.sequence_v2 import VidalV2Record
from battery_benchmark.utils.device import get_device
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty
from battery_benchmark.utils.hashing import sha256_file

FEATURE_ORDERS = {
    "legacy_vit": ("voltage", "current", "temperature"),
    "paper_ivt": ("current", "voltage", "temperature"),
}


def load_record(row) -> VidalV2Record:
    """Load one manifest row without creating forbidden target-adaptation SOC."""
    measurements = load_mat_file(row.path).measurements
    required = ["Time", "Voltage", "Current", "Battery_Temp_degC"]
    labelled = bool(row.label_available)
    if labelled:
        required.append("Ah")
    missing = set(required) - set(measurements)
    if missing:
        raise ValueError(f"{row.record_id} missing fields: {sorted(missing)}")

    size = min(len(measurements[name]) for name in required)
    frame = pd.DataFrame(
        {
            "timestamp": np.asarray(measurements["Time"][:size], dtype=float),
            "voltage": np.asarray(measurements["Voltage"][:size], dtype=float),
            "current": np.asarray(measurements["Current"][:size], dtype=float),
            "temperature": np.asarray(measurements["Battery_Temp_degC"][:size], dtype=float),
            **({"ah": np.asarray(measurements["Ah"][:size], dtype=float)} if labelled else {}),
        }
    ).dropna()
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    metadata = {
        "record_id": row.record_id,
        "profile": row.profile,
        "role": row.role,
        "label_usage": row.label_usage,
    }
    if labelled:
        ah = frame["ah"].to_numpy()
        denominator, soc = compute_vidal_drive_soc(ah)
        frame["soc"] = soc
        ah_delta = ah - ah[0]
        metadata.update(
            {
                "soc_denominator_ah": denominator,
                "delivered_ah": abs(float(ah_delta.min())),
                "soc_end_per_file_percent": float(soc[-1] * 100),
            }
        )
        for capacity in (2.75, 2.9):
            nominal_soc = 1.0 + ah_delta / capacity
            suffix = str(capacity).replace(".", "_")
            metadata[f"soc_end_{suffix}ah_percent"] = float(nominal_soc[-1] * 100)
            metadata[f"mae_per_file_vs_{suffix}ah_percent"] = float(
                np.mean(np.abs(soc - nominal_soc)) * 100
            )
    else:
        # resample_vidal_record expects a SOC column; zeros carry no label information.
        frame["soc"] = 0.0
    record = resample_vidal_record(
        VidalV2Record(frame, metadata),
        target_hz=1.0,
    )
    if not labelled:
        record.data["soc"] = np.nan
    return record


def evenly_limit(
    x: np.ndarray,
    y: np.ndarray | None,
    maximum: int | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if maximum is None or len(x) <= maximum:
        return x, y
    indices = np.linspace(0, len(x) - 1, maximum, dtype=int)
    return x[indices], None if y is None else y[indices]


def supervised_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=min(batch_size, len(x)),
        shuffle=shuffle,
    )


def unlabelled_loader(x: np.ndarray, batch_size: int) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=min(batch_size, len(x)),
        shuffle=True,
    )


def feature_ranges(
    x: np.ndarray,
    feature_order: tuple[str, str, str],
) -> dict[str, dict[str, float]]:
    return {
        name: {
            "min": float(x[..., index].min()),
            "max": float(x[..., index].max()),
            "outside_0_1_fraction": float(np.mean((x[..., index] < 0.0) | (x[..., index] > 1.0))),
        }
        for index, name in enumerate(feature_order)
    }


def _gradient_l2(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared = sum(
        float(gradient.detach().square().sum()) for gradient in gradients if gradient is not None
    )
    return squared**0.5


@torch.no_grad()
def target_feature_drift(
    model: DARM,
    initial_encoder: torch.nn.Module,
    data: DataLoader,
    device: torch.device,
) -> float:
    model.target_encoder.eval()
    squared, count = 0.0, 0
    for (x,) in data:
        x = x.to(device)
        current = model.encode_target(x)[0]
        initial = initial_encoder(x)[0]
        squared += float((current - initial).square().sum())
        count += current.numel()
    return (squared / count) ** 0.5 if count else 0.0


def _save_adaptation_state(
    path: Path,
    model: DARM,
    optimizers: dict[str, torch.optim.Optimizer],
    epoch: int,
    lowest_total_loss: float,
    stale_epochs: int,
    metrics: dict[str, object],
    *,
    checkpoint_kind: str,
    target_bn_mode: str,
    adversarial_update: str,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dicts": {
                name: optimizer.state_dict() for name, optimizer in optimizers.items()
            },
            "epoch": epoch,
            "lowest_total_loss": lowest_total_loss,
            "stale_epochs": stale_epochs,
            "metrics": metrics,
            "metadata": {
                "checkpoint_kind": checkpoint_kind,
                "target_bn_mode": target_bn_mode,
                "adversarial_update": adversarial_update,
                "selection_uses_target_labels": False,
                "diagnostic": adversarial_update == "eq12_joint",
            },
        },
        path,
    )


def _load_adaptation_state(
    path: Path,
    model: DARM,
    optimizers: dict[str, torch.optim.Optimizer],
    *,
    target_bn_mode: str,
    adversarial_update: str,
    device: torch.device | str,
) -> dict[str, object]:
    state = torch.load(path, map_location=device, weights_only=False)
    metadata = state.get("metadata", {})
    if metadata.get("target_bn_mode", "legacy") != target_bn_mode:
        raise ValueError("Checkpoint target_bn_mode does not match the requested mode")
    if metadata.get("adversarial_update", "grl_joint") != adversarial_update:
        raise ValueError("Checkpoint adversarial_update does not match the requested mode")
    model.load_state_dict(state["model_state_dict"])
    optimizer_states = state.get("optimizer_state_dicts")
    if optimizer_states is None and "optimizer_state_dict" in state:
        optimizer_states = {"joint": state["optimizer_state_dict"]}
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(optimizer_states[name])
    model.set_target_bn_mode(target_bn_mode)
    return state


def _save_training_state(
    path: Path,
    model: DARM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
    stale_epochs: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "stale_epochs": stale_epochs,
            "metrics": metrics,
        },
        path,
    )


def train_source(
    model: DARM,
    data: DataLoader,
    epochs: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
    output: Path,
    resume: bool,
) -> dict[str, float]:
    optimizer = torch.optim.Adam(model.source_parameters(), lr=learning_rate)
    last_path, best_path = output / "source_last.pt", output / "source_best.pt"
    start, best_loss, stale, last_metrics = 0, float("inf"), 0, {}
    if resume and last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start = int(state["epoch"]) + 1
        best_loss = float(state["best_loss"])
        stale = int(state["stale_epochs"])
        last_metrics = state["metrics"]

    progress = tqdm(range(start, epochs), desc="Source pre-training", initial=start, total=epochs)
    model.train()
    for epoch in progress:
        losses = []
        for x, y in data:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model.forward_source(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_loss = float(np.mean(losses))
        last_metrics = {"mse": epoch_loss, "epoch": epoch + 1}
        improved = epoch_loss < best_loss
        best_loss, stale = (epoch_loss, 0) if improved else (best_loss, stale + 1)
        _save_training_state(last_path, model, optimizer, epoch, best_loss, stale, last_metrics)
        if improved:
            _save_training_state(best_path, model, optimizer, epoch, best_loss, stale, last_metrics)
        if epoch + 1 in {1, 2, 5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500}:
            _save_training_state(
                output / f"source_epoch_{epoch + 1:03d}.pt",
                model,
                optimizer,
                epoch,
                best_loss,
                stale,
                last_metrics,
            )
        progress.set_postfix(mse=f"{epoch_loss:.6f}")
        if patience and stale >= patience:
            break

    if not best_path.is_file():
        raise RuntimeError("Source training produced no checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    return {**last_metrics, "best_mse": float(best["best_loss"])}


def adapt(
    model: DARM,
    source: DataLoader,
    target: DataLoader,
    epochs: int,
    learning_rate: float,
    adversarial_weight: float,
    alpha: float,
    beta: float,
    patience: int,
    device: torch.device,
    output: Path,
    resume: bool,
    target_bn_mode: str,
    adversarial_update: str,
    diagnostic_loaders: dict[str, DataLoader],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.set_target_bn_mode(target_bn_mode)
    if adversarial_update == "alternating_flipped":
        optimizers = {
            "discriminator": torch.optim.Adam(model.discriminator.parameters(), lr=learning_rate),
            "target": torch.optim.Adam(model.target_adaptation_parameters(), lr=learning_rate),
        }
    else:
        optimizers = {"joint": torch.optim.Adam(model.adaptation_parameters(), lr=learning_rate)}

    last_path = output / "adapt_last.pt"
    proxy_path = output / "adapt_lowest_total_loss_proxy.pt"
    start, lowest_total, stale, last_metrics = 0, float("inf"), 0, {}
    initial_encoder = deepcopy(model.target_encoder).to(device).eval()
    for parameter in initial_encoder.parameters():
        parameter.requires_grad_(False)
    initial_bn = target_bn_buffer_snapshot(model.target_encoder)

    initialization_path = output / "adapt_initialization.pt"
    if not resume or not initialization_path.is_file():
        _save_adaptation_state(
            initialization_path,
            model,
            optimizers,
            0,
            lowest_total,
            stale,
            {},
            checkpoint_kind="initialization_source_only",
            target_bn_mode=target_bn_mode,
            adversarial_update=adversarial_update,
        )
    if resume and last_path.is_file():
        state = _load_adaptation_state(
            last_path,
            model,
            optimizers,
            target_bn_mode=target_bn_mode,
            adversarial_update=adversarial_update,
            device=device,
        )
        start = int(state["epoch"])
        lowest_total = float(state.get("lowest_total_loss", state.get("best_loss", float("inf"))))
        stale = int(state["stale_epochs"])
        last_metrics = state["metrics"]

    progress = tqdm(range(start, epochs), desc="Target adaptation", initial=start, total=epochs)
    diagnostics_path = output / "adaptation_diagnostics.json"
    diagnostics: list[dict[str, object]] = (
        json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if resume and diagnostics_path.is_file()
        else []
    )
    model.train()
    for epoch in progress:
        batches = []
        first_batch_gradient_l2: dict[str, float] | None = None
        source_iterator = cycle(source)
        for (target_x,) in target:
            source_x, _ = next(source_iterator)
            source_x, target_x = source_x.to(device), target_x.to(device)
            target_parameters = [
                parameter
                for parameter in model.target_encoder.parameters()
                if parameter.requires_grad
            ]
            if adversarial_update == "alternating_flipped":
                discriminator_optimizer = optimizers["discriminator"]
                target_optimizer = optimizers["target"]
                discriminator_optimizer.zero_grad(set_to_none=True)
                discriminator_loss, domain_metrics = model.discriminator_step_loss(
                    source_x, target_x
                )
                discriminator_loss.backward()
                discriminator_optimizer.step()
                discriminator_optimizer.zero_grad(set_to_none=True)

                target_optimizer.zero_grad(set_to_none=True)
                loss, losses = model.alternating_target_loss(
                    source_x,
                    target_x,
                    adversarial_weight=adversarial_weight,
                    alpha=alpha,
                    beta=beta,
                )
                if first_batch_gradient_l2 is None:
                    first_batch_gradient_l2 = {
                        "adversarial": _gradient_l2(
                            adversarial_weight * losses["adversarial"], target_parameters
                        ),
                        "reconstruction": _gradient_l2(
                            alpha * losses["reconstruction"], target_parameters
                        ),
                        "mmd": _gradient_l2(beta * losses["mmd"], target_parameters),
                        "total": _gradient_l2(losses["total"], target_parameters),
                    }
                    target_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                target_optimizer.step()
            else:
                optimizer = optimizers["joint"]
                optimizer.zero_grad(set_to_none=True)
                loss, losses, domain_metrics = model.adaptation_loss(
                    source_x,
                    target_x,
                    adversarial_weight=adversarial_weight,
                    alpha=alpha,
                    beta=beta,
                    adversarial_update=adversarial_update,
                )
                if first_batch_gradient_l2 is None:
                    first_batch_gradient_l2 = {
                        "adversarial": _gradient_l2(
                            adversarial_weight * losses["adversarial"], target_parameters
                        ),
                        "reconstruction": _gradient_l2(
                            alpha * losses["reconstruction"], target_parameters
                        ),
                        "mmd": _gradient_l2(beta * losses["mmd"], target_parameters),
                        "total": _gradient_l2(losses["total"], target_parameters),
                    }
                    optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batches.append(
                {
                    **domain_metrics,
                    **{name: float(value.detach()) for name, value in losses.items()},
                }
            )
        last_metrics = {
            key: float(np.mean([batch[key] for batch in batches])) for key in batches[0]
        }
        last_metrics["epoch"] = epoch + 1
        last_metrics["gradient_l2"] = first_batch_gradient_l2 or {}
        target_test = {
            profile: evaluate(model, loader, device)
            for profile, loader in diagnostic_loaders.items()
        }
        last_metrics["target_test"] = target_test
        last_metrics["mean_profile_rmse_percent"] = float(
            np.mean([metrics["rmse_percent"] for metrics in target_test.values()])
        )
        last_metrics["target_feature_rms_drift"] = target_feature_drift(
            model, initial_encoder, target, device
        )
        last_metrics["target_bn_running_stat_drift"] = target_bn_buffer_drift(
            initial_bn, target_bn_buffer_snapshot(model.target_encoder)
        )
        epoch_loss = last_metrics["total"]
        improved = epoch_loss < lowest_total
        lowest_total, stale = (epoch_loss, 0) if improved else (lowest_total, stale + 1)
        _save_adaptation_state(
            last_path,
            model,
            optimizers,
            epoch + 1,
            lowest_total,
            stale,
            last_metrics,
            checkpoint_kind="last",
            target_bn_mode=target_bn_mode,
            adversarial_update=adversarial_update,
        )
        if improved:
            _save_adaptation_state(
                proxy_path,
                model,
                optimizers,
                epoch + 1,
                lowest_total,
                stale,
                last_metrics,
                checkpoint_kind="lowest_total_loss_proxy",
                target_bn_mode=target_bn_mode,
                adversarial_update=adversarial_update,
            )
        if epoch + 1 in {1, 2, 5, 10}:
            _save_adaptation_state(
                output / f"adapt_epoch_{epoch + 1:03d}.pt",
                model,
                optimizers,
                epoch + 1,
                lowest_total,
                stale,
                last_metrics,
                checkpoint_kind=f"epoch_{epoch + 1}",
                target_bn_mode=target_bn_mode,
                adversarial_update=adversarial_update,
            )
        diagnostics.append({"epoch": epoch + 1, "training": dict(last_metrics)})
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        model.train()
        progress.set_postfix(total=f"{epoch_loss:.6f}")
        if patience and stale >= patience:
            break

    if not last_path.is_file():
        raise RuntimeError("Target adaptation produced no checkpoint")
    oracle = min(
        diagnostics,
        key=lambda row: row["training"]["mean_profile_rmse_percent"],
    )
    return {
        **last_metrics,
        "lowest_total_loss_proxy": lowest_total,
        "selected_checkpoint": "last",
        "target_bn_mode": target_bn_mode,
        "adversarial_update": adversarial_update,
        "diagnostic": adversarial_update == "eq12_joint",
        "diagnostic_oracle": {
            "used_for_training_or_selection": False,
            "epoch": oracle["epoch"],
            "mean_profile_rmse_percent": oracle["training"]["mean_profile_rmse_percent"],
        },
    }, diagnostics


@torch.no_grad()
def evaluate(model: DARM, data: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    predictions, targets = [], []
    for x, y in data:
        predictions.append(model(x.to(device)).cpu())
        targets.append(y)
    prediction, target = torch.cat(predictions), torch.cat(targets)
    error = prediction - target
    return {
        "rmse_percent": float(torch.sqrt(torch.mean(error.square())) * 100),
        "mae_percent": float(torch.mean(torch.abs(error)) * 100),
        "n_windows": len(target),
    }


def run_target_bn_forward_audit(
    model: DARM,
    target: DataLoader,
    target_tests: dict[str, DataLoader],
    device: torch.device,
    output: Path,
) -> dict[str, object]:
    """Separate BN buffer drift from all gradient-based adaptation."""
    model.initialize_target()
    initial_state = deepcopy(model.state_dict())
    before = {profile: evaluate(model, loader, device) for profile, loader in target_tests.items()}
    after: dict[str, dict[str, dict[str, float]]] = {}
    drift: dict[str, dict[str, object]] = {}
    for mode in ("legacy", "frozen_stats"):
        model.load_state_dict(initial_state)
        model.initialize_target()
        drift[mode] = audit_target_bn_forward(
            model,
            (batch[0].to(device) for batch in target),
            mode,
        )
        after[mode] = {
            profile: evaluate(model, loader, device) for profile, loader in target_tests.items()
        }

    rows = [
        {
            "profile": profile,
            "rmse_before": metrics["rmse_percent"],
            "rmse_after_legacy_bn_forward": after["legacy"][profile]["rmse_percent"],
            "rmse_after_frozen_stats_forward": after["frozen_stats"][profile]["rmse_percent"],
            "delta_rmse": (after["legacy"][profile]["rmse_percent"] - metrics["rmse_percent"]),
        }
        for profile, metrics in before.items()
    ]
    result = {
        "audit": "target_bn_forward_no_grad",
        "backward_called": False,
        "optimizer_created_or_stepped": False,
        "profiles": rows,
        "bn_buffer_drift_summary": drift,
        "parameter_drift_summary": {
            mode: summary["parameter_drift"] for mode, summary in drift.items()
        },
    }
    (output / "target_bn_forward_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))
    for mode, summary in drift.items():
        print(
            mode,
            {key: value for key, value in summary["bn_buffer_drift"].items() if key != "layers"},
            summary["parameter_drift"],
        )
    return result


def _group_records(manifest: pd.DataFrame, role: str) -> dict[str, list[VidalV2Record]]:
    grouped: dict[str, list[VidalV2Record]] = {}
    for row in manifest.loc[manifest["role"] == role].itertuples(index=False):
        grouped.setdefault(row.profile, []).append(load_record(row))
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/liu_2025/panasonic_25c_to_0c.csv"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/Panasonic_18650PF"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/reproductions/liu_2025/25c_to_0c")
    )
    parser.add_argument("--source-epochs", type=int, default=500)
    parser.add_argument("--transfer-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--source-lr", type=float, default=1e-3)
    parser.add_argument("--transfer-lr", type=float, default=1e-4)
    parser.add_argument("--adversarial-weight", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--feature-order",
        choices=tuple(FEATURE_ORDERS),
        default="legacy_vit",
    )
    parser.add_argument(
        "--source-normalization",
        choices=("source_train", "source_domain"),
        default="source_train",
    )
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--source-artifact", type=Path)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--compare-source-checkpoints", action="store_true")
    parser.add_argument("--audit-preprocessing", action="store_true")
    parser.add_argument("--audit-target-bn-forward", action="store_true")
    parser.add_argument("--diagnose-adaptation", action="store_true")
    parser.add_argument(
        "--target-bn-mode",
        choices=TARGET_BN_MODES,
        default="legacy",
    )
    parser.add_argument(
        "--adversarial-update",
        choices=ADVERSARIAL_UPDATES,
        default="grl_joint",
    )
    parser.add_argument(
        "--target-normalization",
        choices=("source", "target_adapt", "target_domain"),
        default="source",
    )
    parser.add_argument("--clip-normalized-inputs", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.compare_source_checkpoints or args.audit_target_bn_forward) and not args.source_artifact:
        raise ValueError("Checkpoint comparison and audit modes require --source-artifact")
    if args.smoke:
        args.source_epochs = args.transfer_epochs = 1
        args.max_windows = args.max_windows or 512
        args.patience = 0
    args.output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    manifest = load_experiment_manifest(
        args.manifest, args.data_root, verify_hashes=args.verify_hashes
    )
    feature_order = FEATURE_ORDERS[args.feature_order]

    source_records = [
        load_record(row)
        for row in manifest.loc[manifest["role"] == "source_train"].itertuples(index=False)
    ]
    target_records = [
        load_record(row)
        for row in manifest.loc[manifest["role"] == "target_adapt"].itertuples(index=False)
    ]
    source_tests = _group_records(manifest, "source_test")
    target_tests = _group_records(manifest, "target_test")

    source_x, source_y = evenly_limit(
        *build_liu_windows(source_records, require_labels=True, feature_order=feature_order),
        args.max_windows,
    )
    target_x, target_y = evenly_limit(
        *build_liu_windows(target_records, require_labels=False, feature_order=feature_order),
        args.max_windows,
    )
    assert source_y is not None and target_y is None
    source_test_arrays = {
        profile: evenly_limit(
            *build_liu_windows(records, require_labels=True, feature_order=feature_order),
            args.max_windows,
        )
        for profile, records in source_tests.items()
    }
    target_test_arrays = {
        profile: evenly_limit(
            *build_liu_windows(records, require_labels=True, feature_order=feature_order),
            args.max_windows,
        )
        for profile, records in target_tests.items()
    }

    model = DARM().to(device)
    normalizer = MinMaxNormalizer()
    source_artifact_metadata: dict[str, object]
    if args.source_artifact:
        source_artifact_path = args.source_artifact
        source_artifact_metadata = load_source_artifact(
            source_artifact_path, model, normalizer, device=device
        )
        if source_artifact_metadata.get("source_manifest_sha256") != sha256_file(args.manifest):
            raise ValueError("Source artifact manifest hash does not match --manifest")
        if source_artifact_metadata.get("feature_order") != list(feature_order):
            raise ValueError("Source artifact feature order does not match --feature-order")
        if source_artifact_metadata.get("source_normalization", "source_train") != (
            args.source_normalization
        ):
            raise ValueError(
                "Source artifact normalization does not match --source-normalization"
            )
        source_training: dict[str, object] = {"reused": True}
    else:
        source_artifact_path = args.output / "source_artifact.pt"
        source_normalization_x = source_x
        if args.source_normalization == "source_domain":
            source_normalization_x = np.concatenate(
                [source_x, *(x for x, _ in source_test_arrays.values())]
            )
        normalizer.fit(source_normalization_x)

    target_normalizer = normalizer
    if args.target_normalization == "target_adapt":
        target_normalizer = MinMaxNormalizer().fit(target_x)
    elif args.target_normalization == "target_domain":
        target_normalizer = MinMaxNormalizer().fit(
            np.concatenate([target_x, *(x for x, _ in target_test_arrays.values())])
        )
    source_x = normalizer.transform(source_x).astype(np.float32)
    target_x = target_normalizer.transform(target_x).astype(np.float32)
    source_test_arrays = {
        profile: (normalizer.transform(x).astype(np.float32), y)
        for profile, (x, y) in source_test_arrays.items()
    }
    target_test_arrays = {
        profile: (target_normalizer.transform(x).astype(np.float32), y)
        for profile, (x, y) in target_test_arrays.items()
    }
    normalized_feature_ranges_before_clipping = None
    if args.clip_normalized_inputs:
        normalized_feature_ranges_before_clipping = {
            "source_train": feature_ranges(source_x, feature_order),
            "source_test": {
                profile: feature_ranges(x, feature_order)
                for profile, (x, _) in source_test_arrays.items()
            },
            "target_adapt": feature_ranges(target_x, feature_order),
            "target_test": {
                profile: feature_ranges(x, feature_order)
                for profile, (x, _) in target_test_arrays.items()
            },
        }
        source_x = np.clip(source_x, 0.0, 1.0)
        target_x = np.clip(target_x, 0.0, 1.0)
        source_test_arrays = {
            profile: (np.clip(x, 0.0, 1.0), y) for profile, (x, y) in source_test_arrays.items()
        }
        target_test_arrays = {
            profile: (np.clip(x, 0.0, 1.0), y) for profile, (x, y) in target_test_arrays.items()
        }
    normalized_feature_ranges = {
        "source_train": feature_ranges(source_x, feature_order),
        "source_test": {
            profile: feature_ranges(x, feature_order)
            for profile, (x, _) in source_test_arrays.items()
        },
        "target_adapt": feature_ranges(target_x, feature_order),
        "target_test": {
            profile: feature_ranges(x, feature_order)
            for profile, (x, _) in target_test_arrays.items()
        },
    }
    if args.audit_preprocessing:
        label_rows = []
        for role, groups in (
            ("source_train", {"Cycle_1-4": source_records}),
            ("source_test", source_tests),
            ("target_test", target_tests),
        ):
            for profile, records in groups.items():
                for record in records:
                    label_rows.append(
                        {
                            "role": role,
                            "profile": record.metadata.get("profile", profile),
                            **{
                                key: value
                                for key, value in record.metadata.items()
                                if key
                                in {
                                    "record_id",
                                    "soc_denominator_ah",
                                    "delivered_ah",
                                    "soc_end_per_file_percent",
                                    "soc_end_2_75ah_percent",
                                    "mae_per_file_vs_2_75ah_percent",
                                    "soc_end_2_9ah_percent",
                                    "mae_per_file_vs_2_9ah_percent",
                                }
                            },
                        }
                    )
        audit = {
            "paper": "Liu et al. 2025, doi:10.1109/TTE.2024.3483973",
            "manifest_sha256": sha256_file(args.manifest),
            "feature_order": list(feature_order),
            "source_artifact": str(source_artifact_path) if args.source_artifact else None,
            "source_normalization": args.source_normalization,
            "source_normalizer_state_dict": normalizer.state_dict(),
            "normalized_feature_ranges": normalized_feature_ranges,
            "label_formula_comparison": label_rows,
        }
        (args.output / "preprocessing_audit.json").write_text(
            json.dumps(audit, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(audit, indent=2))
        return

    source_loader = supervised_loader(source_x, source_y, args.batch_size, True)
    target_loader = unlabelled_loader(target_x, args.batch_size)
    source_test_loaders = {
        profile: supervised_loader(x, y, args.batch_size, False)
        for profile, (x, y) in source_test_arrays.items()
    }
    target_test_loaders = {
        profile: supervised_loader(x, y, args.batch_size, False)
        for profile, (x, y) in target_test_arrays.items()
    }

    if not args.source_artifact:
        source_training = train_source(
            model,
            source_loader,
            args.source_epochs,
            args.source_lr,
            args.patience,
            device,
            args.output,
            args.resume,
        )
        source_artifact_metadata = {
            "paper": "Liu et al. 2025, doi:10.1109/TTE.2024.3483973",
            "source_domain": "panasonic_25c",
            "source_manifest": str(args.manifest),
            "source_manifest_sha256": sha256_file(args.manifest),
            "feature_order": list(feature_order),
            "sampling_hz": 1.0,
            "window_length": 40,
            "window_stride": 1,
            "source_epochs": args.source_epochs,
            "source_learning_rate": args.source_lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "source_windows": len(source_x),
            "source_normalization": args.source_normalization,
            "git_commit": get_full_commit_hash(Path.cwd()),
            "git_dirty": is_dirty(Path.cwd()),
        }
        save_source_artifact(source_artifact_path, model, normalizer, source_artifact_metadata)
        model.initialize_target()
    source_artifact_sha256 = sha256_file(source_artifact_path)
    (args.output / "source_artifact.json").write_text(
        json.dumps(
            {
                **source_artifact_metadata,
                "artifact": str(source_artifact_path),
                "artifact_sha256": source_artifact_sha256,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    source_test = {
        profile: evaluate(model, test_loader, device)
        for profile, test_loader in source_test_loaders.items()
    }
    target_source_only = {
        profile: evaluate(model, test_loader, device)
        for profile, test_loader in target_test_loaders.items()
    }
    if args.source_only:
        milestone_diagnostics = []
        checkpoint_paths = sorted(args.output.glob("source_epoch_*.pt"))
        checkpoint_paths.extend(
            path
            for path in (args.output / "source_best.pt", args.output / "source_last.pt")
            if path.is_file()
        )
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.initialize_target()
            checkpoint_source_test = {
                profile: evaluate(model, loader, device)
                for profile, loader in source_test_loaders.items()
            }
            milestone_diagnostics.append(
                {
                    "checkpoint": checkpoint_path.name,
                    "epoch": int(checkpoint["epoch"]) + 1,
                    "training_mse": float(checkpoint["metrics"]["mse"]),
                    "mean_profile_rmse_percent": float(
                        np.mean(
                            [
                                metrics["rmse_percent"]
                                for metrics in checkpoint_source_test.values()
                            ]
                        )
                    ),
                    "source_test": checkpoint_source_test,
                }
            )
        source_only_result = {
            "paper": "Liu et al. 2025, doi:10.1109/TTE.2024.3483973",
            "protocol": "Panasonic 25C source estimator",
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "source_artifact": str(source_artifact_path),
            "source_artifact_sha256": source_artifact_sha256,
            "source_training": source_training,
            "source_test": source_test,
            "target_source_only": target_source_only,
            "feature_order": list(feature_order),
            "window_length": 40,
            "source_windows": len(source_x),
            "source_normalization": args.source_normalization,
            "source_milestone_diagnostics": {
                "diagnostic_only": True,
                "used_for_training_or_checkpoint_selection": False,
                "checkpoints": milestone_diagnostics,
            },
        }
        (args.output / "source_only_results.json").write_text(
            json.dumps(source_only_result, indent=2), encoding="utf-8"
        )
        print("Source test")
        print(pd.DataFrame(source_test).T.to_string())
        print("Target source-only")
        print(pd.DataFrame(target_source_only).T.to_string())
        return
    if args.compare_source_checkpoints:
        comparison = {
            "source_artifact": {
                "path": str(source_artifact_path),
                "sha256": source_artifact_sha256,
                "source_test": source_test,
                "target_source_only": target_source_only,
            }
        }
        for filename in ("source_best.pt", "source_last.pt"):
            checkpoint_path = source_artifact_path.parent / filename
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Missing source checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.initialize_target()
            comparison[checkpoint_path.stem] = {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "training": checkpoint["metrics"],
                "source_test": {
                    profile: evaluate(model, loader, device)
                    for profile, loader in source_test_loaders.items()
                },
                "target_source_only": {
                    profile: evaluate(model, loader, device)
                    for profile, loader in target_test_loaders.items()
                },
            }
        comparison_result = {
            "paper": "Liu et al. 2025, doi:10.1109/TTE.2024.3483973",
            "manifest_sha256": sha256_file(args.manifest),
            "feature_order": list(feature_order),
            "source_artifact": str(source_artifact_path),
            "checkpoints": comparison,
        }
        (args.output / "source_checkpoint_comparison.json").write_text(
            json.dumps(comparison_result, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(comparison_result, indent=2))
        return
    if args.audit_target_bn_forward:
        run_target_bn_forward_audit(
            model,
            target_loader,
            target_test_loaders,
            device,
            args.output,
        )
        return
    target_training, adaptation_diagnostics = adapt(
        model,
        source_loader,
        target_loader,
        args.transfer_epochs,
        args.transfer_lr,
        args.adversarial_weight,
        args.alpha,
        args.beta,
        args.patience,
        device,
        args.output,
        args.resume,
        args.target_bn_mode,
        args.adversarial_update,
        target_test_loaders,
    )
    target_darm = {
        profile: evaluate(model, test_loader, device)
        for profile, test_loader in target_test_loaders.items()
    }

    result = {
        "paper": "Liu et al. 2025, doi:10.1109/TTE.2024.3483973",
        "protocol": "Panasonic 25C -> 0C",
        "device": str(device),
        "run_mode": "smoke" if args.smoke else "formal",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "source_artifact": str(source_artifact_path),
        "source_artifact_sha256": source_artifact_sha256,
        "window_length": 40,
        "feature_order": list(feature_order),
        "source_normalization": args.source_normalization,
        "source_windows": len(source_x),
        "target_adaptation_windows": len(target_x),
        "source_training": source_training,
        "target_training": target_training,
        "target_bn_mode": args.target_bn_mode,
        "adversarial_update": args.adversarial_update,
        "diagnostic": args.adversarial_update == "eq12_joint",
        "adaptation_loss_weights": {
            "adversarial": args.adversarial_weight,
            "reconstruction": args.alpha,
            "mmd": args.beta,
        },
        "adaptation_diagnostics": adaptation_diagnostics,
        "clip_normalized_inputs": args.clip_normalized_inputs,
        "target_normalization": args.target_normalization,
        "target_normalizer_state_dict": target_normalizer.state_dict(),
        "normalized_feature_ranges_before_clipping": normalized_feature_ranges_before_clipping,
        "normalized_feature_ranges": normalized_feature_ranges,
        "source_test": source_test,
        "target_source_only": target_source_only,
        "target_darm": target_darm,
        "known_reconstruction": {
            "soc_label": "per-file cycler Ah normalization; paper formula unspecified",
            "normalization": (
                "source scaler: source Cycle 1-4; target scaler: target Cycle 1"
                if args.target_normalization == "target_adapt"
                else (
                    "source scaler: source Cycle 1-4; target scaler: all target-domain "
                    "input covariates (transductive diagnostic)"
                    if args.target_normalization == "target_domain"
                    else "fit on source Cycle 1-4 only"
                )
            ),
            "diagnostic_target_labels": (
                "post-epoch evaluation only; never used for optimization, early stopping, "
                "or checkpoint selection"
            ),
            "window_stride": 1,
            "mk_mmd_bandwidth": "median x [0.5, 1, 2, 4]; paper unspecified",
            "target_batch_norm": args.target_bn_mode,
            "adversarial_gradient": {
                "grl_joint": "gradient reversal with joint discriminator/target update",
                "alternating_flipped": (
                    "detached discriminator step, then flipped target-as-source step"
                ),
                "eq12_joint": (
                    "diagnostic Eq.12 literal joint minimization without gradient reversal"
                ),
            }[args.adversarial_update],
            "checkpoint_selection": (
                "last is the formal output; lowest total loss is proxy-only; "
                "target-label oracle is post-hoc diagnostic only"
            ),
            "early_stopping_monitor": ("training total-loss proxy; paper only reports patience=20"),
        },
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), args.output / "model_final.pt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
