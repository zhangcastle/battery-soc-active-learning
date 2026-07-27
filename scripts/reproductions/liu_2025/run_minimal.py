#!/usr/bin/env python3
"""Minimal real-data reproduction of Liu et al. (2025) DARM.

Default protocol: Panasonic 18650PF, 25 C source -> 0 C target.
The full paper settings are the defaults; use --smoke for a quick executable check.
"""

from __future__ import annotations

import argparse
import json
import random
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from battery_benchmark.data.mat_loader import load_mat_file
from battery_benchmark.labels.soc_calculator import compute_vidal_drive_soc
from battery_benchmark.preprocessing.normalization import MinMaxNormalizer
from battery_benchmark.reproductions.liu_2025.darm import DARM
from battery_benchmark.reproductions.vidal_2019.resampling import resample_vidal_record
from battery_benchmark.reproductions.vidal_2019.sequence_v2 import VidalV2Record
from battery_benchmark.utils.device import get_device

PROFILES = ("LA92", "UDDS", "US06", "HWFET", "NN")


def _drive_dir(root: Path, temperature: int) -> Path:
    base = root / f"{temperature}degC"
    matches = [
        path for path in base.iterdir() if path.is_dir() and path.name.lower() == "drive cycles"
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one drive-cycle directory under {base}")
    return matches[0]


def _profile_files(root: Path, temperature: int, profile: str) -> list[Path]:
    directory = _drive_dir(root, temperature)
    files = sorted(directory.glob(f"*_{profile}_Pan18650PF.mat"))
    if profile == "HWFET" and not files:
        files = sorted(directory.glob("*_HWFT?_Pan18650PF.mat"))
    if not files:
        raise FileNotFoundError(f"No {profile} file at {temperature} C in {directory}")
    return files


def _cycle_file(root: Path, temperature: int, number: int) -> Path:
    files = sorted(_drive_dir(root, temperature).glob(f"*_Cycle_{number}_Pan18650PF.mat"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected one Cycle {number} file at {temperature} C, found {len(files)}"
        )
    return files[0]


def load_record(path: Path) -> VidalV2Record:
    measurements = load_mat_file(path).measurements
    required = ("Time", "Voltage", "Current", "Battery_Temp_degC", "Ah")
    missing = set(required) - set(measurements)
    if missing:
        raise ValueError(f"{path.name} missing fields: {sorted(missing)}")

    size = min(len(measurements[name]) for name in required)
    frame = pd.DataFrame(
        {
            "timestamp": np.asarray(measurements["Time"][:size], dtype=float),
            "voltage": np.asarray(measurements["Voltage"][:size], dtype=float),
            "current": np.asarray(measurements["Current"][:size], dtype=float),
            "temperature": np.asarray(measurements["Battery_Temp_degC"][:size], dtype=float),
            "ah": np.asarray(measurements["Ah"][:size], dtype=float),
        }
    ).dropna()
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    _, frame["soc"] = compute_vidal_drive_soc(frame["ah"].to_numpy())
    return resample_vidal_record(
        VidalV2Record(frame, {"source_filename": path.name}), target_hz=1.0
    )


def windows(records: list[VidalV2Record], length: int = 40) -> tuple[np.ndarray, np.ndarray]:
    feature_windows, targets = [], []
    for record in records:
        values = record.data[["voltage", "current", "temperature"]].to_numpy(np.float32)
        soc = record.data["soc"].to_numpy(np.float32)
        for start in range(len(values) - length + 1):
            feature_windows.append(values[start : start + length])
            targets.append(soc[start + length - 1])
    if not feature_windows:
        raise ValueError("No complete windows were produced")
    return np.stack(feature_windows), np.asarray(targets, dtype=np.float32)


def evenly_limit(
    x: np.ndarray, y: np.ndarray, maximum: int | None
) -> tuple[np.ndarray, np.ndarray]:
    if maximum is None or len(x) <= maximum:
        return x, y
    indices = np.linspace(0, len(x) - 1, maximum, dtype=int)
    return x[indices], y[indices]


def loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=min(batch_size, len(x)),
        shuffle=shuffle,
    )


def train_source(
    model: DARM,
    data: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> list[float]:
    optimizer = torch.optim.Adam(model.source_parameters(), lr=learning_rate)
    history = []
    model.train()
    for _ in range(epochs):
        losses = []
        for x, y in data:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model.forward_source(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return history


def adapt(
    model: DARM,
    source: DataLoader,
    target: DataLoader,
    epochs: int,
    learning_rate: float,
    alpha: float,
    beta: float,
    device: torch.device,
) -> list[dict[str, float]]:
    model.initialize_target()
    optimizer = torch.optim.Adam(model.adaptation_parameters(), lr=learning_rate)
    history = []
    model.train()
    for _ in range(epochs):
        batches = []
        source_iterator = cycle(source)
        for target_x, _ in target:
            source_x, _ = next(source_iterator)
            source_x, target_x = source_x.to(device), target_x.to(device)
            optimizer.zero_grad()
            loss, parts = model.adaptation_loss(source_x, target_x, alpha=alpha, beta=beta)
            loss.backward()
            optimizer.step()
            batches.append(parts)
        history.append(
            {key: float(np.mean([batch[key] for batch in batches])) for key in batches[0]}
        )
    return history


@torch.no_grad()
def evaluate(model: DARM, data: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    predictions, targets = [], []
    for x, y in data:
        predictions.append(model(x.to(device)).cpu())
        targets.append(y)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    error = prediction - target
    return {
        "rmse_percent": float(torch.sqrt(torch.mean(error.square())) * 100),
        "mae_percent": float(torch.mean(torch.abs(error)) * 100),
        "n_windows": len(target),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/Panasonic_18650PF"))
    parser.add_argument("--target-temperature", type=int, default=0)
    parser.add_argument("--source-epochs", type=int, default=500)
    parser.add_argument("--transfer-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--source-lr", type=float, default=1e-3)
    parser.add_argument("--transfer-lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/reproductions/liu_2025"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.source_epochs = args.transfer_epochs = 1
        args.max_windows = args.max_windows or 512

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device("auto")

    source_records = [
        load_record(_cycle_file(args.data_root, 25, number)) for number in range(1, 5)
    ]
    target_record = load_record(_cycle_file(args.data_root, args.target_temperature, 1))
    test_records = {
        profile: [
            load_record(path)
            for path in _profile_files(args.data_root, args.target_temperature, profile)
        ]
        for profile in PROFILES
    }

    source_x, source_y = evenly_limit(*windows(source_records), args.max_windows)
    target_x, target_y = evenly_limit(*windows([target_record]), args.max_windows)
    tests = {
        profile: evenly_limit(*windows(records), args.max_windows)
        for profile, records in test_records.items()
    }

    normalizer = MinMaxNormalizer()
    source_x = normalizer.fit_transform(source_x).astype(np.float32)
    target_x = normalizer.transform(target_x).astype(np.float32)
    tests = {
        profile: (normalizer.transform(x).astype(np.float32), y)
        for profile, (x, y) in tests.items()
    }

    source_loader = loader(source_x, source_y, args.batch_size, True)
    target_loader = loader(target_x, target_y, args.batch_size, True)
    test_loaders = {
        profile: loader(x, y, args.batch_size, False) for profile, (x, y) in tests.items()
    }

    model = DARM().to(device)
    source_history = train_source(model, source_loader, args.source_epochs, args.source_lr, device)
    model.initialize_target()
    source_only = {
        profile: evaluate(model, test_loader, device)
        for profile, test_loader in test_loaders.items()
    }
    transfer_history = adapt(
        model,
        source_loader,
        target_loader,
        args.transfer_epochs,
        args.transfer_lr,
        args.alpha,
        args.beta,
        device,
    )
    adapted = {
        profile: evaluate(model, test_loader, device)
        for profile, test_loader in test_loaders.items()
    }

    result = {
        "paper": "Liu et al. 2025, doi:10.1109/TTE.2024.3483973",
        "protocol": f"Panasonic 25C -> {args.target_temperature}C",
        "device": str(device),
        "paper_defaults": not args.smoke,
        "source_epochs": args.source_epochs,
        "transfer_epochs": args.transfer_epochs,
        "source_final_mse": source_history[-1],
        "transfer_final_losses": transfer_history[-1],
        "source_only": source_only,
        "darm": adapted,
        "known_reconstruction": {
            "soc_label": "per-file Ah normalization; paper only says capacity/SOC",
            "normalization": "fit on source Cycle 1-4 only to prevent target-test leakage",
            "mk_mmd_bandwidth": "median x [0.5, 1, 2, 4]; paper unspecified",
            "adversarial_gradient": "gradient reversal; paper pseudocode omits sign detail",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"25C_to_{args.target_temperature}C.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), args.output / f"25C_to_{args.target_temperature}C.pt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
