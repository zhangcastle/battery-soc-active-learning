"""End-to-end Vidal Case 1 training and evaluation pipeline."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from battery_benchmark.data.vidal_case1_windows import (
    FEATURES, Case1WindowDataset, Case1Windows, build_case1_windows,
)
from battery_benchmark.data.vidal_sequence_v2_dataset import (
    VidalSequenceV2Dataset, VidalV2Record, validate_case1_roots,
)
from battery_benchmark.evaluation.metrics import compute_metrics
from battery_benchmark.models.vidal_lstm import VidalLSTM
from battery_benchmark.preprocessing.normalization import MinMaxNormalizer
from battery_benchmark.training.checkpointing import load_checkpoint, save_checkpoint
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty


DATASETS = ("panasonic_18650pf", "lg_hg2", "samsung_30t")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_dataframe(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


@dataclass
class Case1Config:
    dataset: str
    run_mode: str = "smoke"
    random_seed: int = 42
    seq_len: int = 50
    stride: int = 50
    epochs: int = 2
    batch_size: int = 512
    learning_rate: float = 0.01
    scheduler_step_size: int = 100
    scheduler_gamma: float = 0.5
    shuffle: bool = True
    num_workers: int = 0
    max_train_windows: int | None = None
    allowed_statuses: tuple[str, ...] = ("PASS", "WARNING")
    development_only: bool = True
    verify_hashes: bool = False

    def validate(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if self.run_mode not in {"smoke", "formal"}:
            raise ValueError("run_mode must be smoke or formal")
        if self.stride != self.seq_len:
            raise ValueError("fixed_non_overlapping Case 1 requires stride == seq_len")
        if self.run_mode == "formal":
            if self.development_only or self.epochs != 515 or self.max_train_windows is not None:
                raise ValueError("Formal runs require 515 epochs, all windows, development_only=false")
        elif not self.development_only:
            raise ValueError("Smoke runs must be development_only")


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool() & torch.isfinite(target)
    if not valid.any():
        raise ValueError("Batch has no valid targets")
    return torch.mean((pred[valid] - target[valid]) ** 2)


def fit_train_normalizer(records: list[VidalV2Record], dataset: str, repo_root: Path) -> tuple[MinMaxNormalizer, dict[str, Any]]:
    values = np.concatenate([r.data[list(FEATURES)].to_numpy(dtype=np.float32) for r in records])
    normalizer = MinMaxNormalizer().fit(values)
    state = normalizer.state_dict()
    metadata = {
        "dataset": dataset, "feature_names": list(FEATURES),
        "feature_min": state["min"], "feature_max": state["max"],
        "feature_range": list(state["feature_range"]), "fit_split": "full_train",
        "n_fit_samples": len(values), "git_commit": get_full_commit_hash(repo_root),
        "git_dirty": is_dirty(repo_root), "timestamp": utc_now(),
        "normalization_status": "reconstruction_assumption",
    }
    return normalizer, metadata


def transform_records(records: list[VidalV2Record], normalizer: MinMaxNormalizer) -> list[VidalV2Record]:
    transformed = []
    for record in records:
        frame = record.data.copy()
        frame.loc[:, list(FEATURES)] = normalizer.transform(
            frame[list(FEATURES)].to_numpy(dtype=np.float32)
        )
        transformed.append(VidalV2Record(frame, record.metadata))
    return transformed


def cache_case1_inputs(source_root: Path, cache_root: Path, dataset: str) -> Path:
    """Copy full_train and test to local cache and verify counts/sizes."""
    target_dataset = cache_root / dataset
    for split in ("full_train", "test"):
        source = source_root / dataset / split
        target = target_dataset / split
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        source_files = sorted(p for p in source.rglob("*") if p.is_file())
        target_files = sorted(p for p in target.rglob("*") if p.is_file())
        if len(source_files) != len(target_files):
            raise RuntimeError(f"Cache file count mismatch for {dataset}/{split}")
        if sum(p.stat().st_size for p in source_files) != sum(p.stat().st_size for p in target_files):
            raise RuntimeError(f"Cache size mismatch for {dataset}/{split}")
    return cache_root


def _run_id(repo_root: Path) -> str:
    commit = get_full_commit_hash(repo_root)
    short = commit[:8] if commit != "unknown" else "unknown"
    dirty = "_dirty" if is_dirty(repo_root) else ""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_{short}{dirty}"


def resolve_run_dir(output_root: Path, cfg: Case1Config, repo_root: Path) -> Path:
    base = output_root / ("_smoke" if cfg.run_mode == "smoke" else cfg.dataset)
    run_dir = base / cfg.dataset / f"seed_{cfg.random_seed}" / _run_id(repo_root) if cfg.run_mode == "smoke" else base / f"seed_{cfg.random_seed}" / _run_id(repo_root)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {run_dir}")
    return run_dir


def _save_checkpoint(path: Path, model, optimizer, scheduler, epoch, scaler, config, extra):
    save_checkpoint(path, model, optimizer, epoch, extra["train_mse"], config,
                    scheduler=scheduler, normalizer=scaler, extra=extra)


def train_model(model, loader, cfg: Case1Config, device: torch.device, run_dir: Path,
                scaler: MinMaxNormalizer, checkpoint_extra: dict[str, Any],
                resume_path: Path | None = None):
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.scheduler_step_size, cfg.scheduler_gamma)
    history: list[dict[str, Any]] = []
    start_epoch = 0
    model.to(device)
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        history_path = run_dir / "training_history.csv"
        if history_path.is_file():
            history = pd.read_csv(history_path).to_dict("records")
    for epoch in range(start_epoch, cfg.epochs):
        started = time.perf_counter(); model.train(); loss_sum = 0.0; valid_sum = 0
        for x, y, mask, _ in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(x), y, mask)
            loss.backward(); optimizer.step()
            n_valid = int(mask.sum().item()); loss_sum += float(loss.item()) * n_valid; valid_sum += n_valid
        train_mse = loss_sum / max(valid_sum, 1)
        history.append({"epoch": epoch + 1, "train_mse": train_mse,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "elapsed_time_s": time.perf_counter() - started,
                        "n_valid_samples": valid_sum})
        scheduler.step()
        extra = {**checkpoint_extra, "train_mse": train_mse, "random_seed": cfg.random_seed,
                 "device": str(device)}
        _save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, scheduler,
                         epoch, scaler, asdict(cfg), extra)
    if cfg.run_mode == "formal":
        shutil.copy2(run_dir / "checkpoints" / "last.pt", run_dir / "checkpoints" / "epoch_515.pt")
    return pd.DataFrame(history), optimizer, scheduler


@torch.no_grad()
def evaluate_model(model, windows: Case1Windows, raw_records: list[VidalV2Record],
                   cfg: Case1Config, device: torch.device) -> tuple[dict[str, float], pd.DataFrame]:
    dataset = Case1WindowDataset(windows)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
    model.eval(); rows = []
    raw_map = {str(r.metadata["source_filename"]): r.data for r in raw_records}
    for x, y, mask, indices in loader:
        pred = model(x.to(device)).cpu().numpy()
        y_np, mask_np = y.numpy(), mask.numpy()
        for batch_i, window_index in enumerate(indices.tolist()):
            meta = windows.metadata[window_index]
            valid = int(mask_np[batch_i].sum())
            raw = raw_map[str(meta["source_filename"])].iloc[
                int(meta["window_start"]):int(meta["window_start"]) + valid
            ]
            for j in range(valid):
                reference = float(y_np[batch_i, j]); prediction = float(pred[batch_i, j])
                rows.append({
                    "dataset": cfg.dataset, "seed": cfg.random_seed, "protocol_split": "test",
                    "sequence_order": meta["sequence_order"],
                    "sequence_segment_id": meta["sequence_segment_id"],
                    "source_filename": meta["source_filename"], "file_type": meta["file_type"],
                    "paper_profile": meta["paper_profile"],
                    "sample_index_in_record": int(meta["window_start"]) + j,
                    "timestamp": float(raw.iloc[j]["timestamp"]),
                    "voltage": float(raw.iloc[j]["voltage"]),
                    "current": float(raw.iloc[j]["current"]),
                    "temperature": float(raw.iloc[j]["temperature"]),
                    "soc_reference": reference, "soc_predicted": prediction,
                    "signed_error": prediction - reference,
                    "absolute_error": abs(prediction - reference),
                    "label_qc_status": meta["label_qc_status"],
                    "warning_reasons": meta["warning_reasons"],
                })
    predictions = pd.DataFrame(rows).sort_values(
        ["sequence_order", "sample_index_in_record"], kind="stable"
    ).reset_index(drop=True)
    metric = compute_metrics(predictions["soc_reference"].to_numpy(), predictions["soc_predicted"].to_numpy())
    result = {
        "mae_fraction": metric["mae"], "mae_percent": metric["mae"] * 100,
        "rmse_fraction": metric["rmse"], "rmse_percent": metric["rmse"] * 100,
        "max_error_fraction": metric["max_error"], "max_error_percent": metric["max_error"] * 100,
        "bias_fraction": metric["bias"], "bias_percent": metric["bias"] * 100,
        "r2": metric["r2"],
    }
    return result, predictions


def metrics_by_profile(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["paper_profile", "file_type", "source_filename"], sort=False):
        m = compute_metrics(group["soc_reference"].to_numpy(), group["soc_predicted"].to_numpy())
        rows.append({"paper_profile": keys[0], "file_type": keys[1], "source_filename": keys[2],
                     "n_samples": len(group), "mae": m["mae"], "rmse": m["rmse"],
                     "max_error": m["max_error"], "bias": m["bias"]})
    return pd.DataFrame(rows)


def plot_prediction_and_error(predictions: pd.DataFrame, dataset: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    x = np.arange(len(predictions))
    axes[0].plot(x, predictions["soc_reference"], label="Reference SOC", lw=1.1)
    axes[0].plot(x, predictions["soc_predicted"], label="Predicted SOC", lw=1.0)
    axes[0].set_ylabel("SOC"); axes[0].set_ylim(-0.02, 1.02); axes[0].legend(loc="best")
    axes[0].set_title(f"{dataset} Vidal Case 1 test prediction")
    axes[1].plot(x, predictions["signed_error"], lw=0.8, color="tab:red")
    axes[1].axhline(0, color="black", lw=0.7); axes[1].set_ylabel("Predicted - reference")
    axes[1].set_xlabel("Test sample order")
    boundaries = predictions.groupby("sequence_order", sort=False).head(1)
    for _, row in boundaries.iterrows():
        pos = int(row.name)
        for ax in axes: ax.axvline(pos, color="0.7", lw=0.6, ls="--")
        axes[0].text(pos, 1.01, str(row["paper_profile"]), rotation=90, va="bottom", fontsize=7)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def plot_training_loss(history: pd.DataFrame, dataset: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(history["epoch"], history["train_mse"], lw=1.2)
    ax.set(title=f"{dataset} Vidal Case 1 training loss", xlabel="Epoch", ylabel="Training MSE")
    for epoch in (100, 200, 300, 400, 500):
        if epoch <= int(history["epoch"].max()): ax.axvline(epoch, color="0.75", ls="--", lw=0.7)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def run_case1(cfg: Case1Config, input_root: Path, output_root: Path, repo_root: Path,
              device_name: str = "auto", resume_path: Path | None = None) -> dict[str, Any]:
    cfg.validate()
    if cfg.run_mode == "formal" and is_dirty(repo_root):
        raise RuntimeError("Formal run requires a clean Git working tree")
    checks = validate_case1_roots(input_root)
    if (checks["status"] != "PASS").any():
        raise FileNotFoundError("Vidal v2 input preflight failed")
    train = VidalSequenceV2Dataset(input_root, cfg.dataset, "full_train", cfg.allowed_statuses, cfg.verify_hashes)
    test = VidalSequenceV2Dataset(input_root, cfg.dataset, "test", cfg.allowed_statuses, cfg.verify_hashes)
    scaler, scaler_meta = fit_train_normalizer(train.records, cfg.dataset, repo_root)
    train_windows = build_case1_windows(transform_records(train.records, scaler), cfg.seq_len, cfg.stride, False)
    test_windows = build_case1_windows(transform_records(test.records, scaler), cfg.seq_len, cfg.stride, True)
    train_dataset = Case1WindowDataset(train_windows)
    if cfg.max_train_windows is not None:
        train_dataset = Subset(train_dataset, range(min(cfg.max_train_windows, len(train_dataset))))
    generator = torch.Generator().manual_seed(cfg.random_seed)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else
                          "cpu" if device_name == "auto" else device_name)
    loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle,
                        generator=generator, num_workers=cfg.num_workers,
                        pin_memory=device.type == "cuda")
    torch.manual_seed(cfg.random_seed); np.random.seed(cfg.random_seed)
    model = VidalLSTM()
    run_dir = resume_path.resolve().parents[1] if resume_path else resolve_run_dir(output_root, cfg, repo_root)
    for name in ("checkpoints", "figures", "predictions", "logs"):
        (run_dir / name).mkdir(parents=True, exist_ok=bool(resume_path))
    inventory = pd.concat([train.inventory, test.inventory], ignore_index=True)
    inventory_hash = hash_dataframe(inventory)
    history, _, _ = train_model(model, loader, cfg, device, run_dir, scaler, {
        "git_commit": get_full_commit_hash(repo_root), "git_dirty": is_dirty(repo_root),
        "dataset_inventory_hash": inventory_hash,
    }, resume_path=resume_path)
    metrics, predictions = evaluate_model(model, test_windows, test.records, cfg, device)
    profile = metrics_by_profile(predictions)
    inventory.to_csv(run_dir / "dataset_inventory.csv", index=False)
    history.to_csv(run_dir / "training_history.csv", index=False)
    predictions.to_csv(run_dir / "predictions" / "test_predictions.csv", index=False)
    profile.to_csv(run_dir / "metrics_by_profile.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "scaler.json").write_text(json.dumps(scaler_meta, indent=2), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
    metadata = {
        "dataset": cfg.dataset, "run_mode": cfg.run_mode, "development_only": cfg.development_only,
        "git_commit": get_full_commit_hash(repo_root), "git_dirty": is_dirty(repo_root),
        "device": str(device), "created_at": utc_now(), "input_root": str(input_root),
        "inventory_hash": inventory_hash, "train_records": len(train.records), "test_records": len(test.records),
        "train_windows": len(train_windows.X), "test_windows": len(test_windows.X),
        "test_valid_samples": int(test_windows.valid_mask.sum()),
        "dropped_train_remainder": train_windows.dropped_samples,
        "trainable_parameters": model.count_trainable_parameters(),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "sequence_construction_status": "reconstruction_assumption",
        "shuffle_status": "reconstruction_assumption",
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_prediction_and_error(predictions, cfg.dataset,
        run_dir / "figures" / f"{cfg.dataset}_case1_test_soc_prediction_and_error.png")
    plot_training_loss(history, cfg.dataset,
        run_dir / "figures" / f"{cfg.dataset}_case1_training_loss.png")
    return {"run_dir": str(run_dir), "metrics": metrics, "metadata": metadata,
            "scaler": scaler_meta, "initial_loss": float(history.iloc[0]["train_mse"]),
            "final_loss": float(history.iloc[-1]["train_mse"])}


def restore_smoke_checkpoint(checkpoint: Path, cfg: Case1Config, device: torch.device) -> VidalLSTM:
    model = VidalLSTM(); optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.scheduler_step_size, cfg.scheduler_gamma)
    scaler = MinMaxNormalizer(); load_checkpoint(checkpoint, model, optimizer, scheduler, scaler, device)
    return model


def write_case1_summary(results: list[dict[str, Any]], output_root: Path) -> Path:
    """Write formal-only three-battery metric table and compact comparison plot."""
    if any(r["metadata"]["development_only"] for r in results):
        raise ValueError("Smoke results cannot enter the formal Case 1 summary")
    summary_root = output_root / "case1_summary"
    tables = summary_root / "tables"; figures = summary_root / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    rows = [{"dataset": r["metadata"]["dataset"], **r["metrics"]} for r in results]
    frame = pd.DataFrame(rows); path = summary_root / "case1_summary.csv"
    frame.to_csv(path, index=False); frame.to_csv(tables / "case1_summary.csv", index=False)
    x = np.arange(len(frame)); width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, key, label in [(-width, "rmse_percent", "RMSE"), (0, "mae_percent", "MAE"),
                                (width, "max_error_percent", "MAX")]:
        ax.bar(x + offset, frame[key], width, label=label)
    ax.set_xticks(x, frame["dataset"]); ax.set_ylabel("SOC error (%)")
    ax.set_title("Vidal Case 1 comparison"); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "case1_rmse_mae_max_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path
