"""Vidal et al. (2019) Case 2 transfer-learning pipeline."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
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

from battery_benchmark.data.vidal_case1_resampling import (
    resample_vidal_records, sampling_inventory,
)
from battery_benchmark.data.vidal_case1_windows import (
    Case1WindowDataset, build_case1_windows,
)
from battery_benchmark.data.vidal_sequence_v2_dataset import (
    VidalSequenceV2Dataset, validate_case1_roots, sha256_file,
)
from battery_benchmark.experiments.vidal_case1 import (
    DATASETS, _merge_sampling_inventory, _sampling_protocol, cache_case1_inputs,
    evaluate_model, fit_train_normalizer, hash_dataframe, masked_mse,
    metrics_by_profile, plot_prediction_and_error, plot_training_loss,
    transform_records, utc_now,
)
from battery_benchmark.evaluation.metrics import compute_metrics
from battery_benchmark.models.vidal_lstm import VidalLSTM
from battery_benchmark.preprocessing.normalization import MinMaxNormalizer
from battery_benchmark.training.checkpointing import load_checkpoint, save_checkpoint
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty


PAIR_DATASETS = DATASETS


@dataclass
class Case2Config:
    source_dataset: str
    target_dataset: str
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
    resample_hz: float | None = 1.0
    source_checkpoint: Path | None = None
    target_case1_metrics_path: Path | None = None

    @property
    def dataset(self) -> str:
        """Compatibility shim for Case 1 evaluation helpers."""
        return self.target_dataset

    def validate(self) -> None:
        for name, dataset in (("source_dataset", self.source_dataset), ("target_dataset", self.target_dataset)):
            if dataset not in PAIR_DATASETS:
                raise ValueError(f"Unsupported {name}: {dataset}")
        if self.source_dataset == self.target_dataset:
            raise ValueError("Case 2 requires source_dataset != target_dataset")
        if self.run_mode not in {"smoke", "formal"}:
            raise ValueError("run_mode must be smoke or formal")
        if self.stride != self.seq_len:
            raise ValueError("Case 2 fixed_non_overlapping windows require stride == seq_len")
        if self.resample_hz is not None and self.resample_hz <= 0:
            raise ValueError("resample_hz must be positive or null")
        if self.source_checkpoint is None:
            raise ValueError("Case 2 requires a source Case 1 checkpoint")
        if self.run_mode == "formal":
            if self.development_only or self.epochs != 515 or self.max_train_windows is not None:
                raise ValueError("Formal Case 2 requires 515 epochs, all windows, development_only=false")
        elif not self.development_only:
            raise ValueError("Smoke Case 2 must be development_only")


def all_case2_pairs() -> list[tuple[str, str]]:
    return [(source, target) for source in PAIR_DATASETS for target in PAIR_DATASETS if source != target]


def _run_id(repo_root: Path) -> str:
    commit = get_full_commit_hash(repo_root)
    short = commit[:8] if commit != "unknown" else "unknown"
    dirty = "_dirty" if is_dirty(repo_root) else ""
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S") + f"_{short}{dirty}"


def resolve_case2_run_dir(output_root: Path, cfg: Case2Config, repo_root: Path) -> Path:
    pair = f"source_{cfg.source_dataset}__target_{cfg.target_dataset}"
    base = output_root / "_smoke" / pair if cfg.run_mode == "smoke" else output_root / pair
    run_dir = base / f"seed_{cfg.random_seed}" / _run_id(repo_root)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {run_dir}")
    return run_dir


def load_case1_weights_for_transfer(
    model: VidalLSTM,
    checkpoint_path: str | Path,
    expected_source_dataset: str,
    *,
    require_formal: bool = True,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load only Case 1 model weights for transfer and validate provenance."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing source checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Source checkpoint missing model_state_dict")
    config = checkpoint.get("config", {})
    dataset = config.get("dataset", checkpoint.get("dataset"))
    run_mode = config.get("run_mode", checkpoint.get("run_mode"))
    development_only = config.get("development_only", checkpoint.get("development_only"))
    epochs = config.get("epochs", checkpoint.get("epochs"))
    if dataset != expected_source_dataset:
        raise ValueError(f"Source checkpoint dataset mismatch: {dataset} != {expected_source_dataset}")
    if require_formal:
        if run_mode != "formal" or development_only is not False or int(epochs) != 515:
            raise ValueError("Source checkpoint must be a formal Case 1 515-epoch run")
        if "epoch" not in checkpoint:
            raise ValueError("Source checkpoint must include completed epoch metadata")
        actual_epoch = int(checkpoint["epoch"])
        if actual_epoch != 514:
            raise ValueError(
                "Source checkpoint must be the completed epoch-515 checkpoint "
                f"(zero-based epoch 514), got {actual_epoch}"
            )

    source_state = checkpoint["model_state_dict"]
    model.load_state_dict(source_state, strict=True)
    for name, value in model.state_dict().items():
        if not torch.allclose(value.detach().cpu(), source_state[name].detach().cpu()):
            raise ValueError(f"Transferred parameter mismatch: {name}")
    if model.count_trainable_parameters() != 571:
        raise ValueError("Transferred model does not expose 571 trainable parameters")
    if model.lstm.bias_hh_l0.requires_grad or torch.count_nonzero(model.lstm.bias_hh_l0).item() != 0:
        raise ValueError("Frozen secondary LSTM bias was not preserved")
    return {
        "source_checkpoint_path": str(path),
        "source_checkpoint_sha256": sha256_file(path),
        "source_checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "source_checkpoint_dataset": dataset,
        "source_checkpoint_git_commit": checkpoint.get("git_commit", "unknown"),
        "source_checkpoint_inventory_hash": checkpoint.get("dataset_inventory_hash", ""),
        "source_case1_run_id": path.resolve().parents[1].name if len(path.resolve().parents) > 1 else "",
        "weights_loaded_from_source": True,
        "all_trainable_layers_unlocked": True,
    }


def _optimizer_parameter_count(optimizer: torch.optim.Optimizer) -> int:
    return sum(p.numel() for group in optimizer.param_groups for p in group["params"])


@torch.no_grad()
def compute_window_mse(model: VidalLSTM, windows, cfg: Case2Config, device: torch.device) -> float:
    dataset = Case1WindowDataset(windows)
    return compute_dataset_mse(model, dataset, cfg.batch_size, cfg.num_workers, device)


@torch.no_grad()
def compute_dataset_mse(
    model: VidalLSTM,
    dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    total = 0.0
    valid = 0
    for x, y, mask, _ in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        pred = model(x)
        loss = masked_mse(pred, y, mask)
        n_valid = int(mask.sum().item())
        total += float(loss.item()) * n_valid
        valid += n_valid
    return total / max(valid, 1)


def train_target_model(
    model: VidalLSTM,
    loader: DataLoader,
    cfg: Case2Config,
    device: torch.device,
    run_dir: Path,
    scaler: MinMaxNormalizer,
    checkpoint_extra: dict[str, Any],
) -> tuple[pd.DataFrame, torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR]:
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.scheduler_step_size, cfg.scheduler_gamma)
    if _optimizer_parameter_count(optimizer) != 571:
        raise ValueError("Target optimizer must contain exactly 571 trainable parameters")
    model.to(device)
    history: list[dict[str, Any]] = []
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
            n_valid = int(mask.sum().item())
            loss_sum += float(loss.item()) * n_valid
            valid_sum += n_valid
        train_mse = loss_sum / max(valid_sum, 1)
        history.append({
            "epoch": epoch + 1,
            "train_mse": train_mse,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_time_s": time.perf_counter() - started,
            "n_valid_samples": valid_sum,
        })
        scheduler.step()
        extra = {**checkpoint_extra, "train_mse": train_mse, "random_seed": cfg.random_seed, "device": str(device)}
        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            epoch,
            train_mse,
            _serializable_config(cfg),
            scheduler=scheduler,
            normalizer=scaler,
            extra=extra,
        )
    if cfg.run_mode == "formal":
        shutil.copy2(run_dir / "checkpoints" / "last.pt", run_dir / "checkpoints" / "epoch_515.pt")
    return pd.DataFrame(history), optimizer, scheduler


def _serializable_config(cfg: Case2Config) -> dict[str, Any]:
    data = asdict(cfg)
    for key in ("source_checkpoint", "target_case1_metrics_path"):
        if data[key] is not None:
            data[key] = str(data[key])
    return data


def find_target_case1_run(case1_output_root: Path, target_dataset: str, seed: int = 42) -> Path:
    root = case1_output_root / target_dataset / f"seed_{seed}"
    if not root.is_dir():
        raise FileNotFoundError(f"No Case 1 formal root for {target_dataset}: {root}")
    candidates = []
    for metadata_path in root.glob("*/run_metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run = metadata_path.parent
        if (
            metadata.get("run_mode") == "formal"
            and metadata.get("development_only") is False
            and metadata.get("dataset") == target_dataset
            and (run / "metrics.json").is_file()
            and (run / "checkpoints" / "epoch_515.pt").is_file()
        ):
            candidates.append(run)
    if not candidates:
        raise FileNotFoundError(f"No formal Case 1 run found for target {target_dataset}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def load_target_case1_baseline(
    case1_output_root: Path | None,
    target_dataset: str,
    *,
    seed: int = 42,
    explicit_run: Path | None = None,
    metrics_path: Path | None = None,
    require_formal: bool = True,
) -> dict[str, Any] | None:
    if metrics_path is not None:
        path = Path(metrics_path)
        run = path.parent
    elif explicit_run is not None:
        run = Path(explicit_run)
        path = run / "metrics.json"
    elif case1_output_root is not None:
        run = find_target_case1_run(Path(case1_output_root), target_dataset, seed)
        path = run / "metrics.json"
    else:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Missing Case 1 metrics: {path}")
    metadata_path = run / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Case 1 metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset") != target_dataset:
        raise ValueError("Target Case 1 baseline dataset mismatch")
    if require_formal and (metadata.get("run_mode") != "formal" or metadata.get("development_only") is not False):
        raise ValueError("Target Case 1 baseline must be formal")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    return {"run_path": str(run), "metrics_path": str(path), "metrics": metrics, "metadata": metadata}


def compare_to_case1(case2_metrics: dict[str, float], baseline: dict[str, Any] | None) -> dict[str, Any]:
    if baseline is None:
        return {"baseline_available": False}
    case1 = baseline["metrics"]
    rmse_delta = case2_metrics["rmse_percent"] - case1["rmse_percent"]
    return {
        "baseline_available": True,
        "target_case1_run_path": baseline["run_path"],
        "case1_target_rmse_percent": case1["rmse_percent"],
        "case2_rmse_percent": case2_metrics["rmse_percent"],
        "rmse_delta_percent_points": rmse_delta,
        "rmse_relative_change_percent": rmse_delta / case1["rmse_percent"] * 100,
        "case1_target_mae_percent": case1["mae_percent"],
        "case2_mae_percent": case2_metrics["mae_percent"],
        "mae_delta_percent_points": case2_metrics["mae_percent"] - case1["mae_percent"],
        "case1_target_max_percent": case1["max_error_percent"],
        "case2_max_percent": case2_metrics["max_error_percent"],
        "max_delta_percent_points": case2_metrics["max_error_percent"] - case1["max_error_percent"],
    }


def plot_case1_vs_case2(comparison: dict[str, Any], source: str, target: str, path: Path) -> None:
    if not comparison.get("baseline_available"):
        return
    labels = ["RMSE", "MAE", "MAX"]
    case1 = [
        comparison["case1_target_rmse_percent"],
        comparison["case1_target_mae_percent"],
        comparison["case1_target_max_percent"],
    ]
    case2 = [
        comparison["case2_rmse_percent"],
        comparison["case2_mae_percent"],
        comparison["case2_max_percent"],
    ]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, case1, width, label=f"{target} Case 1")
    ax.bar(x + width / 2, case2, width, label=f"{source} -> {target} Case 2")
    ax.set_xticks(x, labels)
    ax.set_ylabel("SOC error (%)")
    ax.set_title("Case 1 baseline vs Case 2 transfer")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_case2(
    cfg: Case2Config,
    input_root: Path,
    output_root: Path,
    repo_root: Path,
    device_name: str = "auto",
    case1_output_root: Path | None = None,
    target_case1_run: Path | None = None,
) -> dict[str, Any]:
    cfg.validate()
    git_commit = get_full_commit_hash(repo_root)
    git_dirty = is_dirty(repo_root)
    if cfg.run_mode == "formal":
        if git_commit == "unknown":
            raise RuntimeError("Formal Case 2 requires a known Git commit")
        if git_dirty:
            raise RuntimeError("Formal Case 2 requires a clean Git working tree")

    checks = validate_case1_roots(input_root, datasets=[cfg.target_dataset], splits=["full_train", "test"])
    if (checks["status"] != "PASS").any():
        raise FileNotFoundError("Vidal v2 target input preflight failed")

    target_train = VidalSequenceV2Dataset(input_root, cfg.target_dataset, "full_train", cfg.allowed_statuses, cfg.verify_hashes)
    target_test = VidalSequenceV2Dataset(input_root, cfg.target_dataset, "test", cfg.allowed_statuses, cfg.verify_hashes)
    train_records = resample_vidal_records(target_train.records, cfg.resample_hz)
    test_records = resample_vidal_records(target_test.records, cfg.resample_hz)
    scaler, scaler_meta = fit_train_normalizer(train_records, cfg.target_dataset, repo_root)
    scaler_meta.update({
        "sampling_protocol": _sampling_protocol(cfg.resample_hz),
        "resample_hz": cfg.resample_hz,
        "target_scaler_refit": True,
        "source_scaler_used": False,
    })
    train_windows = build_case1_windows(transform_records(train_records, scaler), cfg.seq_len, cfg.stride, False)
    test_windows = build_case1_windows(transform_records(test_records, scaler), cfg.seq_len, cfg.stride, True)
    full_train_dataset = Case1WindowDataset(train_windows)
    train_dataset = full_train_dataset
    if cfg.max_train_windows is not None:
        train_dataset = Subset(train_dataset, range(min(cfg.max_train_windows, len(train_dataset))))
    actual_train_window_count = len(train_dataset)

    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    generator = torch.Generator().manual_seed(cfg.random_seed)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    model = VidalLSTM()
    source_meta = load_case1_weights_for_transfer(
        model,
        cfg.source_checkpoint,
        cfg.source_dataset,
        require_formal=True,
        device=device,
    )
    model.to(device)
    initial_target_mse = compute_dataset_mse(
        model,
        train_dataset,
        cfg.batch_size,
        cfg.num_workers,
        device,
    )
    run_dir = resolve_case2_run_dir(output_root, cfg, repo_root)
    for name in ("checkpoints", "figures", "predictions", "logs"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    sampling = pd.concat([
        sampling_inventory(train_records, cfg.target_dataset, "full_train"),
        sampling_inventory(test_records, cfg.target_dataset, "test"),
    ], ignore_index=True)
    inventory = _merge_sampling_inventory(
        pd.concat([target_train.inventory, target_test.inventory], ignore_index=True),
        sampling,
    )
    inventory_hash = hash_dataframe(inventory)
    loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        generator=generator,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    history, optimizer, scheduler = train_target_model(
        model,
        loader,
        cfg,
        device,
        run_dir,
        scaler,
        {
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "dataset_inventory_hash": inventory_hash,
            "sampling_protocol": _sampling_protocol(cfg.resample_hz),
            "resample_hz": cfg.resample_hz,
            **source_meta,
            "optimizer_reinitialized": True,
            "scheduler_reinitialized": True,
            "target_scaler_refit": True,
        },
    )
    metrics, predictions = evaluate_model(model, test_windows, test_records, cfg, device)
    profile = metrics_by_profile(predictions)
    baseline = load_target_case1_baseline(
        case1_output_root,
        cfg.target_dataset,
        seed=cfg.random_seed,
        explicit_run=target_case1_run,
        metrics_path=cfg.target_case1_metrics_path,
        require_formal=cfg.run_mode == "formal",
    )
    comparison = compare_to_case1(metrics, baseline)

    inventory.to_csv(run_dir / "dataset_inventory.csv", index=False)
    sampling.to_csv(run_dir / "sampling_inventory.csv", index=False)
    history.to_csv(run_dir / "training_history.csv", index=False)
    predictions.to_csv(run_dir / "predictions" / "test_predictions.csv", index=False)
    profile.to_csv(run_dir / "metrics_by_profile.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "comparison_to_case1.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (run_dir / "scaler.json").write_text(json.dumps(scaler_meta, indent=2), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(_serializable_config(cfg), sort_keys=False), encoding="utf-8")

    metadata = {
        "source_dataset": cfg.source_dataset,
        "target_dataset": cfg.target_dataset,
        "dataset": cfg.target_dataset,
        "run_mode": cfg.run_mode,
        "development_only": cfg.development_only,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "device": str(device),
        "created_at": utc_now(),
        "input_root": str(input_root),
        "inventory_hash": inventory_hash,
        "target_train_records": len(target_train.records),
        "target_test_records": len(target_test.records),
        "train_windows": len(train_windows.X),
        "actual_train_window_count": actual_train_window_count,
        "test_windows": len(test_windows.X),
        "test_valid_samples": int(test_windows.valid_mask.sum()),
        "dropped_train_remainder": train_windows.dropped_samples,
        "trainable_parameters": model.count_trainable_parameters(),
        "optimizer_parameter_count": _optimizer_parameter_count(optimizer),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "source_checkpoint": source_meta,
        "weights_loaded_from_source": True,
        "all_trainable_layers_unlocked": True,
        "optimizer_reinitialized": True,
        "scheduler_reinitialized": True,
        "target_scaler_refit": True,
        "initial_target_mse_before_training": initial_target_mse,
        "initial_mse_window_count": actual_train_window_count,
        "final_target_train_mse": float(history.iloc[-1]["train_mse"]),
        "scheduler_last_epoch": scheduler.last_epoch,
        "sampling_protocol": _sampling_protocol(cfg.resample_hz),
        "resample_hz": cfg.resample_hz,
        "target_case1_run_path": comparison.get("target_case1_run_path", ""),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    pair = f"{cfg.source_dataset}_to_{cfg.target_dataset}"
    plot_prediction_and_error(
        predictions, pair, run_dir / "figures" / f"{pair}_case2_test_soc_prediction_and_error.png",
        case_label="Vidal Case 2",
    )
    plot_training_loss(
        history, pair, run_dir / "figures" / f"{pair}_case2_training_loss.png",
        case_label="Vidal Case 2",
    )
    plot_case1_vs_case2(comparison, cfg.source_dataset, cfg.target_dataset, run_dir / "figures" / f"{pair}_case1_vs_case2.png")

    return {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "comparison_to_case1": comparison,
        "metadata": metadata,
        "scaler": scaler_meta,
        "initial_loss": float(history.iloc[0]["train_mse"]),
        "final_loss": float(history.iloc[-1]["train_mse"]),
    }


def write_case2_summary(results: list[dict[str, Any]], output_root: Path) -> Path:
    if not results:
        raise ValueError("No Case 2 results to summarise")
    summary_root = output_root / "case2_summary"
    tables = summary_root
    figures = summary_root
    tables.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        meta = result["metadata"]
        comp = result["comparison_to_case1"]
        rows.append({
            "source_dataset": meta["source_dataset"],
            "target_dataset": meta["target_dataset"],
            "case1_target_rmse_percent": comp.get("case1_target_rmse_percent"),
            "case2_rmse_percent": result["metrics"]["rmse_percent"],
            "rmse_delta_percent_points": comp.get("rmse_delta_percent_points"),
            "rmse_relative_change_percent": comp.get("rmse_relative_change_percent"),
            "case1_target_mae_percent": comp.get("case1_target_mae_percent"),
            "case2_mae_percent": result["metrics"]["mae_percent"],
            "case1_target_max_percent": comp.get("case1_target_max_percent"),
            "case2_max_percent": result["metrics"]["max_error_percent"],
            "source_checkpoint_path": meta["source_checkpoint"]["source_checkpoint_path"],
            "target_case1_run_path": comp.get("target_case1_run_path", ""),
            "git_commit": meta["git_commit"],
            "seed": meta.get("random_seed", 42),
            "run_dir": result["run_dir"],
        })
    frame = pd.DataFrame(rows)
    summary_path = tables / "case2_summary.csv"
    frame.to_csv(summary_path, index=False)
    matrix = frame.pivot(index="source_dataset", columns="target_dataset", values="case2_rmse_percent")
    matrix.to_csv(tables / "case2_rmse_matrix.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    for i, source in enumerate(matrix.index):
        for j, target in enumerate(matrix.columns):
            value = matrix.loc[source, target]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white")
    ax.set_title("Vidal Case 2 RMSE (%)")
    fig.colorbar(im, ax=ax, label="RMSE (%)")
    fig.tight_layout()
    fig.savefig(figures / "case2_rmse_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary_path
