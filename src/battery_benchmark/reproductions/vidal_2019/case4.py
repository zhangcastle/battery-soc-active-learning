"""Vidal Case 4 three-battery reconstruction.

Case 4 is implemented as a two-stage protocol because the public Samsung 48G
source data are unavailable:

Stage A: Panasonic + LG HG2 + Samsung 30T partial_train multi-source pretraining.
Stage B: Stage A model weights -> target full_train fine-tuning and test.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from battery_benchmark.reproductions.vidal_2019.resampling import resample_vidal_records, sampling_inventory
from battery_benchmark.reproductions.vidal_2019.windows import (
    FEATURES,
    Case1WindowDataset,
    Case1Windows,
    build_case1_windows,
)
from battery_benchmark.reproductions.vidal_2019.sequence_v2 import VidalSequenceV2Dataset, sha256_file, validate_case1_roots
from battery_benchmark.reproductions.vidal_2019.case1 import (
    DATASETS,
    _merge_sampling_inventory,
    _sampling_protocol,
    evaluate_model,
    fit_train_normalizer,
    hash_dataframe,
    masked_mse,
    metrics_by_profile,
    plot_prediction_and_error,
    plot_training_loss,
    transform_records,
    utc_now,
)
from battery_benchmark.reproductions.vidal_2019.case2 import (
    _optimizer_parameter_count,
    compare_to_case1,
    load_target_case1_baseline,
)
from battery_benchmark.reproductions.vidal_2019.model import VidalLSTM
from battery_benchmark.preprocessing.normalization import MinMaxNormalizer
from battery_benchmark.training.checkpointing import save_checkpoint
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty

CASE4_SOURCE_DATASETS = ("panasonic_18650pf", "lg_hg2", "samsung_30t")
CASE4_STAGE_PRETRAIN = "multi_source_partial_pretraining"
CASE4_STAGE_FINETUNE = "target_full_train_finetuning"
RECONSTRUCTION_SCOPE = "three_battery"


@dataclass
class Case4PretrainConfig:
    source_datasets: tuple[str, ...] = CASE4_SOURCE_DATASETS
    source_train_split: str = "partial_train"
    run_mode: str = "smoke"
    random_seed: int = 42
    seq_len: int = 50
    stride: int = 50
    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.01
    scheduler_step_size: int = 100
    scheduler_gamma: float = 0.5
    shuffle: bool = True
    num_workers: int = 0
    max_train_windows: int | None = 32
    allowed_statuses: tuple[str, ...] = ("PASS", "WARNING")
    development_only: bool = True
    verify_hashes: bool = False
    resample_hz: float | None = 1.0
    sampling_balance: str = "natural"
    scaler_fit_scope: str = "multi_source_partial_train"
    case_number: int = 4

    def validate(self) -> None:
        if self.case_number != 4:
            raise ValueError("Case 4 Stage A case_number must be 4")
        if tuple(self.source_datasets) != CASE4_SOURCE_DATASETS:
            raise ValueError("Case 4 reconstruction requires Panasonic, LG HG2, and Samsung 30T sources")
        if self.source_train_split != "partial_train":
            raise ValueError("Case 4 Stage A source_train_split must be partial_train")
        if self.sampling_balance != "natural":
            raise ValueError("Case 4 Stage A uses natural sampling only")
        if self.scaler_fit_scope != "multi_source_partial_train":
            raise ValueError("Case 4 Stage A scaler must fit multi_source_partial_train")
        _validate_common(self.run_mode, self.development_only, self.epochs, self.max_train_windows, self.seq_len, self.stride, self.resample_hz)


@dataclass
class Case4FinetuneConfig:
    target_dataset: str = "samsung_30t"
    target_train_split: str = "full_train"
    run_mode: str = "smoke"
    random_seed: int = 42
    seq_len: int = 50
    stride: int = 50
    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.01
    scheduler_step_size: int = 100
    scheduler_gamma: float = 0.5
    shuffle: bool = True
    num_workers: int = 0
    max_train_windows: int | None = 32
    allowed_statuses: tuple[str, ...] = ("PASS", "WARNING")
    development_only: bool = True
    verify_hashes: bool = False
    resample_hz: float | None = 1.0
    source_checkpoint: Path | None = None
    target_case1_metrics_path: Path | None = None
    case_number: int = 4

    @property
    def dataset(self) -> str:
        return self.target_dataset

    def validate(self) -> None:
        if self.case_number != 4:
            raise ValueError("Case 4 Stage B case_number must be 4")
        if self.target_dataset not in DATASETS:
            raise ValueError(f"Unsupported Case 4 target_dataset: {self.target_dataset}")
        if self.target_train_split != "full_train":
            raise ValueError("Case 4 Stage B target_train_split must be full_train")
        if self.source_checkpoint is None:
            raise ValueError("Case 4 Stage B requires a Stage A source_checkpoint")
        _validate_common(self.run_mode, self.development_only, self.epochs, self.max_train_windows, self.seq_len, self.stride, self.resample_hz)


def _validate_common(run_mode: str, development_only: bool, epochs: int, max_train_windows: int | None,
                     seq_len: int, stride: int, resample_hz: float | None) -> None:
    if run_mode not in {"smoke", "formal"}:
        raise ValueError("run_mode must be smoke or formal")
    if stride != seq_len:
        raise ValueError("Case 4 fixed_non_overlapping windows require stride == seq_len")
    if resample_hz is not None and resample_hz <= 0:
        raise ValueError("resample_hz must be positive or null")
    if run_mode == "formal":
        if development_only or epochs != 515 or max_train_windows is not None:
            raise ValueError("Formal Case 4 requires 515 epochs, all windows, development_only=false")
    elif not development_only:
        raise ValueError("Smoke Case 4 must be development_only")


def _run_id(repo_root: Path) -> str:
    commit = get_full_commit_hash(repo_root)
    short = commit[:8] if commit != "unknown" else "unknown"
    dirty = "_dirty" if is_dirty(repo_root) else ""
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S") + f"_{short}{dirty}"


def resolve_case4_pretrain_run_dir(output_root: Path, cfg: Case4PretrainConfig, repo_root: Path) -> Path:
    run_dir = output_root / f"seed_{cfg.random_seed}" / _run_id(repo_root)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {run_dir}")
    return run_dir


def resolve_case4_finetune_run_dir(output_root: Path, cfg: Case4FinetuneConfig, repo_root: Path) -> Path:
    run_dir = output_root / f"target_{cfg.target_dataset}" / f"seed_{cfg.random_seed}" / _run_id(repo_root)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {run_dir}")
    return run_dir


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def _serializable_config(cfg: Any) -> dict[str, Any]:
    data = asdict(cfg)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
        elif isinstance(value, tuple):
            data[key] = list(value)
    return data


def _concat_windows(windows_by_dataset: dict[str, Case1Windows]) -> Case1Windows:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    dropped = 0
    for dataset in CASE4_SOURCE_DATASETS:
        windows = windows_by_dataset[dataset]
        if len(windows.X):
            xs.append(windows.X)
            ys.append(windows.y)
            masks.append(windows.valid_mask)
        metadata.extend(windows.metadata)
        dropped += windows.dropped_samples
    if xs:
        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        mask = np.concatenate(masks, axis=0)
    else:
        X = np.empty((0, 50, len(FEATURES)), dtype=np.float32)
        y = np.empty((0, 50), dtype=np.float32)
        mask = np.empty((0, 50), dtype=bool)
    return Case1Windows(X, y, mask, metadata, dropped)


def _fit_multi_source_normalizer(records_by_dataset: dict[str, list], cfg: Case4PretrainConfig, repo_root: Path):
    values = np.concatenate([
        record.data[list(FEATURES)].to_numpy(dtype=np.float32)
        for dataset in cfg.source_datasets
        for record in records_by_dataset[dataset]
    ])
    normalizer = MinMaxNormalizer().fit(values)
    state = normalizer.state_dict()
    metadata = {
        "feature_names": list(FEATURES),
        "feature_min": state["min"],
        "feature_max": state["max"],
        "feature_range": list(state["feature_range"]),
        "fit_split": cfg.source_train_split,
        "scaler_fit_scope": cfg.scaler_fit_scope,
        "n_fit_samples": len(values),
        "source_datasets": list(cfg.source_datasets),
        "git_commit": get_full_commit_hash(repo_root),
        "git_dirty": is_dirty(repo_root),
        "timestamp": utc_now(),
        "normalization_status": "multi_source_partial_train_reconstruction",
    }
    return normalizer, metadata


def _subset_dataset(dataset, max_windows: int | None):
    if max_windows is None:
        return dataset
    return Subset(dataset, range(min(max_windows, len(dataset))))


def _train_model(model: VidalLSTM, loader: DataLoader, cfg, device: torch.device, run_dir: Path,
                 scaler: MinMaxNormalizer, checkpoint_extra: dict[str, Any]):
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.scheduler_step_size, cfg.scheduler_gamma)
    if _optimizer_parameter_count(optimizer) != 571:
        raise ValueError("Case 4 optimizer must contain exactly 571 trainable parameters")
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
        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            epoch,
            train_mse,
            _serializable_config(cfg),
            scheduler=scheduler,
            normalizer=scaler,
            extra={**checkpoint_extra, "train_mse": train_mse, "random_seed": cfg.random_seed, "device": str(device)},
        )
    if cfg.run_mode == "formal":
        shutil.copy2(run_dir / "checkpoints" / "last.pt", run_dir / "checkpoints" / "epoch_515.pt")
    return pd.DataFrame(history), optimizer, scheduler


def _prepare_run_dir(run_dir: Path) -> None:
    for name in ("checkpoints", "figures", "predictions", "logs"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)


def run_case4_pretrain(cfg: Case4PretrainConfig, input_root: Path, output_root: Path, repo_root: Path,
                       device_name: str = "auto") -> dict[str, Any]:
    cfg.validate()
    git_commit = get_full_commit_hash(repo_root)
    git_dirty = is_dirty(repo_root)
    if cfg.run_mode == "formal":
        if git_commit == "unknown":
            raise RuntimeError("Formal Case 4 Stage A requires a known Git commit")
        if git_dirty:
            raise RuntimeError("Formal Case 4 Stage A requires a clean Git working tree")

    checks = validate_case1_roots(input_root, datasets=cfg.source_datasets, splits=[cfg.source_train_split])
    if (checks["status"] != "PASS").any():
        raise FileNotFoundError("Vidal v2 partial_train input preflight failed")

    datasets = {
        dataset: VidalSequenceV2Dataset(input_root, dataset, cfg.source_train_split, cfg.allowed_statuses, cfg.verify_hashes)
        for dataset in cfg.source_datasets
    }
    raw_records_by_dataset = {
        dataset: resample_vidal_records(datasets[dataset].records, cfg.resample_hz)
        for dataset in cfg.source_datasets
    }
    scaler, scaler_meta = _fit_multi_source_normalizer(raw_records_by_dataset, cfg, repo_root)
    scaler_meta.update({"sampling_protocol": _sampling_protocol(cfg.resample_hz), "resample_hz": cfg.resample_hz})
    transformed_by_dataset = {
        dataset: transform_records(raw_records_by_dataset[dataset], scaler)
        for dataset in cfg.source_datasets
    }
    windows_by_dataset = {
        dataset: build_case1_windows(transformed_by_dataset[dataset], cfg.seq_len, cfg.stride, False)
        for dataset in cfg.source_datasets
    }
    train_windows = _concat_windows(windows_by_dataset)
    full_dataset = Case1WindowDataset(train_windows)
    train_dataset = _subset_dataset(full_dataset, cfg.max_train_windows)
    actual_train_window_count = len(train_dataset)

    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    generator = torch.Generator().manual_seed(cfg.random_seed)
    device = _device(device_name)
    model = VidalLSTM().to(device)
    run_dir = resolve_case4_pretrain_run_dir(output_root, cfg, repo_root)
    _prepare_run_dir(run_dir)
    loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle,
                        generator=generator, num_workers=cfg.num_workers, pin_memory=device.type == "cuda")

    sampling = pd.concat([
        sampling_inventory(raw_records_by_dataset[dataset], dataset, cfg.source_train_split)
        for dataset in cfg.source_datasets
    ], ignore_index=True)
    inventory = _merge_sampling_inventory(
        pd.concat([datasets[dataset].inventory for dataset in cfg.source_datasets], ignore_index=True),
        sampling,
    )
    inventory_hash = hash_dataframe(inventory)
    source_record_ids_by_dataset = {
        dataset: [str(x) for x in datasets[dataset].inventory["source_filename"].tolist()]
        for dataset in cfg.source_datasets
    }
    source_window_count_by_dataset = {dataset: len(windows_by_dataset[dataset].X) for dataset in cfg.source_datasets}
    source_sample_count_by_dataset = {
        dataset: int(sum(len(record.data) for record in raw_records_by_dataset[dataset]))
        for dataset in cfg.source_datasets
    }
    history, optimizer, scheduler = _train_model(
        model,
        loader,
        cfg,
        device,
        run_dir,
        scaler,
        {
            "case": 4,
            "stage": CASE4_STAGE_PRETRAIN,
            "reconstruction_scope": RECONSTRUCTION_SCOPE,
            "source_datasets": list(cfg.source_datasets),
            "scaler_fit_scope": cfg.scaler_fit_scope,
            "dataset_inventory_hash": inventory_hash,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        },
    )

    inventory.to_csv(run_dir / "dataset_inventory.csv", index=False)
    sampling.to_csv(run_dir / "sampling_inventory.csv", index=False)
    history.to_csv(run_dir / "training_history.csv", index=False)
    (run_dir / "scaler.json").write_text(json.dumps(scaler_meta, indent=2), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(_serializable_config(cfg), sort_keys=False), encoding="utf-8")
    metadata = {
        "case": 4,
        "stage": CASE4_STAGE_PRETRAIN,
        "reconstruction_scope": RECONSTRUCTION_SCOPE,
        "source_datasets": list(cfg.source_datasets),
        "source_train_split": cfg.source_train_split,
        "source_record_ids_by_dataset": source_record_ids_by_dataset,
        "source_window_count_by_dataset": source_window_count_by_dataset,
        "source_sample_count_by_dataset": source_sample_count_by_dataset,
        "total_train_windows": len(train_windows.X),
        "actual_train_window_count": actual_train_window_count,
        "scaler_fit_scope": cfg.scaler_fit_scope,
        "sampling_balance": cfg.sampling_balance,
        "sampling_protocol": _sampling_protocol(cfg.resample_hz),
        "resample_hz": cfg.resample_hz,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "device": str(device),
        "created_at": utc_now(),
        "random_seed": cfg.random_seed,
        "scheduler_last_epoch": scheduler.last_epoch,
        "trainable_parameters": model.count_trainable_parameters(),
        "optimizer_parameter_count": _optimizer_parameter_count(optimizer),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "development_only": cfg.development_only,
        "run_mode": cfg.run_mode,
        "input_root": str(input_root),
        "inventory_hash": inventory_hash,
        "dropped_train_remainder": train_windows.dropped_samples,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_training_loss(history, "multi_source_partial", run_dir / "figures" / "training_loss.png", case_label="Vidal Case 4 Stage A")
    return {"run_dir": str(run_dir), "metadata": metadata, "scaler": scaler_meta,
            "initial_loss": float(history.iloc[0]["train_mse"]), "final_loss": float(history.iloc[-1]["train_mse"])}


def load_case4_pretrain_weights_for_finetune(model: VidalLSTM, checkpoint_path: str | Path,
                                             *, require_formal: bool = True,
                                             device: torch.device | None = None) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing Case 4 Stage A checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Case 4 source checkpoint missing model_state_dict")
    config = checkpoint.get("config", {})
    case = checkpoint.get("case", config.get("case"))
    stage = checkpoint.get("stage", config.get("stage"))
    run_mode = checkpoint.get("run_mode", config.get("run_mode"))
    development_only = checkpoint.get("development_only", config.get("development_only"))
    epochs = checkpoint.get("epochs", config.get("epochs"))
    source_datasets = tuple(checkpoint.get("source_datasets", config.get("source_datasets", ())))
    if case != 4 or stage != CASE4_STAGE_PRETRAIN:
        raise ValueError("Source checkpoint must be Case 4 Stage A multi-source pretraining")
    if set(source_datasets) != set(CASE4_SOURCE_DATASETS):
        raise ValueError("Source checkpoint must include the three available source datasets")
    if require_formal:
        if run_mode != "formal" or development_only is not False or int(epochs) != 515:
            raise ValueError("Formal Case 4 Stage B requires a formal Stage A 515-epoch checkpoint")
        if int(checkpoint.get("epoch", -1)) != 514:
            raise ValueError("Source checkpoint must be completed epoch 515 (zero-based epoch 514)")
    source_state = checkpoint["model_state_dict"]
    model.load_state_dict(source_state, strict=True)
    if model.count_trainable_parameters() != 571:
        raise ValueError("Case 4 fine-tune model does not expose 571 trainable parameters")
    if model.lstm.bias_hh_l0.requires_grad or torch.count_nonzero(model.lstm.bias_hh_l0).item() != 0:
        raise ValueError("Frozen secondary LSTM bias was not preserved")
    return {
        "source_checkpoint_path": str(path),
        "source_checkpoint_sha256": sha256_file(path),
        "source_checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "source_checkpoint_case": case,
        "source_checkpoint_stage": stage,
        "source_checkpoint_git_commit": checkpoint.get("git_commit", "unknown"),
        "source_checkpoint_source_datasets": list(source_datasets),
        "weights_loaded_from_source": True,
    }


def run_case4_finetune(cfg: Case4FinetuneConfig, input_root: Path, output_root: Path, repo_root: Path,
                       device_name: str = "auto", case1_output_root: Path | None = None,
                       target_case1_run: Path | None = None) -> dict[str, Any]:
    cfg.validate()
    git_commit = get_full_commit_hash(repo_root)
    git_dirty = is_dirty(repo_root)
    if cfg.run_mode == "formal":
        if git_commit == "unknown":
            raise RuntimeError("Formal Case 4 Stage B requires a known Git commit")
        if git_dirty:
            raise RuntimeError("Formal Case 4 Stage B requires a clean Git working tree")

    checks = validate_case1_roots(input_root, datasets=[cfg.target_dataset], splits=[cfg.target_train_split, "test"])
    if (checks["status"] != "PASS").any():
        raise FileNotFoundError("Vidal v2 target full_train/test input preflight failed")
    target_train = VidalSequenceV2Dataset(input_root, cfg.target_dataset, cfg.target_train_split, cfg.allowed_statuses, cfg.verify_hashes)
    target_test = VidalSequenceV2Dataset(input_root, cfg.target_dataset, "test", cfg.allowed_statuses, cfg.verify_hashes)
    train_records = resample_vidal_records(target_train.records, cfg.resample_hz)
    test_records = resample_vidal_records(target_test.records, cfg.resample_hz)
    scaler, scaler_meta = fit_train_normalizer(train_records, cfg.target_dataset, repo_root, fit_split=cfg.target_train_split)
    scaler_meta.update({
        "sampling_protocol": _sampling_protocol(cfg.resample_hz),
        "resample_hz": cfg.resample_hz,
        "target_scaler_refit": True,
        "source_scaler_used": False,
    })
    train_windows = build_case1_windows(transform_records(train_records, scaler), cfg.seq_len, cfg.stride, False)
    test_windows = build_case1_windows(transform_records(test_records, scaler), cfg.seq_len, cfg.stride, True)
    full_train_dataset = Case1WindowDataset(train_windows)
    train_dataset = _subset_dataset(full_train_dataset, cfg.max_train_windows)
    actual_train_window_count = len(train_dataset)

    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    generator = torch.Generator().manual_seed(cfg.random_seed)
    device = _device(device_name)
    model = VidalLSTM()
    source_meta = load_case4_pretrain_weights_for_finetune(
        model,
        cfg.source_checkpoint,
        require_formal=cfg.run_mode == "formal",
        device=device,
    )
    model.to(device)
    initial_target_mse = _compute_dataset_mse(model, train_dataset, cfg.batch_size, cfg.num_workers, device)
    run_dir = resolve_case4_finetune_run_dir(output_root, cfg, repo_root)
    _prepare_run_dir(run_dir)
    loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle,
                        generator=generator, num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
    sampling = pd.concat([
        sampling_inventory(train_records, cfg.target_dataset, cfg.target_train_split),
        sampling_inventory(test_records, cfg.target_dataset, "test"),
    ], ignore_index=True)
    inventory = _merge_sampling_inventory(pd.concat([target_train.inventory, target_test.inventory], ignore_index=True), sampling)
    inventory_hash = hash_dataframe(inventory)
    history, optimizer, scheduler = _train_model(
        model,
        loader,
        cfg,
        device,
        run_dir,
        scaler,
        {
            "case": 4,
            "stage": CASE4_STAGE_FINETUNE,
            "target_dataset": cfg.target_dataset,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "dataset_inventory_hash": inventory_hash,
            **source_meta,
            "optimizer_reinitialized": True,
            "scheduler_reinitialized": True,
            "source_scaler_used": False,
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
    comparison = compare_to_case1(metrics, baseline, case_number=4)

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
        "case": 4,
        "stage": CASE4_STAGE_FINETUNE,
        "target_dataset": cfg.target_dataset,
        "dataset": cfg.target_dataset,
        "target_train_split": cfg.target_train_split,
        "target_train_record_ids": [str(x) for x in target_train.inventory["source_filename"].tolist()],
        "target_train_window_count": len(train_windows.X),
        "actual_train_window_count": actual_train_window_count,
        "test_windows": len(test_windows.X),
        "test_valid_samples": int(test_windows.valid_mask.sum()),
        **source_meta,
        "optimizer_reinitialized": True,
        "scheduler_reinitialized": True,
        "source_scaler_used": False,
        "target_scaler_refit": True,
        "initial_target_mse_before_training": initial_target_mse,
        "initial_mse_window_count": actual_train_window_count,
        "final_target_train_mse": float(history.iloc[-1]["train_mse"]),
        "scheduler_last_epoch": scheduler.last_epoch,
        "trainable_parameters": model.count_trainable_parameters(),
        "optimizer_parameter_count": _optimizer_parameter_count(optimizer),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "device": str(device),
        "created_at": utc_now(),
        "run_mode": cfg.run_mode,
        "development_only": cfg.development_only,
        "input_root": str(input_root),
        "inventory_hash": inventory_hash,
        "sampling_protocol": _sampling_protocol(cfg.resample_hz),
        "resample_hz": cfg.resample_hz,
        "target_case1_run_path": comparison.get("target_case1_run_path", ""),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pair = f"multi_source_to_{cfg.target_dataset}"
    plot_prediction_and_error(
        predictions,
        pair,
        run_dir / "figures" / f"{pair}_case4_test_soc_prediction_and_error.png",
        case_label="Vidal Case 4",
    )
    plot_training_loss(history, pair, run_dir / "figures" / f"{pair}_case4_training_loss.png", case_label="Vidal Case 4")
    _plot_case1_vs_case4(comparison, cfg.target_dataset, run_dir / "figures" / f"{pair}_case1_vs_case4.png")
    return {"run_dir": str(run_dir), "metrics": metrics, "comparison_to_case1": comparison,
            "metadata": metadata, "scaler": scaler_meta,
            "initial_loss": float(history.iloc[0]["train_mse"]), "final_loss": float(history.iloc[-1]["train_mse"])}


@torch.no_grad()
def _compute_dataset_mse(model: VidalLSTM, dataset, batch_size: int, num_workers: int, device: torch.device) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    total = 0.0
    valid = 0
    for x, y, mask, _ in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        loss = masked_mse(model(x), y, mask)
        n_valid = int(mask.sum().item())
        total += float(loss.item()) * n_valid
        valid += n_valid
    return total / max(valid, 1)


def _plot_case1_vs_case4(comparison: dict[str, Any], target: str, path: Path) -> None:
    if not comparison.get("baseline_available"):
        return
    import matplotlib.pyplot as plt
    labels = ["RMSE", "MAE", "MAX"]
    case1 = [comparison["case1_target_rmse_percent"], comparison["case1_target_mae_percent"], comparison["case1_target_max_percent"]]
    case4 = [comparison["case4_rmse_percent"], comparison["case4_mae_percent"], comparison["case4_max_percent"]]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, case1, width, label=f"{target} Case 1")
    ax.bar(x + width / 2, case4, width, label="Case 4")
    ax.set_xticks(x, labels)
    ax.set_ylabel("SOC error (%)")
    ax.set_title("Case 1 baseline vs Case 4 transfer")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
