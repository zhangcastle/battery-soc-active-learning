from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from battery_benchmark.models.vidal_lstm import VidalLSTM


def make_v2_split(root: Path, dataset: str, split: str, specs=None, timestamp_step: float = 1.0) -> Path:
    specs = specs or [(2, 8, "PASS", "drive_cycle", "Mix2"), (1, 7, "WARNING", "charge", "Charge1")]
    split_root = root / dataset / split
    records = split_root / "records"; records.mkdir(parents=True)
    rows = []
    for order, n, status, file_type, profile in specs:
        name = f"{order:02d}__record_{order}.parquet"
        time = np.arange(n, dtype=float) * timestamp_step
        frame = pd.DataFrame({
            "time_s": time, "timestamp": time,
            "voltage_v": np.linspace(4.2, 3.2, n), "current_a": np.full(n, -1.0),
            "temperature_c": np.full(n, 25.0), "soc_sequence_v2": np.linspace(1, 0.2, n),
        })
        frame.to_parquet(records / name, index=False)
        digest = hashlib.sha256((records / name).read_bytes()).hexdigest()
        rows.append({
            "dataset": dataset, "protocol_split": split, "sequence_order": order,
            "sequence_segment_id": 1 if order < 2 else 2, "source_filename": f"record_{order}.mat",
            "source_raw_path": f"record_{order}.mat", "profile": profile, "file_type": file_type,
            "record_role": "model_input", "row_count": n, "label_qc_status": status,
            "label_qc_reasons": "synthetic warning" if status == "WARNING" else "",
            "artifact_path": f"unused/{name}", "artifact_sha256": digest, "source_sha256": "abc",
            "soc_init_source": "synthetic",
        })
    pd.DataFrame(rows).to_csv(split_root / "sequence_index.csv", index=False)
    (split_root / "sequence_metadata.json").write_text("{}")
    pd.concat([pd.read_parquet(p) for p in sorted(records.glob("*.parquet"))]).to_parquet(
        split_root / "concatenated_sequence.parquet", index=False
    )
    return split_root


def make_case1_checkpoint(path: Path, dataset: str, *, formal: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = VidalLSTM()
    with torch.no_grad():
        for i, parameter in enumerate(model.parameters()):
            if parameter.requires_grad:
                parameter.fill_(0.01 * (i + 1))
    config = {
        "dataset": dataset,
        "run_mode": "formal" if formal else "smoke",
        "development_only": not formal,
        "epochs": 515 if formal else 2,
    }
    torch.save({
        "epoch": 514 if formal else 1,
        "best_metric": 0.1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "scheduler_state_dict": {"last_epoch": 514 if formal else 1},
        "normalizer_state_dict": {"type": "minmax", "feature_range": (0.0, 1.0), "min": [0, 0, 0], "max": [1, 1, 1]},
        "config": config,
        "git_commit": "abc123",
        "dataset_inventory_hash": "inventory",
    }, path)
    return path


def make_case1_run(root: Path, dataset: str, *, formal: bool = True) -> Path:
    run = root / dataset / "seed_42" / "run_001"
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    make_case1_checkpoint(run / "checkpoints" / "epoch_515.pt", dataset, formal=formal)
    (run / "metrics.json").write_text(
        '{"mae_percent": 2.0, "rmse_percent": 3.0, "max_error_percent": 8.0}',
        encoding="utf-8",
    )
    (run / "run_metadata.json").write_text(
        pd.Series({
            "dataset": dataset,
            "run_mode": "formal" if formal else "smoke",
            "development_only": not formal,
        }).to_json(),
        encoding="utf-8",
    )
    return run
