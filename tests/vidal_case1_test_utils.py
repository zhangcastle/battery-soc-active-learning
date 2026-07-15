from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def make_v2_split(root: Path, dataset: str, split: str, specs=None) -> Path:
    specs = specs or [(2, 8, "PASS", "drive_cycle", "Mix2"), (1, 7, "WARNING", "charge", "Charge1")]
    split_root = root / dataset / split
    records = split_root / "records"; records.mkdir(parents=True)
    rows = []
    for order, n, status, file_type, profile in specs:
        name = f"{order:02d}__record_{order}.parquet"
        frame = pd.DataFrame({
            "time_s": np.arange(n, dtype=float), "timestamp": np.arange(n, dtype=float),
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
