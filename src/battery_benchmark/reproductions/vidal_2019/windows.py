"""Record-safe, mask-aware windows for Vidal Case 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from battery_benchmark.reproductions.vidal_2019.sequence_v2 import VidalV2Record


FEATURES = ("voltage", "current", "temperature")


@dataclass
class Case1Windows:
    X: np.ndarray
    y: np.ndarray
    valid_mask: np.ndarray
    metadata: list[dict[str, object]]
    dropped_samples: int


def build_case1_windows(
    records: list[VidalV2Record],
    seq_len: int = 50,
    stride: int = 50,
    keep_incomplete: bool = False,
) -> Case1Windows:
    """Window each record independently; no window can cross a record or segment."""
    if seq_len <= 0 or stride <= 0:
        raise ValueError("seq_len and stride must be positive")
    xs, ys, masks, metadata = [], [], [], []
    dropped = 0
    for record in records:
        df, base = record.data, record.metadata
        n = len(df)
        starts = list(range(0, n, stride))
        for start in starts:
            valid_length = min(seq_len, n - start)
            if valid_length < seq_len and not keep_incomplete:
                dropped += valid_length
                break
            end = start + valid_length
            x = df.loc[df.index[start:end], list(FEATURES)].to_numpy(dtype=np.float32)
            y = df.loc[df.index[start:end], "soc"].to_numpy(dtype=np.float32)
            mask = np.zeros(seq_len, dtype=bool)
            mask[:valid_length] = True
            if valid_length < seq_len:
                x = np.pad(x, ((0, seq_len - valid_length), (0, 0)), mode="edge")
                y = np.pad(y, (0, seq_len - valid_length), mode="edge")
            xs.append(x); ys.append(y); masks.append(mask)
            metadata.append({
                "dataset": base["dataset"], "protocol_split": base["protocol_split"],
                "sequence_order": int(base["sequence_order"]),
                "sequence_segment_id": base["sequence_segment_id"],
                "source_filename": base["source_filename"], "file_type": base["file_type"],
                "paper_profile": base["paper_profile"], "window_start": start,
                "window_end": end, "valid_length": valid_length,
                "label_qc_status": base["label_qc_status"],
                "warning_reasons": base["warning_reasons"],
            })
            if end >= n:
                break
    shape = (0, seq_len, len(FEATURES))
    return Case1Windows(
        np.stack(xs) if xs else np.empty(shape, dtype=np.float32),
        np.stack(ys) if ys else np.empty((0, seq_len), dtype=np.float32),
        np.stack(masks) if masks else np.empty((0, seq_len), dtype=bool),
        metadata, dropped,
    )


class Case1WindowDataset(Dataset):
    def __init__(self, windows: Case1Windows) -> None:
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows.X)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.windows.X[index]),
            torch.from_numpy(self.windows.y[index]),
            torch.from_numpy(self.windows.valid_mask[index]),
            index,
        )
