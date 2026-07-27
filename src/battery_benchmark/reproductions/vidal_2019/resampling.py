"""Minimal record-level resampling for Vidal Case 1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from battery_benchmark.reproductions.vidal_2019.sequence_v2 import VidalV2Record

FEATURE_COLUMNS = ("voltage", "current", "temperature")
RESAMPLE_COLUMNS = (*FEATURE_COLUMNS, "soc")


def _median_positive_dt(time: np.ndarray) -> float:
    if len(time) < 2:
        return 0.0
    dt = np.diff(time.astype(float))
    positive = dt[dt > 0]
    if len(positive) == 0:
        return 0.0
    return float(np.median(positive))


def resample_vidal_record(
    record: VidalV2Record,
    target_hz: float = 1.0,
    *,
    clip_soc: bool = True,
) -> VidalV2Record:
    """Resample one Vidal v2 record to a regular per-record time grid.

    The function never connects records or segments. Duplicate timestamps are
    collapsed with a simple mean over the canonical feature/label columns.
    """
    if target_hz <= 0:
        raise ValueError("target_hz must be positive")

    frame = record.data.copy()
    if "timestamp" not in frame:
        raise ValueError("record is missing canonical timestamp column")
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"record is missing resampling columns: {sorted(missing)}")
    columns = [*FEATURE_COLUMNS, *(["soc"] if "soc" in frame else [])]

    original_n = len(frame)
    if original_n == 0:
        raise ValueError("cannot resample an empty record")

    frame = frame.sort_values("timestamp", kind="stable")
    original_time = frame["timestamp"].to_numpy(dtype=float)
    original_duration = float(original_time[-1] - original_time[0]) if original_n > 1 else 0.0
    median_dt_before = _median_positive_dt(original_time)
    duplicate_count = int(frame["timestamp"].duplicated(keep=False).sum())

    grouped = frame.groupby("timestamp", as_index=False, sort=True)[columns].mean()
    time = grouped["timestamp"].to_numpy(dtype=float)
    if len(time) == 0 or not np.isfinite(time).all():
        raise ValueError("record has no finite timestamp values")
    if np.any(np.diff(time) <= 0):
        raise ValueError("deduplicated timestamp axis is not strictly increasing")

    step = 1.0 / target_hz
    start, stop = float(time[0]), float(time[-1])
    if stop <= start:
        new_time = np.array([start], dtype=float)
    else:
        count = int(np.floor((stop - start) / step)) + 1
        new_time = start + np.arange(count, dtype=float) * step
        if new_time[-1] < stop - 1e-9:
            new_time = np.append(new_time, stop)
        if new_time[-1] > stop + 1e-9:
            raise ValueError("resampled timestamp exceeds original range")

    resampled = pd.DataFrame({"timestamp": new_time})
    resampled["time_s"] = new_time
    soc_clipping_count = 0
    for column in columns:
        values = grouped[column].to_numpy(dtype=float)
        if len(time) == 1:
            out = np.full(len(new_time), values[0], dtype=float)
        else:
            out = np.interp(new_time, time, values)
        if column == "soc" and clip_soc:
            soc_clipping_count = int(np.count_nonzero((out < 0.0) | (out > 1.0)))
            out = np.clip(out, 0.0, 1.0)
        resampled[column] = out

    metadata = dict(record.metadata)
    resampled_duration = float(new_time[-1] - new_time[0]) if len(new_time) > 1 else 0.0
    metadata.update(
        {
            "original_n_samples": int(original_n),
            "duplicate_timestamp_count": duplicate_count,
            "deduplicated_n_samples": int(len(grouped)),
            "resampled_n_samples": int(len(resampled)),
            "original_duration_s": original_duration,
            "resampled_duration_s": resampled_duration,
            "median_dt_s_before": median_dt_before,
            "target_hz": float(target_hz),
            "resampled": True,
            "soc_clipping_enabled": bool(clip_soc and "soc" in columns),
            "soc_clipping_count": soc_clipping_count,
        }
    )
    return VidalV2Record(resampled, metadata)


def resample_vidal_records(
    records: list[VidalV2Record], target_hz: float | None
) -> list[VidalV2Record]:
    """Apply record-level resampling when ``target_hz`` is set."""
    if target_hz is None:
        unchanged = []
        for record in records:
            metadata = dict(record.metadata)
            n = len(record.data)
            time = record.data["timestamp"].to_numpy(dtype=float)
            duration = float(time[-1] - time[0]) if n > 1 else 0.0
            metadata.update(
                {
                    "original_n_samples": int(n),
                    "duplicate_timestamp_count": int(
                        record.data["timestamp"].duplicated(keep=False).sum()
                    ),
                    "deduplicated_n_samples": int(record.data["timestamp"].nunique()),
                    "resampled_n_samples": int(n),
                    "original_duration_s": duration,
                    "resampled_duration_s": duration,
                    "median_dt_s_before": _median_positive_dt(time),
                    "target_hz": None,
                    "resampled": False,
                }
            )
            unchanged.append(VidalV2Record(record.data.copy(), metadata))
        return unchanged
    return [resample_vidal_record(record, target_hz=target_hz) for record in records]


def sampling_inventory(records: list[VidalV2Record], dataset: str, split: str) -> pd.DataFrame:
    """Summarise original and resampled counts for a set of records."""
    rows = []
    for record in records:
        meta = record.metadata
        rows.append(
            {
                "dataset": dataset,
                "protocol_split": split,
                "sequence_order": meta.get("sequence_order"),
                "sequence_segment_id": meta.get("sequence_segment_id"),
                "source_filename": meta.get("source_filename"),
                "file_type": meta.get("file_type"),
                "paper_profile": meta.get("paper_profile"),
                "original_n_samples": meta.get("original_n_samples"),
                "duplicate_timestamp_count": meta.get("duplicate_timestamp_count"),
                "deduplicated_n_samples": meta.get("deduplicated_n_samples"),
                "resampled_n_samples": meta.get("resampled_n_samples"),
                "original_duration_s": meta.get("original_duration_s"),
                "resampled_duration_s": meta.get("resampled_duration_s"),
                "median_dt_s_before": meta.get("median_dt_s_before"),
                "target_hz": meta.get("target_hz"),
                "resampled": meta.get("resampled"),
            }
        )
    return pd.DataFrame(rows)
