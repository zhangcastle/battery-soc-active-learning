"""Development-only CMU dynamic-SOC bridge to Vidal record/window inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from battery_benchmark.reproductions.vidal_2019.resampling import resample_vidal_record
from battery_benchmark.reproductions.vidal_2019.sequence_v2 import VidalV2Record
from battery_benchmark.reproductions.vidal_2019.windows import (
    FEATURES,
    Case1Windows,
    build_case1_windows,
)

ADAPTER_VERSION = "cmu_evtol_to_vidal_adapter_v1"
EXPECTED_LABEL_SOURCE = "cmu_causal_dynamic_capacity_reference_development_v1"
EXPECTED_SOC_PROTOCOL = "cmu_evtol_dynamic_capacity_soc_development_v1"
FEATURE_MAP = {
    "voltage_v": "voltage",
    "current_a": "current",
    "Temperature__C": "temperature",
    "soc_dynamic": "soc",
}
PROVENANCE_COLUMNS = (
    "cell_id",
    "mission_cycle_id",
    "development_candidate_id",
    "source_row",
    "time_s",
    "elapsed_candidate_s",
    "capacity_ah",
    "capacity_source",
    "source_rpt_id",
    "dynamic_soc_protocol_version",
    "label_source",
)
REQUIRED_COLUMNS = frozenset(
    (*FEATURE_MAP, *PROVENANCE_COLUMNS, "label_is_ground_truth", "formal_benchmark_eligible")
)
UDA_COLUMNS = (
    "cell_id",
    "mission_cycle_id",
    "development_candidate_id",
    "source_row",
    "time_s",
    "elapsed_candidate_s",
    "voltage_v",
    "current_a",
    "Temperature__C",
)


class TransformOnlyNormalizer(Protocol):
    def transform(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class CMUVidalAdapterResult:
    records: list[VidalV2Record]
    inventory: pd.DataFrame


def _require_valid_frame(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"CMU dynamic SOC input missing columns: {sorted(missing)}")
    if frame.empty:
        return
    if frame[["cell_id", "mission_cycle_id", "development_candidate_id"]].isna().any().any():
        raise ValueError("CMU record identity fields must not be missing")
    if (
        (frame[["cell_id", "mission_cycle_id", "development_candidate_id"]].astype(str) == "")
        .any()
        .any()
    ):
        raise ValueError("CMU record identity fields must not be empty")
    numeric = frame[[*FEATURE_MAP, "source_row", "time_s", "elapsed_candidate_s", "capacity_ah"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(
            "CMU adapter input contains non-finite feature, SOC, time, row, or capacity"
        )
    if not frame["soc_dynamic"].between(0.0, 1.0).all():
        raise ValueError("CMU dynamic SOC lies outside [0, 1]")
    if frame.duplicated(["cell_id", "source_row"]).any():
        raise ValueError("CMU adapter input contains duplicate source_row")
    if (frame.groupby("mission_cycle_id", dropna=False)["cell_id"].nunique() > 1).any():
        raise ValueError("A mission_cycle_id cannot span multiple cells")
    if not (frame["mission_cycle_id"] == frame["development_candidate_id"]).all():
        raise ValueError("development_candidate_id and mission_cycle_id must agree")
    if set(frame["label_source"].astype(str)) != {EXPECTED_LABEL_SOURCE}:
        raise ValueError(f"label_source must be {EXPECTED_LABEL_SOURCE}")
    if set(frame["dynamic_soc_protocol_version"].astype(str)) != {EXPECTED_SOC_PROTOCOL}:
        raise ValueError(f"dynamic SOC protocol must be {EXPECTED_SOC_PROTOCOL}")
    if frame["label_is_ground_truth"].astype(bool).any():
        raise ValueError("CMU development SOC must not be marked as ground truth")
    if frame["formal_benchmark_eligible"].astype(bool).any():
        raise ValueError("CMU development SOC must not be marked formal-benchmark eligible")

    keys = list(zip(frame["cell_id"].astype(str), frame["mission_cycle_id"].astype(str)))
    completed: set[tuple[str, str]] = set()
    previous = keys[0]
    for key in keys[1:]:
        if key != previous:
            completed.add(previous)
            if key in completed:
                raise ValueError(
                    "CMU candidate rows must be contiguous and preserve lifecycle order"
                )
            previous = key


def _record_from_group(group: pd.DataFrame, sequence_order: int, source_path: str) -> VidalV2Record:
    cell_id = str(group["cell_id"].iloc[0])
    mission_id = str(group["mission_cycle_id"].iloc[0])
    if group["cell_id"].nunique() != 1 or group["mission_cycle_id"].nunique() != 1:
        raise ValueError("A CMU record cannot mix cells or candidates")
    source_rows = group["source_row"].to_numpy(dtype=np.int64)
    time = group["time_s"].to_numpy(dtype=float)
    elapsed = group["elapsed_candidate_s"].to_numpy(dtype=float)
    if np.any(np.diff(source_rows) <= 0):
        raise ValueError(f"{mission_id}: source_row must be strictly increasing in input order")
    if np.any(np.diff(time) <= 0) or np.any(np.diff(elapsed) <= 0):
        raise ValueError(f"{mission_id}: time axes must be strictly increasing in input order")
    if not np.array_equal(elapsed, time - time[0]):
        raise ValueError(f"{mission_id}: elapsed_candidate_s must equal time_s - first time_s")
    if group["capacity_ah"].nunique(dropna=False) != 1:
        raise ValueError(f"{mission_id}: capacity_ah must be constant within a candidate")
    for column in ("capacity_source", "source_rpt_id"):
        if group[column].astype("string").fillna("<null>").nunique(dropna=False) != 1:
            raise ValueError(f"{mission_id}: {column} must be constant within a candidate")

    canonical = group.copy()
    canonical["timestamp"] = elapsed
    for source, target in FEATURE_MAP.items():
        canonical[target] = canonical[source]
    metadata = {
        "dataset": "cmu_evtol_vtc6",
        "domain_role": "target",
        "protocol_split": "development_dynamic_soc_v1",
        "sequence_order": sequence_order,
        "sequence_segment_id": mission_id,
        "source_filename": Path(source_path).name,
        "file_type": "development_mission_candidate",
        "paper_profile": "cmu_evtol_target_candidate",
        "label_qc_status": "WARNING",
        "warning_reasons": "development_reference_not_ground_truth",
        "cell_id": cell_id,
        "mission_cycle_id": mission_id,
        "development_candidate_id": str(group["development_candidate_id"].iloc[0]),
        "source_path": source_path,
        "original_source_row_start": int(source_rows[0]),
        "original_source_row_end": int(source_rows[-1]),
        "original_time_s_start": float(time[0]),
        "original_time_s_end": float(time[-1]),
        "capacity_ah": float(group["capacity_ah"].iloc[0]),
        "capacity_source": str(group["capacity_source"].iloc[0]),
        "source_rpt_id": group["source_rpt_id"].iloc[0],
        "dynamic_soc_protocol_version": EXPECTED_SOC_PROTOCOL,
        "label_source": EXPECTED_LABEL_SOURCE,
        "label_is_ground_truth": False,
        "formal_benchmark_eligible": False,
        "adapter_version": ADAPTER_VERSION,
    }
    return VidalV2Record(canonical, metadata)


def _resample_cmu_record(record: VidalV2Record, target_hz: float) -> VidalV2Record:
    original = record.data
    original_elapsed = original["elapsed_candidate_s"].to_numpy(dtype=float)
    original_time = original["time_s"].to_numpy(dtype=float)
    original_rows = original["source_row"].to_numpy(dtype=np.int64)
    out = resample_vidal_record(record, target_hz=target_hz, clip_soc=False)
    frame = out.data
    elapsed = frame["timestamp"].to_numpy(dtype=float)
    right = np.searchsorted(original_elapsed, elapsed, side="left")
    right = np.clip(right, 0, len(original_rows) - 1)
    exact = original_elapsed[right] == elapsed
    left = np.where(exact, right, np.maximum(right - 1, 0))
    frame["elapsed_candidate_s"] = elapsed
    frame["original_time_s"] = np.interp(elapsed, original_elapsed, original_time)
    frame["source_row"] = pd.array(np.where(exact, original_rows[right], -1), dtype="Int64")
    frame.loc[~exact, "source_row"] = pd.NA
    frame["source_row_left"] = original_rows[left]
    frame["source_row_right"] = original_rows[right]
    for column in (
        "cell_id",
        "mission_cycle_id",
        "development_candidate_id",
        "capacity_ah",
        "capacity_source",
        "source_rpt_id",
        "dynamic_soc_protocol_version",
        "label_source",
    ):
        if column in record.data:
            frame[column] = record.data[column].iloc[0]
    metadata = dict(out.metadata)
    metadata["provenance_interpolation"] = "source_row_left/right bracket; no extrapolation"
    return VidalV2Record(frame, metadata)


def adapt_cmu_dynamic_soc(
    frame: pd.DataFrame,
    *,
    source_path: str = "<memory>",
    target_hz: float | None = 1.0,
) -> CMUVidalAdapterResult:
    """Validate and adapt CMU development candidates without fitting a scaler."""
    _require_valid_frame(frame)
    if target_hz is not None and target_hz <= 0:
        raise ValueError("target_hz must be positive or None")
    records: list[VidalV2Record] = []
    for order, (_, group) in enumerate(
        frame.groupby(["cell_id", "mission_cycle_id"], sort=False, dropna=False)
    ):
        group_source = (
            str(group["_adapter_source_path"].iloc[0])
            if "_adapter_source_path" in group
            else source_path
        )
        record = _record_from_group(group, order, group_source)
        if target_hz is not None:
            record = _resample_cmu_record(record, target_hz)
        else:
            metadata = dict(record.metadata)
            n = len(record.data)
            dt = np.diff(record.data["timestamp"].to_numpy(dtype=float))
            metadata.update(
                {
                    "original_n_samples": n,
                    "deduplicated_n_samples": n,
                    "resampled_n_samples": n,
                    "duplicate_timestamp_count": 0,
                    "original_duration_s": float(record.data["timestamp"].iloc[-1]),
                    "resampled_duration_s": float(record.data["timestamp"].iloc[-1]),
                    "median_dt_s_before": float(np.median(dt)) if len(dt) else 0.0,
                    "target_hz": None,
                    "resampled": False,
                    "soc_clipping_enabled": False,
                    "soc_clipping_count": 0,
                }
            )
            record = VidalV2Record(record.data, metadata)
        records.append(record)
    inventory = pd.DataFrame(
        [
            {
                "cell_id": r.metadata["cell_id"],
                "mission_cycle_id": r.metadata["mission_cycle_id"],
                "original_rows": r.metadata["original_n_samples"],
                "adapted_rows": r.metadata["resampled_n_samples"],
                "duration_s": r.metadata["resampled_duration_s"],
                "median_dt_s_before": r.metadata["median_dt_s_before"],
                "duplicate_timestamp_count": r.metadata["duplicate_timestamp_count"],
                "soc_clipping_count": r.metadata["soc_clipping_count"],
            }
            for r in records
        ],
        columns=(
            "cell_id",
            "mission_cycle_id",
            "original_rows",
            "adapted_rows",
            "duration_s",
            "median_dt_s_before",
            "duplicate_timestamp_count",
            "soc_clipping_count",
        ),
    )
    return CMUVidalAdapterResult(records, inventory)


def load_cmu_dynamic_soc_for_vidal(
    paths: list[str | Path], *, target_hz: float | None = 1.0
) -> CMUVidalAdapterResult:
    if not paths:
        return adapt_cmu_dynamic_soc(
            pd.DataFrame(columns=sorted(REQUIRED_COLUMNS)), target_hz=target_hz
        )
    frames = []
    for path in paths:
        path = Path(path)
        frame = pd.read_parquet(path)
        frame = frame.copy()
        frame["_adapter_source_path"] = str(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    result = adapt_cmu_dynamic_soc(
        combined, source_path="multiple_parquet_inputs", target_hz=target_hz
    )
    return result


def adapt_cmu_features_for_uda(
    frame: pd.DataFrame,
    *,
    source_path: str = "<memory>",
    target_hz: float | None = 1.0,
) -> CMUVidalAdapterResult:
    """Adapt CMU features without accepting SOC or capacity label columns."""
    missing = set(UDA_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"CMU UDA input missing columns: {sorted(missing)}")
    if frame.empty:
        return CMUVidalAdapterResult([], pd.DataFrame())
    identity = frame[["cell_id", "mission_cycle_id", "development_candidate_id"]]
    if identity.isna().any().any() or (identity.astype(str) == "").any().any():
        raise ValueError("CMU UDA identity fields must not be missing")
    numeric = frame[
        [
            "source_row",
            "time_s",
            "elapsed_candidate_s",
            "voltage_v",
            "current_a",
            "Temperature__C",
        ]
    ]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("CMU UDA input contains non-finite values")
    if frame.duplicated(["cell_id", "source_row"]).any():
        raise ValueError("CMU UDA input contains duplicate source_row")
    if not (frame["mission_cycle_id"] == frame["development_candidate_id"]).all():
        raise ValueError("development_candidate_id and mission_cycle_id must agree")

    records = []
    for order, (_, group) in enumerate(
        frame.groupby(["cell_id", "mission_cycle_id"], sort=False, dropna=False)
    ):
        source_rows = group["source_row"].to_numpy(dtype=np.int64)
        time = group["time_s"].to_numpy(dtype=float)
        elapsed = group["elapsed_candidate_s"].to_numpy(dtype=float)
        if np.any(np.diff(source_rows) <= 0):
            raise ValueError("CMU UDA source_row must be strictly increasing")
        if np.any(np.diff(time) <= 0) or np.any(np.diff(elapsed) <= 0):
            raise ValueError("CMU UDA time axes must be strictly increasing")
        canonical = group.copy()
        canonical["timestamp"] = elapsed
        for source, target in {
            "voltage_v": "voltage",
            "current_a": "current",
            "Temperature__C": "temperature",
        }.items():
            canonical[target] = canonical[source]
        record = VidalV2Record(
            canonical,
            {
                "dataset": "cmu_evtol_vtc6",
                "domain_role": "target_adapt",
                "sequence_order": order,
                "sequence_segment_id": str(group["mission_cycle_id"].iloc[0]),
                "source_filename": Path(source_path).name,
                "cell_id": str(group["cell_id"].iloc[0]),
                "mission_cycle_id": str(group["mission_cycle_id"].iloc[0]),
                "label_available": False,
                "label_usage": "forbidden",
            },
        )
        records.append(_resample_cmu_record(record, target_hz) if target_hz is not None else record)
    inventory = pd.DataFrame(
        [
            {
                "cell_id": record.metadata["cell_id"],
                "mission_cycle_id": record.metadata["mission_cycle_id"],
                "adapted_rows": len(record.data),
            }
            for record in records
        ]
    )
    return CMUVidalAdapterResult(records, inventory)


def load_cmu_features_for_uda(
    paths: list[str | Path],
    *,
    target_hz: float | None = 1.0,
) -> CMUVidalAdapterResult:
    """Read only feature/provenance columns required for unsupervised adaptation."""
    frames = []
    for path in paths:
        path = Path(path)
        frames.append(pd.read_parquet(path, columns=list(UDA_COLUMNS)))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=UDA_COLUMNS)
    return adapt_cmu_features_for_uda(
        combined,
        source_path="multiple_parquet_inputs",
        target_hz=target_hz,
    )


def build_cmu_vidal_windows(
    records: list[VidalV2Record],
    *,
    seq_len: int = 50,
    stride: int = 50,
    keep_incomplete: bool = False,
) -> Case1Windows:
    """Reuse Vidal windows and enrich each window with CMU provenance."""
    if tuple(FEATURES) != ("voltage", "current", "temperature"):
        raise RuntimeError("Unexpected Vidal feature order")
    windows = build_case1_windows(records, seq_len, stride, keep_incomplete)
    by_order = {int(r.metadata["sequence_order"]): r for r in records}
    for metadata in windows.metadata:
        record = by_order[int(metadata["sequence_order"])]
        frame = record.data
        start, end = int(metadata["window_start"]), int(metadata["window_end"])
        metadata.update(
            {
                "cell_id": record.metadata["cell_id"],
                "mission_cycle_id": record.metadata["mission_cycle_id"],
                "development_candidate_id": record.metadata["development_candidate_id"],
                "elapsed_candidate_s_start": float(frame["timestamp"].iloc[start]),
                "elapsed_candidate_s_end": float(frame["timestamp"].iloc[end - 1]),
                "original_time_s_start": float(frame["original_time_s"].iloc[start])
                if "original_time_s" in frame
                else float(frame["time_s"].iloc[start]),
                "original_time_s_end": float(frame["original_time_s"].iloc[end - 1])
                if "original_time_s" in frame
                else float(frame["time_s"].iloc[end - 1]),
                "source_row_left": int(frame["source_row_left"].iloc[start])
                if "source_row_left" in frame
                else int(frame["source_row"].iloc[start]),
                "source_row_right": int(frame["source_row_right"].iloc[end - 1])
                if "source_row_right" in frame
                else int(frame["source_row"].iloc[end - 1]),
            }
        )
    return windows


def transform_cmu_records(
    records: list[VidalV2Record], normalizer: TransformOnlyNormalizer
) -> list[VidalV2Record]:
    """Apply an already-fitted source normalizer; fitting is intentionally absent."""
    transformed = []
    for record in records:
        frame = record.data.copy()
        frame.loc[:, list(FEATURES)] = normalizer.transform(
            frame[list(FEATURES)].to_numpy(dtype=np.float32)
        )
        transformed.append(VidalV2Record(frame, record.metadata))
    return transformed
