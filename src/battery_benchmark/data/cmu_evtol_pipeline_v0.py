"""Development-only CMU raw-to-candidate-to-SOC pipeline v0."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

from battery_benchmark.data.cmu_evtol_raw import (
    CANONICAL_COLUMNS,
    load_cmu_v3_main_file,
    sha256_file,
)
from battery_benchmark.labels.cmu_soc import (
    LABEL_SOURCE,
    REFERENCE_CAPACITY_AH,
    integrate_cmu_candidate_soc,
    validate_cmu_reference_capacity,
)
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty


BASELINE_CELLS = ("VAH01", "VAH17", "VAH27")
PIPELINE_STATUS = "development_only"
PHASE_ROLES = ("high_1", "lower", "high_2")
MANIFEST_COLUMNS = (
    "development_candidate_id", "cell_id", "continuous_block_id",
    "start_source_row", "end_source_row", "start_time_s", "end_time_s",
    "row_count", "screening_status", "primary_reason", "all_reason_codes",
    *(f"{role}_{suffix}" for role in PHASE_ROLES for suffix in (
        "source_row_start", "source_row_end", "duration_s", "median_power_W"
    )),
)
ROW_COLUMNS = (
    *CANONICAL_COLUMNS,
    "development_candidate_id", "development_phase_index",
    "development_phase_role", "elapsed_candidate_s", "capacity_mode",
    "capacity_ah", "soc_initial", "soc_raw", "soc_reference_v0",
    "label_source", "label_is_ground_truth", "formal_benchmark_eligible", "qc_flags",
)


def add_power_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["electrical_power_W"] = result["Ecell_V"] * result["I_mA"] / 1000.0
    result["discharge_power_W"] = np.maximum(-result["electrical_power_W"], 0.0)
    return result


def contiguous_ns_runs(data: pd.DataFrame) -> pd.DataFrame:
    """Describe same-Ns runs without crossing canonical continuous blocks."""
    required = {
        "cell_id", "continuous_block_id", "source_row", "time_s", "Ecell_V",
        "I_mA", "cycleNumber", "Ns",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Ns run evidence missing canonical column(s): {missing}")
    frame = add_power_columns(data)
    starts = frame["continuous_block_id"].ne(frame["continuous_block_id"].shift())
    starts |= frame["Ns"].ne(frame["Ns"].shift())
    records = []
    for _, group in frame.groupby(starts.cumsum(), sort=False):
        start_time, end_time = float(group["time_s"].iloc[0]), float(group["time_s"].iloc[-1])
        current = group["I_mA"]
        records.append({
            "cell_id": group["cell_id"].iloc[0],
            "continuous_block_id": int(group["continuous_block_id"].iloc[0]),
            "ns_value": group["Ns"].iloc[0],
            "source_row_start": int(group["source_row"].iloc[0]),
            "source_row_end": int(group["source_row"].iloc[-1]),
            "time_start_s": start_time,
            "time_end_s": end_time,
            "duration_s": end_time - start_time,
            "row_count": len(group),
            "charge_fraction": float((current > 0).mean()),
            "discharge_fraction": float((current < 0).mean()),
            "zero_current_fraction": float((current == 0).mean()),
            "electrical_power_median_W": float(group["electrical_power_W"].median()),
            "electrical_power_p10_W": float(group["electrical_power_W"].quantile(0.10)),
            "electrical_power_p90_W": float(group["electrical_power_W"].quantile(0.90)),
            "discharge_power_median_W": float(group["discharge_power_W"].median()),
            "start_voltage_V": float(group["Ecell_V"].iloc[0]),
            "end_voltage_V": float(group["Ecell_V"].iloc[-1]),
            "cycle_number_start": group["cycleNumber"].iloc[0],
            "cycle_number_end": group["cycleNumber"].iloc[-1],
        })
    runs = pd.DataFrame(records)
    runs["previous_ns"] = np.nan
    runs["next_ns"] = np.nan
    for indices in runs.groupby("continuous_block_id", sort=False).groups.values():
        ordered = list(indices)
        runs.loc[ordered, "previous_ns"] = runs.loc[ordered, "ns_value"].shift().to_numpy()
        runs.loc[ordered, "next_ns"] = runs.loc[ordered, "ns_value"].shift(-1).to_numpy()
    return runs


def load_diagnostic_protocol(path: str | Path) -> tuple[dict, str]:
    path = Path(path)
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol.get("implementation_authorized") is not False:
        raise ValueError("CMU development pipeline requires the disabled draft protocol")
    return protocol, sha256_file(path)


def _between(value: float, bounds: Sequence[float]) -> bool:
    return bool(bounds[0] <= value <= bounds[1])


def _rpt_like(run: pd.Series, thresholds: dict) -> bool:
    return all((
        _between(run.discharge_power_median_W, thresholds["rpt_like_discharge_power_W"]),
        run.duration_s >= thresholds["rpt_like_minimum_duration_s"],
        run.start_voltage_V - run.end_voltage_V >= thresholds["rpt_like_minimum_voltage_drop_V"],
        run.end_voltage_V <= thresholds["rpt_like_maximum_end_voltage_V"],
        run.discharge_fraction >= thresholds["minimum_discharge_fraction"],
    ))


def _classify_reasons(
    protocol: dict,
    hard_reasons: Iterable[str],
    uncertain_reasons: Iterable[str],
) -> tuple[str, str, str]:
    configured = protocol["classification"]["reason_codes"]
    triggered = set(hard_reasons) | set(uncertain_reasons)
    ordered = [
        code
        for status in protocol["classification"]["precedence"]
        for code in configured[status]
        if code in triggered
    ]
    status = primary = None
    for configured_status in protocol["classification"]["precedence"]:
        matching = [code for code in configured[configured_status] if code in triggered]
        if matching:
            status, primary = configured_status, matching[0]
            break
    if status is None:
        status, primary = "accept", configured["accept"][0]
        ordered = [primary]
    status = "diagnostic_screening_match" if status == "accept" else status
    return status, primary, json.dumps(ordered, separators=(",", ":"))


def enumerate_diagnostic_candidates(runs: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    """Enumerate draft high/lower/high candidates, retaining non-match reasons."""
    t = protocol["diagnostic_screening_thresholds"]
    records: list[dict[str, object]] = []
    claimed_runs: set[int] = set()
    for index, first in runs.iterrows():
        if not (
            first.discharge_fraction >= t["minimum_discharge_fraction"]
            and _between(first.discharge_power_median_W, t["high_discharge_power_W"])
            and _between(first.duration_s, t["first_high_duration_s"])
        ):
            continue
        triple = runs.iloc[index : index + 3]
        hard_reasons: list[str] = []
        uncertain_reasons: list[str] = []
        if len(triple) < 3:
            uncertain_reasons.append("missing_required_segment")
        else:
            first, middle, last = triple.itertuples(index=False)
            if len({first.continuous_block_id, middle.continuous_block_id, last.continuous_block_id}) != 1:
                hard_reasons.append("crosses_continuous_block")
            if min(first.duration_s, middle.duration_s, last.duration_s) <= 0:
                hard_reasons.append("segment_duration_zero_or_negative")
            if min(first.discharge_fraction, middle.discharge_fraction, last.discharge_fraction) < t["minimum_discharge_fraction"]:
                hard_reasons.append("severe_charge_or_rest_contamination")
            if any(_rpt_like(run, t) for run in triple.itertuples(index=False)):
                hard_reasons.append("clearly_matches_long_low_rate_diagnostic_evidence")
            if _between(middle.discharge_power_median_W, t["high_discharge_power_W"]) or _between(last.discharge_power_median_W, t["lower_discharge_power_W"]):
                hard_reasons.append("segment_order_definitively_incorrect")
            thresholds_match = all((
                _between(middle.discharge_power_median_W, t["lower_discharge_power_W"]),
                _between(last.discharge_power_median_W, t["high_discharge_power_W"]),
                _between(middle.duration_s, t["lower_duration_s"]),
                _between(last.duration_s, t["last_high_duration_s"]),
            ))
            if not thresholds_match:
                uncertain_reasons.append("power_or_duration_near_but_not_within_screening_range")
            if thresholds_match and claimed_runs.intersection(range(index, index + 3)):
                uncertain_reasons.append("overlaps_earlier_candidate")
            if not hard_reasons and not uncertain_reasons:
                claimed_runs.update(range(index, index + 3))

        status, primary_reason, all_reason_codes = _classify_reasons(
            protocol, hard_reasons, uncertain_reasons
        )
        last = triple.iloc[-1]
        record: dict[str, object] = {
            "development_candidate_id": (
                f"cmu_v3_dev_candidate_v0:{first.cell_id}:b{int(first.continuous_block_id)}:"
                f"s{int(first.source_row_start)}:e{int(last.source_row_end)}"
            ),
            "cell_id": first.cell_id,
            "continuous_block_id": int(first.continuous_block_id),
            "start_source_row": int(first.source_row_start),
            "end_source_row": int(last.source_row_end),
            "start_time_s": float(first.time_start_s),
            "end_time_s": float(last.time_end_s),
            "row_count": int(triple["row_count"].sum()),
            "screening_status": status,
            "primary_reason": primary_reason,
            "all_reason_codes": all_reason_codes,
        }
        for position, role in enumerate(PHASE_ROLES):
            run = triple.iloc[position] if position < len(triple) else None
            record[f"{role}_source_row_start"] = int(run.source_row_start) if run is not None else None
            record[f"{role}_source_row_end"] = int(run.source_row_end) if run is not None else None
            record[f"{role}_duration_s"] = float(run.duration_s) if run is not None else None
            record[f"{role}_median_power_W"] = float(run.discharge_power_median_W) if run is not None else None
        records.append(record)
    return pd.DataFrame.from_records(records, columns=MANIFEST_COLUMNS)


def build_candidate_rows(
    canonical: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    capacity_ah: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_cmu_reference_capacity(capacity_ah)
    outputs = []
    manifest = manifest.copy()
    manifest["capacity_ah"] = float(capacity_ah)
    manifest["soc_start"] = np.nan
    manifest["soc_end_raw"] = np.nan
    manifest["soc_out_of_bounds_count"] = 0
    for row_index, candidate in manifest.loc[
        manifest["screening_status"].eq("diagnostic_screening_match")
    ].iterrows():
        rows = canonical.loc[
            canonical["source_row"].between(candidate.start_source_row, candidate.end_source_row)
        ].copy()
        rows["development_candidate_id"] = candidate.development_candidate_id
        rows["development_phase_index"] = 0
        rows["development_phase_role"] = ""
        for phase_index, role in enumerate(PHASE_ROLES, start=1):
            mask = rows["source_row"].between(
                candidate[f"{role}_source_row_start"], candidate[f"{role}_source_row_end"]
            )
            rows.loc[mask, "development_phase_index"] = phase_index
            rows.loc[mask, "development_phase_role"] = role
        rows["elapsed_candidate_s"] = rows["time_s"] - rows["time_s"].iloc[0]
        rows = integrate_cmu_candidate_soc(rows, capacity_ah=capacity_ah)
        count = int(rows["qc_flags"].eq("soc_out_of_bounds").sum())
        manifest.loc[row_index, ["soc_start", "soc_end_raw", "soc_out_of_bounds_count"]] = [
            float(rows["soc_raw"].iloc[0]), float(rows["soc_raw"].iloc[-1]), count
        ]
        outputs.append(rows)
    rows = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=ROW_COLUMNS)
    return rows.loc[:, ROW_COLUMNS], manifest


def process_cmu_cell(
    main_path: str | Path,
    protocol: dict,
    protocol_sha256: str,
    *,
    capacity_ah: float = 3.0,
    repo_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    validate_cmu_reference_capacity(capacity_ah)
    result = load_cmu_v3_main_file(main_path, repo_root=repo_root)
    cell_id = result.metadata["cell_id"]
    if cell_id not in BASELINE_CELLS:
        raise ValueError(f"CMU pipeline v0 supports baseline cells only: {cell_id}")
    manifest = enumerate_diagnostic_candidates(contiguous_ns_runs(result.data), protocol)
    rows, manifest = build_candidate_rows(result.data, manifest, capacity_ah=capacity_ah)
    now = datetime.now(timezone.utc).isoformat()
    git_revision = get_full_commit_hash(repo_root)
    git_dirty = is_dirty(repo_root)
    for key, value in {
        "input_sha256": result.metadata["source_sha256"],
        "protocol_sha256": protocol_sha256,
        "git_revision": git_revision,
        "git_dirty": git_dirty,
        "generated_at_utc": now,
        "formal_benchmark_eligible": False,
    }.items():
        manifest[key] = value
    counts = manifest["screening_status"].value_counts().to_dict()
    qc = {
        "cell_id": cell_id,
        "pipeline_status": PIPELINE_STATUS,
        "formal_benchmark_eligible": False,
        "label_is_ground_truth": False,
        "capacity_mode": "constant",
        "capacity_ah": REFERENCE_CAPACITY_AH,
        "soc_initial_source": "protocol_full_charge_assumption",
        "screening_unit": "anchor_window",
        "anchor_window_status_counts": {key: int(value) for key, value in counts.items()},
        "processed_row_count": len(rows),
        "soc_out_of_bounds_count": int(rows["qc_flags"].eq("soc_out_of_bounds").sum()) if len(rows) else 0,
    }
    return rows, manifest, qc


def write_pipeline_outputs(
    raw_root: str | Path,
    output_root: str | Path,
    protocol_path: str | Path,
    *,
    asset_inventory_path: str | Path,
    cells: Iterable[str] = BASELINE_CELLS,
    capacity_ah: float = 3.0,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    cells = tuple(cells)
    validate_cmu_reference_capacity(capacity_ah)
    if len(cells) != len(set(cells)):
        raise ValueError("CMU pipeline v0 rejects duplicate cell IDs")
    invalid = sorted(set(cells) - set(BASELINE_CELLS))
    if invalid:
        raise ValueError(f"CMU pipeline v0 rejects non-baseline cells: {invalid}")
    protocol, protocol_sha = load_diagnostic_protocol(protocol_path)
    inventory = pd.read_csv(asset_inventory_path).set_index("cell_id")
    raw_root, output_root = Path(raw_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifests, summaries = [], []
    for cell_id in cells:
        rows, manifest, qc = process_cmu_cell(
            raw_root / f"{cell_id}.csv", protocol, protocol_sha,
            capacity_ah=capacity_ah, repo_root=repo_root,
        )
        expected_sha = inventory.at[cell_id, "sha256"]
        actual_sha = manifest["input_sha256"].iloc[0] if len(manifest) else sha256_file(raw_root / f"{cell_id}.csv")
        if actual_sha != expected_sha:
            raise ValueError(f"Asset inventory SHA mismatch for {cell_id}")
        rows.to_parquet(output_root / f"{cell_id}_candidate_rows.parquet", index=False)
        manifest.to_csv(output_root / f"{cell_id}_candidate_manifest.csv", index=False)
        (output_root / f"{cell_id}_qc_summary.json").write_text(
            json.dumps(qc, indent=2, sort_keys=True), encoding="utf-8"
        )
        manifests.append(manifest)
        summaries.append(qc)
    combined = pd.concat(manifests, ignore_index=True)
    combined.to_csv(output_root / "candidate_manifest.csv", index=False)
    qc_summary = {
        "pipeline_status": PIPELINE_STATUS,
        "formal_benchmark_eligible": False,
        "label_is_ground_truth": False,
        "capacity_mode": "constant",
        "capacity_ah": REFERENCE_CAPACITY_AH,
        "soc_initial_source": "protocol_full_charge_assumption",
        "screening_unit": "anchor_window",
        "cells": summaries,
    }
    (output_root / "qc_summary.json").write_text(json.dumps(qc_summary, indent=2), encoding="utf-8")
    metadata = {
        "pipeline_status": PIPELINE_STATUS,
        "formal_benchmark_eligible": False,
        "label_is_ground_truth": False,
        "segmentation_protocol_status": "draft",
        "capacity_mode": "constant",
        "capacity_ah": REFERENCE_CAPACITY_AH,
        "soc_initial": 1.0,
        "soc_initial_source": "protocol_full_charge_assumption",
        "current_sign": "positive_charge_negative_discharge",
        "integration_method": "trapezoidal",
        "screening_unit": "anchor_window",
        "label_source": LABEL_SOURCE,
        "protocol_sha256": protocol_sha,
        "git_revision": get_full_commit_hash(repo_root),
        "git_dirty": is_dirty(repo_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cells": list(cells),
    }
    (output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
