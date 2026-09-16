"""Development-only CMU raw-to-candidate-to-SOC pipeline v0."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from battery_benchmark.datasets.cmu_evtol.ns_phase_labeling_v1 import (
    MISSION_FLIGHT_NS_SEQUENCE,
    MISSION_FLIGHT_PHASE_ROLES,
)
from battery_benchmark.datasets.cmu_evtol.raw import (
    CANONICAL_COLUMNS,
    load_cmu_v3_main_file,
    sha256_file,
)
from battery_benchmark.datasets.cmu_evtol.soc import (
    LABEL_SOURCE,
    REFERENCE_CAPACITY_AH,
    integrate_cmu_candidate_soc,
    validate_cmu_reference_capacity,
)
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty


BASELINE_CELLS = ("VAH01", "VAH17", "VAH27")
SUPPORTED_CELLS = (*BASELINE_CELLS, "VAH30", "VAH15", "VAH12", "VAH11")
PIPELINE_STATUS = "development_only"
# Renamed from the old ("high_1", "lower", "high_2") power-heuristic roles:
# a mission candidate is now detected by an EXACT Ns sequence match
# (take_off=4, cruise=5, landing=6; see ns_phase_labeling_v1), so the role
# names are the actual protocol phase names rather than power-shape guesses.
PHASE_ROLES = MISSION_FLIGHT_PHASE_ROLES
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


def enumerate_diagnostic_candidates(runs: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    """Enumerate exact take_off/cruise/landing (Ns 4,5,6) mission candidates.

    Detection is purely Ns-sequence based on the run table
    ``contiguous_ns_runs`` already produces: a candidate is three
    POSITIONALLY ADJACENT rows of ``runs`` -- mirroring the old code's
    ``runs.iloc[index:index+3]`` -- within a single ``continuous_block_id``,
    whose ``ns_value`` sequence is exactly ``MISSION_FLIGHT_NS_SEQUENCE``
    (4, 5, 6). ``Ns`` is a stable, protocol-level tester column (see
    ``ns_phase_labeling_v1``), so this match is exact rather than
    thresholded, and each run may be claimed by at most one candidate
    (mirrors the old code's ``claimed_runs`` non-overlap guard).

    Because the match is exact, there is no longer an ambiguous "was this
    candidate but had wrong power/duration" case to retain the old
    fine-grained hard/soft rejection-reason taxonomy for: only genuine
    (4, 5, 6) matches are enumerated here. Non-matching runs simply produce
    no manifest row (see the module docstring / task notes for why this is
    the simpler and more honest choice for exact Ns matching).

    ``protocol`` is still required and still accepted positionally exactly
    as before, so ``rpt.py``'s call site and ``load_diagnostic_protocol``'s
    YAML-loading/governance contract (``implementation_authorized is
    False``) keep working unchanged; its ``diagnostic_screening_thresholds``
    content is simply no longer consulted, since Ns-sequence matching needs
    no power/duration thresholds.
    """
    del protocol
    records: list[dict[str, object]] = []
    claimed_runs: set[int] = set()
    run_count = len(runs)
    for index in range(max(run_count - 2, 0)):
        if index in claimed_runs:
            continue
        triple = runs.iloc[index : index + 3]
        first, middle, last = triple.itertuples(index=False)
        if (first.ns_value, middle.ns_value, last.ns_value) != MISSION_FLIGHT_NS_SEQUENCE:
            continue
        if len({first.continuous_block_id, middle.continuous_block_id, last.continuous_block_id}) != 1:
            continue
        triple_indices = range(index, index + 3)
        if claimed_runs.intersection(triple_indices):
            continue
        claimed_runs.update(triple_indices)

        primary_reason = "ns_sequence_take_off_cruise_landing_matched"
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
            "screening_status": "diagnostic_screening_match",
            "primary_reason": primary_reason,
            "all_reason_codes": json.dumps([primary_reason], separators=(",", ":")),
        }
        for position, role in enumerate(PHASE_ROLES):
            run = triple.iloc[position]
            record[f"{role}_source_row_start"] = int(run.source_row_start)
            record[f"{role}_source_row_end"] = int(run.source_row_end)
            record[f"{role}_duration_s"] = float(run.duration_s)
            record[f"{role}_median_power_W"] = float(run.discharge_power_median_W)
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
    if cell_id not in SUPPORTED_CELLS:
        raise ValueError(f"CMU pipeline v0 does not support cell: {cell_id}")
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
    invalid = sorted(set(cells) - set(SUPPORTED_CELLS))
    if invalid:
        raise ValueError(f"CMU pipeline v0 rejects non-baseline or unsupported cells: {invalid}")
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
