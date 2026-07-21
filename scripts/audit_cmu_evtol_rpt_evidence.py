#!/usr/bin/env python3
"""Audit broad RPT evidence in the three CMU baseline cells; emits no RPT labels."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.data.cmu_evtol_pipeline_v0 import (  # noqa: E402
    add_power_columns,
    contiguous_ns_runs,
    enumerate_diagnostic_candidates,
    load_diagnostic_protocol,
)
from battery_benchmark.data.cmu_evtol_raw import (  # noqa: E402
    LOADER_SCHEMA_VERSION,
    load_cmu_v3_main_file,
    sha256_file,
)
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty  # noqa: E402


AUDIT_CELLS = ("VAH01", "VAH17", "VAH27")
AUDIT_STATUS = "draft_evidence_only"
PRIORS = {
    "rated_capacity_ah": 3.0,
    "nominal_c5_current_a": -0.6,
    "broad_discharge_power_w": [0.2, 8.0],
    "broad_fragment_min_duration_s": 900.0,
    "near_c5_current_a": [-0.75, -0.45],
    "strong_start_voltage_v": 4.1,
    "strong_end_voltage_v": 2.55,
    "strong_discharged_ah": [2.4, 3.5],
    "strong_low_rate_fraction": 0.7,
    "resistance_interruption_max_abs_current_a": 0.15,
    "resistance_interruption_max_duration_s": 120.0,
}
AUDIT_COLUMNS = (
    "diagnostic_rpt_candidate_id", "cell_id", "continuous_block_id",
    "start_source_row", "end_source_row", "start_time_s", "end_time_s",
    "duration_s", "row_count", "audit_status", "primary_reason",
    "all_reason_codes", "supporting_evidence", "conflicting_evidence",
    "voltage_start_v", "voltage_end_v", "voltage_min_v", "voltage_max_v",
    "current_mean_a", "current_median_a", "current_std_a", "current_p05_a",
    "current_p95_a", "near_c5_row_fraction", "charge_contamination_fraction",
    "rest_fraction", "ns_sequence", "cycle_number_values",
    "cycle_number_reset_count", "cycle_number_jump_count",
    "qdischarge_start_mah", "qdischarge_end_mah", "qdischarge_reset_count",
    "diagnostic_discharged_ah", "resistance_interruption_count",
    "resistance_interruption_source_rows", "preceding_full_charge_evidence",
    "preceding_rest_evidence", "following_rest_evidence",
    "following_charge_evidence", "candidate_ends_at_file_tail", "boundary_reasons",
    "diagnostic_missions_before_candidate", "missions_since_previous_strong_candidate",
)


def _json(values: object) -> str:
    return json.dumps(values, separators=(",", ":"))


def discover_broad_rpt_windows(runs: pd.DataFrame) -> list[tuple[int, int]]:
    """Merge long low-rate fragments separated by at most one short Ns run."""
    mask = (
        runs["duration_s"].ge(PRIORS["broad_fragment_min_duration_s"])
        & runs["discharge_fraction"].ge(0.7)
        & runs["discharge_power_median_W"].between(*PRIORS["broad_discharge_power_w"])
    )
    fragments = list(runs.index[mask])
    if not fragments:
        return []
    groups: list[list[int]] = [[fragments[0]]]
    for index in fragments[1:]:
        previous = groups[-1][-1]
        same_block = runs.at[index, "continuous_block_id"] == runs.at[previous, "continuous_block_id"]
        if same_block and index - previous <= 2:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [(group[0], group[-1]) for group in groups]


def _discharged_ah(rows: pd.DataFrame) -> tuple[float, np.ndarray]:
    time_s = rows["time_s"].to_numpy(dtype=float)
    discharge_a = np.maximum(-rows["I_mA"].to_numpy(dtype=float) / 1000.0, 0.0)
    delta = np.zeros(len(rows))
    if len(rows) > 1:
        delta[1:] = 0.5 * (discharge_a[:-1] + discharge_a[1:]) * np.diff(time_s) / 3600
    cumulative = np.cumsum(delta)
    return float(cumulative[-1]) if len(cumulative) else 0.0, cumulative


def audit_candidate(
    data: pd.DataFrame,
    runs: pd.DataFrame,
    first_run: int,
    last_run: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    selected_runs = runs.loc[first_run:last_run]
    first, last = selected_runs.iloc[0], selected_runs.iloc[-1]
    rows = data.loc[data["source_row"].between(first.source_row_start, last.source_row_end)].copy()
    rows = add_power_columns(rows)
    discharged_ah, cumulative_ah = _discharged_ah(rows)
    rows["diagnostic_discharged_ah"] = cumulative_ah
    current_a = rows["I_mA"] / 1000.0
    cycle_diff = rows["cycleNumber"].diff()
    q_diff = rows["QDischarge_mA_h"].diff()
    interruptions = selected_runs.loc[
        selected_runs["duration_s"].le(PRIORS["resistance_interruption_max_duration_s"])
    ]
    interruptions = interruptions.loc[
        interruptions.apply(
            lambda run: abs(data.loc[
                data["source_row"].between(run.source_row_start, run.source_row_end), "I_mA"
            ].median())
            <= PRIORS["resistance_interruption_max_abs_current_a"] * 1000,
            axis=1,
        )
    ]
    preceding_runs = runs.loc[max(0, first_run - 3) : first_run - 1]
    following_runs = runs.loc[last_run + 1 : last_run + 3]
    preceding_full_charge = bool(
        not preceding_runs.empty
        and preceding_runs[["start_voltage_V", "end_voltage_V"]].max().max() >= 4.15
        and preceding_runs["charge_fraction"].max() >= 0.5
    )
    preceding_rest = bool(not preceding_runs.empty and preceding_runs["zero_current_fraction"].max() >= 0.5)
    following_rest = bool(not following_runs.empty and following_runs["zero_current_fraction"].max() >= 0.5)
    following_charge = bool(not following_runs.empty and following_runs["charge_fraction"].max() >= 0.5)
    start_v, end_v = float(rows["Ecell_V"].iloc[0]), float(rows["Ecell_V"].iloc[-1])
    duration = float(rows["time_s"].iloc[-1] - rows["time_s"].iloc[0])
    near_c5 = float(current_a.between(*PRIORS["near_c5_current_a"]).mean())
    charge_fraction = float(current_a.gt(0.1).mean())
    supporting, conflicting = [], []
    checks = {
        "starts_near_full_voltage": start_v >= PRIORS["strong_start_voltage_v"],
        "reaches_cutoff_region": float(rows["Ecell_V"].min()) <= PRIORS["strong_end_voltage_v"],
        "diagnostic_ah_near_rated_capacity": PRIORS["strong_discharged_ah"][0] <= discharged_ah <= PRIORS["strong_discharged_ah"][1],
        "predominantly_near_c5": near_c5 >= PRIORS["strong_low_rate_fraction"],
        "single_continuous_block": rows["continuous_block_id"].nunique() == 1,
        "low_charge_contamination": charge_fraction <= 0.02,
    }
    for reason, passed in checks.items():
        (supporting if passed else conflicting).append(reason)
    if all(checks.values()):
        status, primary = "strong_rpt_evidence", supporting[0]
    elif discharged_ah < 1.0 or duration < 3600 or charge_fraction > 0.2:
        status, primary = "non_rpt_candidate", conflicting[0]
    else:
        status, primary = "uncertain_rpt_evidence", conflicting[0]
    candidate_id = (
        f"cmu_v3_diagnostic_rpt_candidate_v0:{first.cell_id}:"
        f"b{int(first.continuous_block_id)}:s{int(first.source_row_start)}:e{int(last.source_row_end)}"
    )
    record = {
        "diagnostic_rpt_candidate_id": candidate_id,
        "cell_id": first.cell_id,
        "continuous_block_id": int(first.continuous_block_id),
        "start_source_row": int(first.source_row_start),
        "end_source_row": int(last.source_row_end),
        "start_time_s": float(rows["time_s"].iloc[0]),
        "end_time_s": float(rows["time_s"].iloc[-1]),
        "duration_s": duration,
        "row_count": len(rows),
        "audit_status": status,
        "primary_reason": primary,
        "all_reason_codes": _json([*supporting, *conflicting]),
        "supporting_evidence": _json(supporting),
        "conflicting_evidence": _json(conflicting),
        "voltage_start_v": start_v,
        "voltage_end_v": end_v,
        "voltage_min_v": float(rows["Ecell_V"].min()),
        "voltage_max_v": float(rows["Ecell_V"].max()),
        "current_mean_a": float(current_a.mean()),
        "current_median_a": float(current_a.median()),
        "current_std_a": float(current_a.std()),
        "current_p05_a": float(current_a.quantile(0.05)),
        "current_p95_a": float(current_a.quantile(0.95)),
        "near_c5_row_fraction": near_c5,
        "charge_contamination_fraction": charge_fraction,
        "rest_fraction": float(current_a.abs().le(0.1).mean()),
        "ns_sequence": _json(selected_runs["ns_value"].tolist()),
        "cycle_number_values": _json(pd.unique(rows["cycleNumber"]).tolist()),
        "cycle_number_reset_count": int(cycle_diff.lt(0).sum()),
        "cycle_number_jump_count": int(cycle_diff.abs().gt(1).sum()),
        "qdischarge_start_mah": float(rows["QDischarge_mA_h"].iloc[0]),
        "qdischarge_end_mah": float(rows["QDischarge_mA_h"].iloc[-1]),
        "qdischarge_reset_count": int(q_diff.lt(-1).sum()),
        "diagnostic_discharged_ah": discharged_ah,
        "resistance_interruption_count": len(interruptions),
        "resistance_interruption_source_rows": _json(interruptions["source_row_start"].astype(int).tolist()),
        "preceding_full_charge_evidence": preceding_full_charge,
        "preceding_rest_evidence": preceding_rest,
        "following_rest_evidence": following_rest,
        "following_charge_evidence": following_charge,
        "candidate_ends_at_file_tail": int(last.source_row_end) == int(data["source_row"].max()),
        "boundary_reasons": _json(sorted(set(filter(None, rows["continuous_block_boundary_reason"])))),
    }
    return record, rows


def audit_cell(data: pd.DataFrame, protocol: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if data.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS), {}
    runs = contiguous_ns_runs(data)
    mission_windows = enumerate_diagnostic_candidates(runs, protocol)
    mission_rows = mission_windows.loc[
        mission_windows["screening_status"].eq("diagnostic_screening_match"), "end_source_row"
    ].sort_values().to_numpy()
    records, candidate_rows = [], {}
    previous_strong_missions = 0
    for first_run, last_run in discover_broad_rpt_windows(runs):
        record, rows = audit_candidate(data, runs, first_run, last_run)
        missions_before = int(np.searchsorted(mission_rows, record["start_source_row"], side="left"))
        record["diagnostic_missions_before_candidate"] = missions_before
        record["missions_since_previous_strong_candidate"] = (
            missions_before - previous_strong_missions
            if record["audit_status"] == "strong_rpt_evidence" and records else None
        )
        if record["audit_status"] == "strong_rpt_evidence":
            previous_strong_missions = missions_before
        records.append(record)
        candidate_rows[record["diagnostic_rpt_candidate_id"]] = rows
    return pd.DataFrame(records, columns=AUDIT_COLUMNS), candidate_rows


def impedance_summary(path: Path, candidates: pd.DataFrame) -> dict[str, object]:
    impedance = pd.read_csv(path)
    cycles = impedance["cycle numbers"]
    candidate_cycles = {
        value
        for values in candidates.get("cycle_number_values", pd.Series(dtype=str))
        for value in json.loads(values)
    }
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "measurement_count": len(impedance),
        "cycle_numbers": cycles.astype(int).tolist(),
        "duplicate_cycle_number_count": int(cycles.duplicated().sum()),
        "candidate_cycle_number_overlap_count": len(candidate_cycles & set(cycles)),
        "mapping_status": "ambiguous_auxiliary_only_no_direct_join",
    }


def _plot_cell(output: Path, cell_id: str, data: pd.DataFrame, audit: pd.DataFrame, rows_by_id: dict[str, pd.DataFrame]) -> None:
    sample = data.iloc[:: max(1, len(data) // 10000)]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sample["time_s"] / 86400, sample["Ecell_V"], lw=0.5)
    for row in audit.itertuples():
        ax.axvspan(row.start_time_s / 86400, row.end_time_s / 86400, alpha=0.18)
    ax.set(title=f"{cell_id} broad RPT evidence timeline", xlabel="Elapsed file time (days)", ylabel="Voltage (V)")
    fig.tight_layout(); fig.savefig(output / f"{cell_id}_overview.png", dpi=150); plt.close(fig)

    with PdfPages(output / f"{cell_id}_candidate_diagnostics.pdf") as pdf:
        for row in audit.itertuples():
            candidate = rows_by_id[row.diagnostic_rpt_candidate_id]
            elapsed_h = (candidate["time_s"] - candidate["time_s"].iloc[0]) / 3600
            fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
            axes[0].plot(elapsed_h, candidate["Ecell_V"], lw=0.8); axes[0].set_ylabel("Voltage (V)")
            axes[1].plot(elapsed_h, candidate["I_mA"] / 1000, lw=0.8); axes[1].set_ylabel("Current (A)")
            axes[2].plot(elapsed_h, candidate["diagnostic_discharged_ah"], lw=0.8); axes[2].set_ylabel("Discharged Ah")
            axes[3].step(elapsed_h, candidate["Ns"], where="post", label="Ns")
            axes[3].step(elapsed_h, candidate["cycleNumber"], where="post", alpha=0.5, label="cycleNumber")
            axes[3].set(xlabel="Elapsed candidate time (h)", ylabel="Raw counters"); axes[3].legend()
            for source_row in json.loads(row.resistance_interruption_source_rows):
                x = float(candidate.loc[candidate["source_row"].eq(source_row), "time_s"].iloc[0] - candidate["time_s"].iloc[0]) / 3600
                for ax in axes: ax.axvline(x, color="tab:red", alpha=0.35, lw=0.8)
            fig.suptitle(f"{row.diagnostic_rpt_candidate_id}\n{row.audit_status}: {row.primary_reason}")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def run_audit(raw_root: Path, output_root: Path, *, repo_root: Path = REPO_ROOT) -> pd.DataFrame:
    protocol, protocol_sha = load_diagnostic_protocol(
        repo_root / "configs/protocols/cmu_evtol_mission_segmentation_v0_1.yaml"
    )
    assets = pd.read_csv(repo_root / "data/manifests/cmu_evtol_vtc6_asset_inventory_v3.csv").set_index("cell_id")
    output_root.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    all_candidates, summaries, inputs = [], {}, {}
    for cell_id in AUDIT_CELLS:
        result = load_cmu_v3_main_file(raw_root / f"{cell_id}.csv", repo_root=repo_root)
        if result.metadata["source_sha256"] != assets.at[cell_id, "sha256"]:
            raise ValueError(f"Asset SHA mismatch for {cell_id}")
        candidates, rows_by_id = audit_cell(result.data, protocol)
        candidates["input_sha256"] = result.metadata["source_sha256"]
        candidates["generated_at_utc"] = generated_at
        candidates.to_csv(output_root / f"{cell_id}_rpt_candidate_audit.csv", index=False)
        impedance = impedance_summary(raw_root / f"{cell_id}_impedance.csv", candidates)
        inputs[cell_id] = {
            "main_file": f"{cell_id}.csv",
            "main_sha256": result.metadata["source_sha256"],
            "impedance_file": impedance["file"],
            "impedance_sha256": impedance["sha256"],
        }
        summaries[cell_id] = {
            "audit_status_counts": candidates["audit_status"].value_counts().to_dict(),
            "diagnostic_capacity_range_ah": [float(candidates["diagnostic_discharged_ah"].min()), float(candidates["diagnostic_discharged_ah"].max())],
            "impedance": impedance,
        }
        (output_root / f"{cell_id}_summary.json").write_text(json.dumps(summaries[cell_id], indent=2), encoding="utf-8")
        _plot_cell(output_root, cell_id, result.data, candidates, rows_by_id)
        all_candidates.append(candidates)
    combined = pd.concat(all_candidates, ignore_index=True)
    combined.to_csv(output_root / "rpt_candidate_audit.csv", index=False)
    (output_root / "rpt_evidence_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    metadata = {
        "audit_status": AUDIT_STATUS,
        "article_id": "14226830",
        "dataset_version": "3",
        "raw_loader_schema_version": LOADER_SCHEMA_VERSION,
        "cells": list(AUDIT_CELLS),
        "inputs": inputs,
        "paper_derived_priors": PRIORS,
        "protocol_sha256": protocol_sha,
        "git_revision": get_full_commit_hash(repo_root),
        "git_dirty": is_dirty(repo_root),
        "generated_at_utc": generated_at,
        "cycle_number_role": "auxiliary_evidence_only",
        "formal_rpt_ids_emitted": False,
        "capacity_observations_emitted": False,
    }
    (output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return combined


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cmu_evtol_rpt_evidence_audit"))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    audit = run_audit(args.raw_root, args.output_root)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Wrote {len(audit)} broad RPT evidence rows in {time.perf_counter() - started:.2f} s (peak_rss_native={peak})")


if __name__ == "__main__":
    main()
