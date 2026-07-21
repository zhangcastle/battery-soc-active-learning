"""Production CMU V3 RPT detection and independent capacity integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

from battery_benchmark.datasets.cmu_evtol.development_pipeline_v0 import (
    contiguous_ns_runs,
    enumerate_diagnostic_candidates,
)
from battery_benchmark.datasets.cmu_evtol.raw import sha256_file


BASELINE_CELLS = ("VAH01", "VAH17", "VAH27")
OBSERVATION_COLUMNS = (
    "rpt_id", "cell_id", "cell_rpt_ordinal", "protocol_version",
    "detector_status", "accepted_for_capacity_lookup", "primary_reason",
    "all_reason_codes", "warning_codes", "continuous_block_id",
    "start_source_row", "end_source_row", "start_time_s", "end_time_s",
    "duration_s", "voltage_start_v", "voltage_end_v", "voltage_min_v",
    "voltage_max_v", "ns_sequence", "interruption_count",
    "near_c5_row_fraction", "charge_contamination_fraction",
    "capacity_net_ah", "capacity_discharge_only_ah",
    "capacity_qdischarge_delta_ah", "independent_vs_qdischarge_delta_ah",
    "capacity_delta_ah", "capacity_delta_fraction", "capacity_retention_fraction",
    "diagnostic_missions_before_rpt", "missions_since_previous_accepted_rpt",
    "preceding_full_charge_evidence", "preceding_rest_evidence",
    "following_rest_evidence", "following_charge_evidence",
    "candidate_ends_at_file_tail", "cycle_number_values", "cycle_number_role",
    "impedance_mapping_status", "source_file", "source_sha256",
    "asset_inventory_sha256", "generated_at_utc",
)


def load_rpt_capacity_protocol(path: str | Path) -> tuple[dict, str]:
    path = Path(path)
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol.get("capacity_lookup_authorized") is not False:
        raise ValueError("RPT capacity v1 must not authorize capacity lookup")
    if protocol.get("dynamic_soc_authorized") is not False:
        raise ValueError("RPT capacity v1 must not authorize dynamic SOC")
    return protocol, sha256_file(path)


def discover_broad_rpt_windows(runs: pd.DataFrame, thresholds: dict) -> list[tuple[int, int]]:
    """Return non-overlapping long low-rate windows within canonical blocks."""
    if runs.empty:
        return []
    mask = (
        runs["duration_s"].ge(thresholds["fragment_min_duration_s"])
        & runs["discharge_fraction"].ge(thresholds["minimum_discharge_fraction"])
        & runs["discharge_power_median_W"].between(*thresholds["discharge_power_W"])
    )
    fragments = list(runs.index[mask])
    groups: list[list[int]] = []
    for index in fragments:
        if not groups:
            groups.append([index])
            continue
        previous = groups[-1][-1]
        same_block = runs.at[index, "continuous_block_id"] == runs.at[previous, "continuous_block_id"]
        allowed_gap = thresholds["maximum_intervening_runs"] + 1
        if same_block and index - previous <= allowed_gap:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [(group[0], group[-1]) for group in groups]


def integrate_rpt_capacity(rows: pd.DataFrame) -> tuple[float, float]:
    """Integrate net removed charge and discharge-only charge, failing closed."""
    if rows.empty:
        raise ValueError("Cannot integrate an empty RPT candidate")
    time_s = rows["time_s"].to_numpy(dtype=float)
    current_a = rows["I_mA"].to_numpy(dtype=float) / 1000.0
    if not np.isfinite(time_s).all():
        raise ValueError("RPT candidate contains non-finite time_s")
    if not np.isfinite(current_a).all():
        raise ValueError("RPT candidate contains non-finite current")
    if len(time_s) < 2 or not np.all(np.diff(time_s) > 0):
        raise ValueError("RPT candidate time_s must be strictly increasing")
    net = -float(np.trapz(current_a, time_s)) / 3600.0
    discharge_only = float(np.trapz(np.maximum(-current_a, 0.0), time_s)) / 3600.0
    return net, discharge_only


def _context(
    data: pd.DataFrame,
    runs: pd.DataFrame,
    first_run: int,
    last_run: int,
    acceptance: dict,
) -> dict[str, object]:
    selected = runs.loc[first_run:last_run]
    first, last = selected.iloc[0], selected.iloc[-1]
    block_id = first.continuous_block_id
    preceding = runs.loc[max(0, first_run - 3) : first_run - 1]
    preceding = preceding.loc[preceding["continuous_block_id"].eq(block_id)]
    following = runs.loc[last_run + 1 : last_run + 3]
    following = following.loc[following["continuous_block_id"].eq(block_id)]
    full_charge = bool(
        not preceding.empty
        and preceding[["start_voltage_V", "end_voltage_V"]].max().max()
        >= acceptance["full_charge_voltage_V"]
        and preceding["charge_fraction"].max() >= 0.5
    )
    preceding_rest = bool(not preceding.empty and preceding["zero_current_fraction"].max() >= 0.5)
    following_rest = bool(not following.empty and following["zero_current_fraction"].max() >= 0.5)
    following_charge = bool(not following.empty and following["charge_fraction"].max() >= 0.5)
    trailing = data.loc[data["source_row"].gt(last.source_row_end), "I_mA"]
    at_tail = bool(
        trailing.empty
        or (trailing.abs() <= acceptance["file_tail_rest_max_abs_current_A"] * 1000).all()
    )
    short = selected.loc[selected["duration_s"].le(acceptance["interruption_max_duration_s"])]
    interruptions = short.loc[
        short.apply(
            lambda run: abs(data.loc[
                data["source_row"].between(run.source_row_start, run.source_row_end), "I_mA"
            ].median())
            <= acceptance["interruption_max_abs_current_A"] * 1000,
            axis=1,
        )
    ]
    return {
        "selected_runs": selected,
        "preceding_full_charge_evidence": full_charge,
        "preceding_rest_evidence": preceding_rest,
        "following_rest_evidence": following_rest,
        "following_charge_evidence": following_charge,
        "candidate_ends_at_file_tail": at_tail,
        "interruption_count": len(interruptions),
    }


def _ordered(protocol: dict, reasons: set[str], status: str) -> list[str]:
    return [code for code in protocol["reason_order"][status] if code in reasons]


def _classify(protocol: dict, rejected: set[str], review: set[str]) -> tuple[str, str, list[str]]:
    triggered = rejected | review
    for status in protocol["status_precedence"]:
        if status == "accepted":
            break
        ordered = _ordered(protocol, triggered, status)
        if ordered:
            all_ordered = [
                code
                for configured_status in ("rejected", "review_required")
                for code in protocol["reason_order"][configured_status]
                if code in triggered
            ]
            return status, ordered[0], all_ordered
    return "accepted", "accepted_protocol_complete", []


def _candidate_observation(
    data: pd.DataFrame,
    runs: pd.DataFrame,
    first_run: int,
    last_run: int,
    protocol: dict,
    *,
    source_sha256: str,
    asset_inventory_sha256: str,
    generated_at_utc: str,
) -> dict[str, object]:
    acceptance = protocol["acceptance"]
    context = _context(data, runs, first_run, last_run, acceptance)
    selected = context.pop("selected_runs")
    first, last = selected.iloc[0], selected.iloc[-1]
    rows = data.loc[data["source_row"].between(first.source_row_start, last.source_row_end)].copy()
    time_s = rows["time_s"].to_numpy(dtype=float)
    current_a = rows["I_mA"].to_numpy(dtype=float) / 1000.0
    rejected: set[str] = set()
    review: set[str] = set()
    warnings: set[str] = set()

    if rows["continuous_block_id"].nunique() != 1:
        rejected.add("crosses_continuous_block")
    source_rows = rows["source_row"].to_numpy(dtype=float)
    if len(source_rows) < 2 or not np.all(np.diff(source_rows) > 0):
        rejected.add("source_row_not_strictly_increasing")
    if not np.isfinite(time_s).all():
        rejected.add("non_finite_time")
    if len(time_s) < 2 or (np.isfinite(time_s).all() and not np.all(np.diff(time_s) > 0)):
        rejected.add("time_not_strictly_increasing")
    if not np.isfinite(current_a).all():
        rejected.add("non_finite_current")
    duration = float(time_s[-1] - time_s[0]) if len(time_s) else np.nan
    if not np.isfinite(duration) or duration <= 0:
        rejected.add("zero_or_negative_duration")

    capacity_net = capacity_discharge = np.nan
    if not rejected.intersection({"non_finite_time", "time_not_strictly_increasing", "non_finite_current", "zero_or_negative_duration"}):
        capacity_net, capacity_discharge = integrate_rpt_capacity(rows)
    if not np.isfinite(capacity_net) or capacity_net < acceptance["minimum_rejected_capacity_Ah"]:
        rejected.add("insufficient_discharge_capacity")

    charge_fraction = float(np.mean(current_a > 0.1)) if len(current_a) else np.nan
    near_c5 = float(np.mean(
        (current_a >= acceptance["near_c5_current_A"][0])
        & (current_a <= acceptance["near_c5_current_A"][1])
    )) if len(current_a) else np.nan
    ns_sequence = selected["ns_value"].tolist()
    voltage_start = float(rows["Ecell_V"].iloc[0]) if len(rows) else np.nan
    voltage_min = float(rows["Ecell_V"].min()) if len(rows) else np.nan

    if charge_fraction > acceptance["charge_contamination_reject_row_fraction"]:
        rejected.add("severe_charge_contamination")
    if duration < acceptance["minimum_candidate_duration_s"]:
        review.add("candidate_duration_below_minimum")
    if not context["preceding_full_charge_evidence"]:
        review.add("missing_preceding_full_charge")
    if not context["preceding_rest_evidence"]:
        review.add("missing_preceding_rest")
    if voltage_start < acceptance["start_voltage_min_V"]:
        review.add("start_voltage_below_full_region")
    if ns_sequence != acceptance["exact_ns_sequence"]:
        review.add("unexpected_ns_sequence")
    discharge_segments_valid = True
    interruption_segments_valid = True
    for run in selected.itertuples():
        segment = rows.loc[rows["source_row"].between(run.source_row_start, run.source_row_end)]
        segment_current = segment["I_mA"].to_numpy(dtype=float) / 1000.0
        if run.ns_value in acceptance["discharge_ns_values"]:
            discharge_segments_valid &= bool(
                len(segment_current)
                and np.mean(
                    (segment_current >= acceptance["near_c5_current_A"][0])
                    & (segment_current <= acceptance["near_c5_current_A"][1])
                ) >= acceptance["near_c5_minimum_row_fraction"]
            )
        elif run.ns_value in acceptance["interruption_ns_values"]:
            interruption_segments_valid &= bool(
                run.duration_s <= acceptance["interruption_max_duration_s"]
                and len(segment_current)
                and abs(np.median(segment_current)) <= acceptance["interruption_max_abs_current_A"]
            )
    if not discharge_segments_valid:
        review.add("discharge_segment_protocol_mismatch")
    if not interruption_segments_valid:
        review.add("interruption_segment_protocol_mismatch")
    if context["interruption_count"] != acceptance["interruption_count"]:
        review.add("interruption_count_mismatch")
    if near_c5 < acceptance["near_c5_minimum_row_fraction"]:
        review.add("not_predominantly_near_c5")
    if charge_fraction > acceptance["charge_contamination_maximum_row_fraction"]:
        review.add("charge_contamination_above_limit")
    if voltage_min > acceptance["cutoff_voltage_max_V"]:
        review.add("cutoff_not_reached")
    if not acceptance["capacity_physical_range_Ah"][0] <= capacity_net <= acceptance["capacity_physical_range_Ah"][1]:
        review.add("capacity_outside_physical_range")
    if context["candidate_ends_at_file_tail"] and not context["following_charge_evidence"]:
        warnings.add("file_tail_without_following_charge")

    q = rows["QDischarge_mA_h"].to_numpy(dtype=float)
    q_delta = float(q[-1] - q[0]) / 1000.0 if len(q) and np.isfinite(q[[0, -1]]).all() else np.nan
    if len(q) > 1 and np.any(np.diff(q) < -1):
        warnings.add("qdischarge_counter_reset")
    net_vs_discharge = capacity_discharge - capacity_net
    independent_vs_q = capacity_net - q_delta
    qc = protocol["capacity_qc"]
    if np.isfinite(net_vs_discharge) and abs(net_vs_discharge) > qc["net_vs_discharge_only_warning_Ah"]:
        warnings.add("net_vs_discharge_only_mismatch")
    if np.isfinite(independent_vs_q) and abs(independent_vs_q) > qc["independent_vs_qdischarge_warning_Ah"]:
        warnings.add("independent_vs_qdischarge_mismatch")

    status, primary, reasons = _classify(protocol, rejected, review)
    warning_order = [code for code in protocol["reason_order"]["warnings"] if code in warnings]
    rpt_id = (
        f"cmu_v3_rpt_capacity_v1:{first.cell_id}:b{int(first.continuous_block_id)}:"
        f"s{int(first.source_row_start)}:e{int(last.source_row_end)}"
    )
    return {
        "rpt_id": rpt_id,
        "cell_id": first.cell_id,
        "cell_rpt_ordinal": 0,
        "protocol_version": protocol["protocol_version"],
        "detector_status": status,
        "accepted_for_capacity_lookup": status == "accepted",
        "primary_reason": primary,
        "all_reason_codes": json.dumps([*reasons, *warning_order], separators=(",", ":")),
        "warning_codes": json.dumps(warning_order, separators=(",", ":")),
        "continuous_block_id": int(first.continuous_block_id),
        "start_source_row": int(first.source_row_start),
        "end_source_row": int(last.source_row_end),
        "start_time_s": float(time_s[0]),
        "end_time_s": float(time_s[-1]),
        "duration_s": duration,
        "voltage_start_v": voltage_start,
        "voltage_end_v": float(rows["Ecell_V"].iloc[-1]),
        "voltage_min_v": voltage_min,
        "voltage_max_v": float(rows["Ecell_V"].max()),
        "ns_sequence": json.dumps(ns_sequence, separators=(",", ":")),
        "interruption_count": int(context["interruption_count"]),
        "near_c5_row_fraction": near_c5,
        "charge_contamination_fraction": charge_fraction,
        "capacity_net_ah": capacity_net,
        "capacity_discharge_only_ah": capacity_discharge,
        "capacity_qdischarge_delta_ah": q_delta,
        "independent_vs_qdischarge_delta_ah": independent_vs_q,
        "capacity_delta_ah": np.nan,
        "capacity_delta_fraction": np.nan,
        "capacity_retention_fraction": np.nan,
        "diagnostic_missions_before_rpt": 0,
        "missions_since_previous_accepted_rpt": np.nan,
        **context,
        "cycle_number_values": json.dumps(pd.unique(rows["cycleNumber"]).tolist(), separators=(",", ":")),
        "cycle_number_role": "auxiliary_observation_only_not_identity_or_join_key",
        "impedance_mapping_status": "ambiguous_auxiliary_only_no_direct_join",
        "source_file": rows["source_file"].iloc[0],
        "source_sha256": source_sha256,
        "asset_inventory_sha256": asset_inventory_sha256,
        "generated_at_utc": generated_at_utc,
    }


def detect_rpt_capacity_observations(
    data: pd.DataFrame,
    protocol: dict,
    *,
    mission_protocol: dict | None = None,
    source_sha256: str = "",
    asset_inventory_sha256: str = "",
    generated_at_utc: str = "",
) -> pd.DataFrame:
    """Discover, classify, integrate, and trend-QC one canonical baseline cell."""
    if data.empty:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)
    cell_ids = pd.unique(data["cell_id"])
    if len(cell_ids) != 1 or cell_ids[0] not in BASELINE_CELLS:
        raise ValueError(f"RPT capacity v1 supports one baseline cell at a time: {cell_ids.tolist()}")
    runs = contiguous_ns_runs(data)
    windows = discover_broad_rpt_windows(runs, protocol["broad_discovery"])
    observations = [
        _candidate_observation(
            data, runs, first, last, protocol,
            source_sha256=source_sha256,
            asset_inventory_sha256=asset_inventory_sha256,
            generated_at_utc=generated_at_utc,
        )
        for first, last in windows
    ]
    result = pd.DataFrame(observations, columns=OBSERVATION_COLUMNS)
    if result.empty:
        return result

    if mission_protocol is not None:
        mission_rows = enumerate_diagnostic_candidates(runs, mission_protocol)
        mission_rows = mission_rows.loc[
            mission_rows["screening_status"].eq("diagnostic_screening_match"), "end_source_row"
        ].sort_values().to_numpy()
        result["diagnostic_missions_before_rpt"] = [
            int(np.searchsorted(mission_rows, row, side="left")) for row in result["start_source_row"]
        ]

    result = result.sort_values("start_source_row").reset_index(drop=True)
    result["cell_rpt_ordinal"] = np.arange(1, len(result) + 1)
    qc = protocol["capacity_qc"]
    previous_accepted_missions: int | None = None
    previous_valid_capacity: float | None = None
    initial_valid_capacity: float | None = None
    for index, row in result.iterrows():
        warnings = json.loads(row["warning_codes"])
        reasons = [code for code in json.loads(row["all_reason_codes"]) if code not in warnings]
        trend_reviews: list[str] = []
        if row.detector_status == "accepted" and initial_valid_capacity is None and not (
            qc["initial_capacity_expected_range_Ah"][0]
            <= row.capacity_net_ah
            <= qc["initial_capacity_expected_range_Ah"][1]
        ):
            trend_reviews.append("initial_capacity_outside_expected_range")
        delta = (
            row.capacity_net_ah - previous_valid_capacity
            if row.detector_status == "accepted" and previous_valid_capacity is not None
            else np.nan
        )
        delta_fraction = delta / previous_valid_capacity if np.isfinite(delta) else np.nan
        result.at[index, "capacity_delta_ah"] = delta
        result.at[index, "capacity_delta_fraction"] = delta_fraction
        if row.detector_status == "accepted" and initial_valid_capacity is not None:
            result.at[index, "capacity_retention_fraction"] = row.capacity_net_ah / initial_valid_capacity
        if np.isfinite(delta) and delta > qc["small_rebound_warning_Ah"]:
            warnings.append("small_capacity_rebound")
        if np.isfinite(delta_fraction) and abs(delta_fraction) > qc["adjacent_change_review_fraction"]:
            trend_reviews.append("adjacent_capacity_jump")
        if trend_reviews and row.detector_status == "accepted":
            result.at[index, "detector_status"] = "review_required"
            result.at[index, "accepted_for_capacity_lookup"] = False
            result.at[index, "primary_reason"] = trend_reviews[0]
        reasons.extend(trend_reviews)
        if result.at[index, "detector_status"] == "accepted":
            if initial_valid_capacity is None:
                initial_valid_capacity = float(row.capacity_net_ah)
                result.at[index, "capacity_retention_fraction"] = 1.0
            previous_valid_capacity = float(row.capacity_net_ah)
            missions = int(row.diagnostic_missions_before_rpt)
            spacing = missions - previous_accepted_missions if previous_accepted_missions is not None else np.nan
            result.at[index, "missions_since_previous_accepted_rpt"] = spacing
            if np.isfinite(spacing) and spacing != qc["expected_internal_mission_spacing"] and not row.candidate_ends_at_file_tail:
                warnings.append("mission_spacing_not_50")
            previous_accepted_missions = missions
        warnings = list(dict.fromkeys(warnings))
        reasons = list(dict.fromkeys([*reasons, *warnings]))
        result.at[index, "warning_codes"] = json.dumps(warnings, separators=(",", ":"))
        result.at[index, "all_reason_codes"] = json.dumps(reasons, separators=(",", ":"))
    return result.loc[:, OBSERVATION_COLUMNS]
