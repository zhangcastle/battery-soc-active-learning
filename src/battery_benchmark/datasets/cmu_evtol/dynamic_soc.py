"""Development-only CMU SOC using the frozen causal capacity lookup v1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from battery_benchmark.datasets.cmu_evtol.capacity_lookup import validate_lookup
from battery_benchmark.datasets.cmu_evtol.development_pipeline_v0 import build_candidate_rows
from battery_benchmark.datasets.cmu_evtol.raw import CANONICAL_COLUMNS


DYNAMIC_SOC_PROTOCOL_VERSION = "cmu_evtol_dynamic_capacity_soc_development_v1"
SOC_ABS_TOLERANCE = 1e-12
LABEL_SOURCE = "cmu_causal_dynamic_capacity_reference_development_v1"
KEY = ["cell_id", "mission_cycle_id"]
RPT_PROVENANCE_COLUMNS = (
    "capacity_source", "source_rpt_id", "source_rpt_capacity_ah",
    "source_rpt_continuous_block_id", "source_rpt_end_source_row",
    "source_rpt_end_time_s", "lookup_protocol_version",
)
COMPARISON_COLUMNS = (
    "cell_id", "mission_cycle_id", "source_row", "soc_fixed_3ah", "soc_dynamic",
    "soc_difference", "abs_soc_difference", "capacity_ah",
    "capacity_difference_from_3ah", "capacity_source", "source_rpt_id",
)
DYNAMIC_OUTPUT_COLUMNS = (
    *CANONICAL_COLUMNS,
    "development_candidate_id", "mission_cycle_id", "mission_identity_source",
    "development_phase_index", "development_phase_role", "elapsed_candidate_s",
    "current_a", "voltage_v", "soc_initial", "soc_dynamic", "capacity_mode",
    "capacity_ah", "capacity_source", "source_rpt_id", "source_rpt_capacity_ah",
    "source_rpt_continuous_block_id", "source_rpt_end_source_row",
    "source_rpt_end_time_s", "causal_lag", "causal_lag_unit",
    "lookup_protocol_version", "dynamic_soc_protocol_version", "label_source",
    "label_is_ground_truth", "formal_benchmark_eligible", "qc_flags",
)


def build_dynamic_soc_rows(
    canonical: pd.DataFrame,
    candidate_manifest: pd.DataFrame,
    lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Build dynamic SOC rows; fixed v0 remains the sole integration implementation."""
    matched = candidate_manifest.loc[
        candidate_manifest["screening_status"].eq("diagnostic_screening_match")
    ].copy()
    matched["mission_cycle_id"] = matched["development_candidate_id"]
    target_keys = set(map(tuple, matched[["cell_id", "mission_cycle_id"]].to_numpy()))
    lookup_keys = set(map(tuple, lookup[KEY].to_numpy()))
    duplicate_count = int(lookup.duplicated(KEY).sum())
    cross_cell_count = sum(
        1 for cell_id, mission_id in target_keys
        if mission_id in set(lookup.loc[lookup["cell_id"].ne(cell_id), "mission_cycle_id"])
        and (cell_id, mission_id) not in lookup_keys
    )
    join_qc = {
        "unmatched_lookup_count": len(target_keys - lookup_keys),
        "duplicate_lookup_match_count": duplicate_count,
        "unused_lookup_count": len(lookup_keys - target_keys),
        "cross_cell_lookup_count": cross_cell_count,
    }
    if any(join_qc.values()):
        raise ValueError(f"Dynamic SOC lookup join failed closed: {join_qc}")

    fixed, _ = build_candidate_rows(canonical, candidate_manifest, capacity_ah=3.0)
    fixed = fixed.copy()
    fixed["mission_cycle_id"] = fixed["development_candidate_id"]
    joined = fixed.merge(lookup, on=KEY, how="left", validate="many_to_one", suffixes=("", "_lookup"))
    if joined["capacity_ah_lookup"].isna().any():
        raise ValueError("Dynamic SOC lookup contains missing capacity")

    outputs = []
    for _, rows in joined.groupby(KEY, sort=False):
        capacity = float(rows["capacity_ah_lookup"].iloc[0])
        if not np.isfinite(capacity) or capacity <= 0:
            raise ValueError("Dynamic SOC lookup capacity must be finite and positive")
        fixed_soc = rows["soc_reference_v0"]
        dynamic_soc = (
            fixed_soc.copy()
            if capacity == 3.0
            else 1.0 + (fixed_soc - 1.0) * (3.0 / capacity)
        )
        result = rows.copy()
        result["current_a"] = result["I_mA"] / 1000.0
        result["voltage_v"] = result["Ecell_V"]
        result["soc_dynamic"] = dynamic_soc
        result["soc_initial"] = 1.0
        result["capacity_mode"] = "causal_lookup_constant_within_candidate"
        result["capacity_ah"] = capacity
        result["dynamic_soc_protocol_version"] = DYNAMIC_SOC_PROTOCOL_VERSION
        result["label_source"] = LABEL_SOURCE
        result["label_is_ground_truth"] = False
        result["formal_benchmark_eligible"] = False
        result["qc_flags"] = np.where(
            (dynamic_soc < 0) | (dynamic_soc > 1), "soc_out_of_bounds", ""
        )
        outputs.append(result)
    dynamic = pd.concat(outputs, ignore_index=True) if outputs else joined.iloc[:0].copy()

    comparison = pd.DataFrame({
        "cell_id": dynamic["cell_id"],
        "mission_cycle_id": dynamic["mission_cycle_id"],
        "source_row": dynamic["source_row"],
        "soc_fixed_3ah": dynamic["soc_reference_v0"],
        "soc_dynamic": dynamic["soc_dynamic"],
        "capacity_ah": dynamic["capacity_ah"],
        "capacity_source": dynamic["capacity_source"],
        "source_rpt_id": dynamic["source_rpt_id"],
    })
    comparison["soc_difference"] = comparison["soc_dynamic"] - comparison["soc_fixed_3ah"]
    comparison["abs_soc_difference"] = comparison["soc_difference"].abs()
    comparison["capacity_difference_from_3ah"] = comparison["capacity_ah"] - 3.0
    comparison = comparison.loc[:, COMPARISON_COLUMNS]
    return fixed, dynamic.loc[:, DYNAMIC_OUTPUT_COLUMNS], comparison, join_qc


def combine_qc(join_qc: dict[str, int], validator_qc: dict[str, int]) -> dict[str, dict[str, int]]:
    """Keep independently produced QC namespaces separate."""
    return {"join_qc": dict(join_qc), "validator_qc": dict(validator_qc)}


def qc_has_errors(qc: dict[str, object]) -> bool:
    """Return True if any nested QC count is non-zero."""
    return any(
        qc_has_errors(value) if isinstance(value, dict) else bool(value)
        for value in qc.values()
    )


def _series_mismatch_count(left: pd.Series, right: pd.Series) -> int:
    left, right = left.reset_index(drop=True), right.reset_index(drop=True)
    common = min(len(left), len(right))
    mismatch = ~(
        left.iloc[:common].eq(right.iloc[:common])
        | (left.iloc[:common].isna() & right.iloc[:common].isna())
    )
    return int(mismatch.sum()) + abs(len(left) - len(right))


def _frame_mismatch_count(left: pd.DataFrame, right: pd.DataFrame) -> int:
    left, right = left.reset_index(drop=True), right.reset_index(drop=True)
    common = min(len(left), len(right))
    equal = left.iloc[:common].eq(right.iloc[:common]) | (
        left.iloc[:common].isna() & right.iloc[:common].isna()
    )
    return int((~equal).any(axis=1).sum()) + abs(len(left) - len(right))


def validate_dynamic_soc(
    fixed: pd.DataFrame,
    dynamic: pd.DataFrame,
    comparison: pd.DataFrame,
    lookup: pd.DataFrame,
    candidate_manifest: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    raise_on_error: bool = True,
) -> dict[str, int]:
    """Independently validate joins, provenance, alignment, and SOC arithmetic."""
    lookup_qc = validate_lookup(lookup, candidate_manifest, observations)
    keys = dynamic[KEY].drop_duplicates()
    lookup_keys = lookup[KEY].drop_duplicates()
    capacity_changes = int(dynamic.groupby(KEY)["capacity_ah"].nunique().gt(1).sum())
    provenance_missing = dynamic.loc[dynamic["capacity_source"].eq("observed_rpt"), [
        "source_rpt_id", "source_rpt_capacity_ah", "source_rpt_continuous_block_id",
        "source_rpt_end_source_row", "source_rpt_end_time_s",
    ]].isna().any(axis=1).sum()
    future = int((
        dynamic.loc[dynamic["capacity_source"].eq("observed_rpt"), "source_rpt_end_source_row"]
        >= dynamic.loc[dynamic["capacity_source"].eq("observed_rpt"), "source_row"]
    ).sum())

    recalculation_mismatch = 0
    for _, rows in dynamic.groupby(KEY, sort=False):
        time_s = rows["time_s"].to_numpy(dtype=float)
        current_a = rows["I_mA"].to_numpy(dtype=float) / 1000.0
        delta_ah = np.zeros(len(rows))
        if len(rows) > 1:
            delta_ah[1:] = 0.5 * (current_a[:-1] + current_a[1:]) * np.diff(time_s) / 3600.0
        expected = 1.0 + np.cumsum(delta_ah) / float(rows["capacity_ah"].iloc[0])
        recalculation_mismatch += int((
            ~np.isclose(rows["soc_dynamic"], expected, rtol=0.0, atol=SOC_ABS_TOLERANCE)
        ).sum())

    fixed_identity = fixed[["cell_id", "mission_cycle_id", "source_row"]]
    dynamic_identity = dynamic[["cell_id", "mission_cycle_id", "source_row"]]
    comparison_identity = comparison[["cell_id", "mission_cycle_id", "source_row"]]
    current_a_mismatch = _series_mismatch_count(fixed["I_mA"] / 1000.0, dynamic["current_a"])
    voltage_v_mismatch = _series_mismatch_count(fixed["Ecell_V"], dynamic["voltage_v"])
    lookup_qc_names = {
        "unmatched_source_rpt_count": "lookup_unmatched_source_rpt_count",
        "duplicate_lookup_row_count": "lookup_duplicate_row_count",
        "future_leakage_count": "future_capacity_usage_count",
        "cross_cell_usage_count": "cross_cell_lookup_count",
        "ineligible_rpt_usage_count": "ineligible_rpt_usage_count",
        "non_latest_rpt_assignment_count": "non_latest_rpt_assignment_count",
        "capacity_value_mismatch_count": "capacity_value_mismatch_count",
        "rpt_provenance_mismatch_count": "invalid_rpt_provenance_count",
        "causal_lag_mismatch_count": "causal_lag_mismatch_count",
        "invalid_fallback_row_count": "invalid_fallback_row_count",
        "missing_mission_count": "lookup_missing_mission_count",
        "unexpected_mission_count": "lookup_unexpected_mission_count",
    }
    qc = {
        "unmatched_lookup_count": len(set(map(tuple, keys.to_numpy())) - set(map(tuple, lookup_keys.to_numpy()))),
        "duplicate_lookup_match_count": int(lookup.duplicated(KEY).sum()),
        "unused_lookup_count": len(set(map(tuple, lookup_keys.to_numpy())) - set(map(tuple, keys.to_numpy()))),
        "missing_capacity_count": int(dynamic["capacity_ah"].isna().sum()),
        "within_candidate_capacity_change_count": capacity_changes,
        "dynamic_missing_rpt_provenance_row_count": int(provenance_missing),
        "dynamic_future_provenance_row_count": future,
        "fixed_dynamic_row_count_mismatch_count": abs(len(fixed) - len(dynamic)),
        "fixed_comparison_row_count_mismatch_count": abs(len(fixed) - len(comparison)),
        "fixed_dynamic_identity_order_mismatch_count": _frame_mismatch_count(fixed_identity, dynamic_identity),
        "fixed_comparison_identity_order_mismatch_count": _frame_mismatch_count(fixed_identity, comparison_identity),
        "fixed_dynamic_time_mismatch_count": _series_mismatch_count(fixed["time_s"], dynamic["time_s"]),
        "fixed_dynamic_current_mismatch_count": _series_mismatch_count(fixed["I_mA"], dynamic["I_mA"]),
        "fixed_dynamic_current_alias_mismatch_count": current_a_mismatch,
        "fixed_dynamic_voltage_mismatch_count": _series_mismatch_count(fixed["Ecell_V"], dynamic["Ecell_V"]),
        "fixed_dynamic_voltage_alias_mismatch_count": voltage_v_mismatch,
        "fixed_comparison_soc_mismatch_count": _series_mismatch_count(
            fixed["soc_reference_v0"], comparison["soc_fixed_3ah"]
        ),
        "nonfinite_soc_count": int((~np.isfinite(dynamic["soc_dynamic"])).sum()),
        "soc_out_of_allowed_range_count": int(((dynamic["soc_dynamic"] < 0) | (dynamic["soc_dynamic"] > 1)).sum()),
        "dynamic_recalculation_mismatch_count": recalculation_mismatch,
    }
    qc.update({output_name: lookup_qc[input_name] for input_name, output_name in lookup_qc_names.items()})
    if raise_on_error and any(qc.values()):
        raise ValueError(f"Dynamic SOC validation failed closed: {qc}")
    return qc
