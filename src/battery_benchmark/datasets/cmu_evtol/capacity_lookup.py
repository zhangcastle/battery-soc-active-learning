"""Strictly causal CMU capacity lookup from frozen RPT observations."""

from __future__ import annotations

import numpy as np
import pandas as pd


LOOKUP_PROTOCOL_VERSION = "cmu_evtol_causal_capacity_lookup_v1"
NOMINAL_CAPACITY_AH = 3.0
LOOKUP_COLUMNS = (
    "cell_id",
    "mission_cycle_id",
    "mission_identity_source",
    "continuous_block_id",
    "mission_start_source_row",
    "mission_start_time_s",
    "capacity_ah",
    "capacity_source",
    "source_rpt_id",
    "source_rpt_capacity_ah",
    "source_rpt_continuous_block_id",
    "source_rpt_end_source_row",
    "source_rpt_end_time_s",
    "causal_lag",
    "causal_lag_unit",
    "lookup_protocol_version",
)


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required column(s): {missing}")


def build_causal_capacity_lookup(
    missions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    nominal_capacity_ah: float = NOMINAL_CAPACITY_AH,
    protocol_version: str = LOOKUP_PROTOCOL_VERSION,
) -> pd.DataFrame:
    """Assign the latest eligible RPT completed before each mission starts.

    Ordering is per-cell source-file row provenance.  The strict inequality
    ``RPT end_source_row < mission start_source_row`` prevents future leakage.
    """
    _require_columns(
        missions,
        {
            "development_candidate_id", "cell_id", "continuous_block_id",
            "start_source_row", "start_time_s", "screening_status",
        },
        "missions",
    )
    _require_columns(
        observations,
        {
            "rpt_id", "cell_id", "detector_status", "accepted_for_capacity_lookup",
            "continuous_block_id", "end_source_row", "end_time_s", "capacity_net_ah",
        },
        "RPT observations",
    )
    if nominal_capacity_ah != NOMINAL_CAPACITY_AH:
        raise ValueError("CMU causal capacity lookup v1 requires nominal_capacity_ah=3.0")

    matched = missions.loc[
        missions["screening_status"].eq("diagnostic_screening_match")
    ].copy()
    if matched["development_candidate_id"].duplicated().any():
        raise ValueError("mission cycle IDs must be unique")

    eligible = observations.loc[
        observations["detector_status"].eq("accepted")
        & observations["accepted_for_capacity_lookup"].eq(True)  # noqa: E712
    ].copy()
    if eligible["rpt_id"].duplicated().any():
        raise ValueError("eligible RPT IDs must be unique")

    records: list[dict[str, object]] = []
    for cell_id, cell_missions in matched.groupby("cell_id", sort=False):
        cell_rpts = eligible.loc[eligible["cell_id"].eq(cell_id)].sort_values(
            "end_source_row", kind="stable"
        )
        if cell_rpts["end_source_row"].duplicated().any():
            raise ValueError(f"eligible RPT end_source_row must be unique within {cell_id}")
        end_rows = cell_rpts["end_source_row"].to_numpy(dtype=np.int64)
        for mission in cell_missions.itertuples(index=False):
            position = int(np.searchsorted(end_rows, int(mission.start_source_row), side="left") - 1)
            rpt = cell_rpts.iloc[position] if position >= 0 else None
            records.append({
                "cell_id": cell_id,
                "mission_cycle_id": mission.development_candidate_id,
                "mission_identity_source": "development_candidate_id",
                "continuous_block_id": int(mission.continuous_block_id),
                "mission_start_source_row": int(mission.start_source_row),
                "mission_start_time_s": float(mission.start_time_s),
                "capacity_ah": (
                    float(rpt.capacity_net_ah) if rpt is not None else float(nominal_capacity_ah)
                ),
                "capacity_source": "observed_rpt" if rpt is not None else "nominal_fallback",
                "source_rpt_id": rpt.rpt_id if rpt is not None else None,
                "source_rpt_capacity_ah": float(rpt.capacity_net_ah) if rpt is not None else np.nan,
                "source_rpt_continuous_block_id": (
                    int(rpt.continuous_block_id) if rpt is not None else pd.NA
                ),
                "source_rpt_end_source_row": int(rpt.end_source_row) if rpt is not None else pd.NA,
                "source_rpt_end_time_s": float(rpt.end_time_s) if rpt is not None else np.nan,
                "causal_lag": (
                    int(mission.start_source_row - rpt.end_source_row) if rpt is not None else pd.NA
                ),
                "causal_lag_unit": "source_rows",
                "lookup_protocol_version": protocol_version,
            })

    result = pd.DataFrame.from_records(records, columns=LOOKUP_COLUMNS)
    if not result.empty:
        result["source_rpt_continuous_block_id"] = result[
            "source_rpt_continuous_block_id"
        ].astype("Int64")
        result["source_rpt_end_source_row"] = result["source_rpt_end_source_row"].astype("Int64")
        result["causal_lag"] = result["causal_lag"].astype("Int64")
    return result


def validate_lookup(
    lookup: pd.DataFrame,
    missions: pd.DataFrame,
    observations: pd.DataFrame,
) -> dict[str, int]:
    """Return explicit leakage/isolation/coverage QC counts."""
    matched = missions.loc[missions["screening_status"].eq("diagnostic_screening_match")]
    rpt_by_id = observations.set_index("rpt_id", drop=False)
    observed = lookup.loc[lookup["capacity_source"].eq("observed_rpt")]
    future = cross_cell = ineligible = unmatched = 0
    for row in observed.itertuples(index=False):
        if row.source_rpt_id not in rpt_by_id.index:
            unmatched += 1
            continue
        rpt = rpt_by_id.loc[row.source_rpt_id]
        future += int(int(rpt.end_source_row) >= int(row.mission_start_source_row))
        cross_cell += int(rpt.cell_id != row.cell_id)
        ineligible += int(
            rpt.detector_status != "accepted" or not bool(rpt.accepted_for_capacity_lookup)
        )
    return {
        "unmatched_source_rpt_count": unmatched,
        "duplicate_lookup_row_count": int(lookup["mission_cycle_id"].duplicated().sum()),
        "future_leakage_count": future,
        "cross_cell_usage_count": cross_cell,
        "ineligible_rpt_usage_count": ineligible,
        "missing_mission_count": int(
            len(set(matched["development_candidate_id"]) - set(lookup["mission_cycle_id"]))
        ),
        "unexpected_mission_count": int(
            len(set(lookup["mission_cycle_id"]) - set(matched["development_candidate_id"]))
        ),
    }
