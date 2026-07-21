from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from battery_benchmark.datasets.cmu_evtol.capacity_lookup import (
    LOOKUP_COLUMNS,
    build_causal_capacity_lookup,
    validate_lookup,
)
from battery_benchmark.datasets.cmu_evtol.development_pipeline_v0 import (
    contiguous_ns_runs,
    enumerate_diagnostic_candidates,
    load_diagnostic_protocol,
)
from battery_benchmark.datasets.cmu_evtol.raw import load_cmu_v3_main_file, sha256_file


ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data/raw/cmu_evtol_vtc6/figshare_14226830"
OBSERVATIONS_PATH = ROOT / "outputs/cmu_evtol_rpt_capacity_v1/capacity_observations.csv"
OBSERVATIONS_SHA256 = "a54f1c99a1b684eec7679b7bb0c8ebb68c36eb34b076464a70f349cfaa5e5e1e"
MISSION_PROTOCOL_PATH = (
    ROOT / "configs/datasets/cmu_evtol/protocols/cmu_evtol_mission_segmentation_v0_1.yaml"
)


def _missions() -> pd.DataFrame:
    return pd.DataFrame({
        "development_candidate_id": ["a0", "a1", "a2", "a3", "a4", "b0"],
        "cell_id": ["VAH01"] * 5 + ["VAH17"],
        "continuous_block_id": [0, 0, 1, 1, 2, 0],
        "start_source_row": [5, 10, 11, 29, 31, 50],
        "start_time_s": [5.0, 10.0, 1.0, 19.0, 1.0, 50.0],
        "screening_status": ["diagnostic_screening_match"] * 6,
        # Deliberately reset/repeat/jump: lookup must ignore cycleNumber.
        "cycleNumber": [10, 10, 1, 900, 2, 10],
    })


def _observations() -> pd.DataFrame:
    return pd.DataFrame({
        "rpt_id": ["r0", "r1", "r_bad", "r_review", "r_b"],
        "cell_id": ["VAH01", "VAH01", "VAH01", "VAH01", "VAH17"],
        "detector_status": ["accepted", "accepted", "accepted", "review_required", "accepted"],
        "accepted_for_capacity_lookup": [True, True, False, True, True],
        "continuous_block_id": [0, 1, 1, 1, 0],
        "end_source_row": [10, 30, 20, 25, 100],
        "end_time_s": [10.0, 20.0, 10.0, 15.0, 100.0],
        "capacity_net_ah": [2.9, 2.8, 1.0, 1.1, 2.7],
    })


def test_strict_causal_step_lookup_and_cell_isolation() -> None:
    missions, observations = _missions(), _observations()
    original = observations.copy(deep=True)
    lookup = build_causal_capacity_lookup(missions, observations)

    assert tuple(lookup.columns) == LOOKUP_COLUMNS
    assert lookup["mission_cycle_id"].is_unique
    assert set(lookup["mission_cycle_id"]) == set(missions["development_candidate_id"])
    assert lookup.set_index("mission_cycle_id")["capacity_ah"].to_dict() == {
        "a0": 3.0,       # before first RPT
        "a1": 3.0,       # strict: RPT end == mission start is not usable
        "a2": 2.9,       # first mission after r0
        "a3": 2.9,       # step hold; rejected/review observations ignored
        "a4": 2.8,       # switches after r1 completes
        "b0": 3.0,       # VAH01 RPTs never leak into VAH17
    }
    indexed = lookup.set_index("mission_cycle_id")
    assert indexed.at["a2", "source_rpt_id"] == "r0"
    assert indexed.at["a3", "source_rpt_id"] == "r0"
    assert indexed.at["a4", "source_rpt_id"] == "r1"
    assert pd.isna(indexed.at["b0", "source_rpt_id"])
    assert indexed.at["a2", "source_rpt_end_source_row"] < indexed.at[
        "a2", "mission_start_source_row"
    ]
    assert not set(lookup["source_rpt_id"].dropna()) & {"r_bad", "r_review", "r_b"}
    assert not any(validate_lookup(lookup, missions, observations).values())
    pd.testing.assert_frame_equal(observations, original)


def test_cycle_number_changes_do_not_affect_lookup() -> None:
    missions = _missions()
    first = build_causal_capacity_lookup(missions, _observations())
    missions["cycleNumber"] = [1, 999, 1, 1, -4, 0]
    second = build_causal_capacity_lookup(missions, _observations())
    pd.testing.assert_frame_equal(first, second)


def test_duplicate_mission_ids_are_rejected() -> None:
    missions = _missions()
    missions.loc[1, "development_candidate_id"] = missions.loc[0, "development_candidate_id"]
    with pytest.raises(ValueError, match="mission cycle IDs must be unique within each cell"):
        build_causal_capacity_lookup(missions, _observations())


def test_validator_rebuilds_latest_rpt_and_checks_values_provenance_and_lag() -> None:
    missions, observations = _missions(), _observations()
    valid = build_causal_capacity_lookup(missions, observations)
    assert not any(validate_lookup(valid, missions, observations).values())

    older = valid.copy()
    index = older.index[older["mission_cycle_id"].eq("a4")][0]
    r0 = observations.set_index("rpt_id").loc["r0"]
    older.loc[index, [
        "capacity_ah", "source_rpt_id", "source_rpt_capacity_ah",
        "source_rpt_continuous_block_id", "source_rpt_end_source_row",
        "source_rpt_end_time_s", "causal_lag",
    ]] = [2.9, "r0", 2.9, r0.continuous_block_id, r0.end_source_row,
          r0.end_time_s, older.at[index, "mission_start_source_row"] - r0.end_source_row]
    assert validate_lookup(older, missions, observations)["non_latest_rpt_assignment_count"] == 1

    wrong_capacity = valid.copy()
    wrong_capacity.loc[index, "capacity_ah"] = 2.7
    assert validate_lookup(wrong_capacity, missions, observations)[
        "capacity_value_mismatch_count"
    ] == 1

    wrong_provenance = valid.copy()
    wrong_provenance.loc[index, "source_rpt_continuous_block_id"] = 99
    assert validate_lookup(wrong_provenance, missions, observations)[
        "rpt_provenance_mismatch_count"
    ] == 1

    wrong_lag = valid.copy()
    wrong_lag.loc[index, "causal_lag"] += 1
    assert validate_lookup(wrong_lag, missions, observations)["causal_lag_mismatch_count"] == 1


def test_validator_detects_invalid_fallback_and_uses_composite_identity() -> None:
    missions, observations = _missions(), _observations()
    missions.loc[missions["cell_id"].eq("VAH17"), "development_candidate_id"] = "a0"
    valid = build_causal_capacity_lookup(missions, observations)
    assert not valid.duplicated(["cell_id", "mission_cycle_id"]).any()
    assert not any(validate_lookup(valid, missions, observations).values())

    invalid_fallback = valid.copy()
    index = invalid_fallback.index[
        invalid_fallback["cell_id"].eq("VAH01")
        & invalid_fallback["mission_cycle_id"].eq("a2")
    ][0]
    invalid_fallback.loc[index, "capacity_ah"] = 3.0
    invalid_fallback.loc[index, "capacity_source"] = "nominal_fallback"
    invalid_fallback.loc[index, [
        "source_rpt_id", "source_rpt_capacity_ah", "source_rpt_continuous_block_id",
        "source_rpt_end_source_row", "source_rpt_end_time_s", "causal_lag",
    ]] = None
    assert validate_lookup(invalid_fallback, missions, observations)[
        "invalid_fallback_row_count"
    ] == 1

    duplicated = pd.concat([valid, valid.iloc[[0]]], ignore_index=True)
    assert validate_lookup(duplicated, missions, observations)["duplicate_lookup_row_count"] == 1


def test_frozen_52_rpt_observations_are_unchanged() -> None:
    assert OBSERVATIONS_PATH.is_file()
    assert sha256_file(OBSERVATIONS_PATH) == OBSERVATIONS_SHA256
    observations = pd.read_csv(OBSERVATIONS_PATH)
    assert len(observations) == 52
    assert observations.groupby("cell_id").size().to_dict() == {
        "VAH01": 18, "VAH17": 21, "VAH27": 13,
    }
    assert observations["detector_status"].eq("accepted").all()
    assert observations["accepted_for_capacity_lookup"].eq(True).all()  # noqa: E712


@pytest.mark.integration
def test_real_lookup_covers_all_screening_matches_without_leakage() -> None:
    if not RAW_ROOT.is_dir():
        pytest.skip("CMU Version 3 raw root is unavailable")
    protocol = load_diagnostic_protocol(MISSION_PROTOCOL_PATH)[0]
    missions = []
    for cell_id in ("VAH01", "VAH17", "VAH27"):
        canonical = load_cmu_v3_main_file(RAW_ROOT / f"{cell_id}.csv").data
        missions.append(enumerate_diagnostic_candidates(contiguous_ns_runs(canonical), protocol))
    missions = pd.concat(missions, ignore_index=True)
    observations = pd.read_csv(OBSERVATIONS_PATH)
    lookup = build_causal_capacity_lookup(missions, observations)

    assert lookup.groupby("cell_id").size().to_dict() == {
        "VAH01": 828, "VAH17": 980, "VAH27": 572,
    }
    assert len(lookup) == 2380
    assert not any(validate_lookup(lookup, missions, observations).values())
    assert (
        lookup.loc[lookup["capacity_source"].eq("observed_rpt"), "source_rpt_end_source_row"]
        .astype(int)
        .to_numpy()
        < lookup.loc[lookup["capacity_source"].eq("observed_rpt"), "mission_start_source_row"]
        .to_numpy()
    ).all()
