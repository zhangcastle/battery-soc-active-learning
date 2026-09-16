"""Tests for the Ns-exact-sequence enumerate_diagnostic_candidates rewrite."""

from __future__ import annotations

import pandas as pd
import pytest

from battery_benchmark.datasets.cmu_evtol.development_pipeline_v0 import (
    PHASE_ROLES,
    build_candidate_rows,
    contiguous_ns_runs,
    enumerate_diagnostic_candidates,
)

CELL_ID = "VAH01"

# A minimal, draft (unauthorized-for-real-use) protocol dict. The new
# enumerate_diagnostic_candidates no longer reads diagnostic_screening_
# thresholds, but the parameter is still required/accepted, so an (almost)
# empty dict exercises that it truly is unused.
PROTOCOL: dict = {"diagnostic_screening_thresholds": {}}


def _segment(ns_value: int, n_rows: int, *, block_id: int, start_row: int, start_time: float) -> pd.DataFrame:
    """Build ``n_rows`` synthetic canonical rows of one Ns segment."""
    rows = []
    for offset in range(n_rows):
        rows.append(
            {
                "cell_id": CELL_ID,
                "continuous_block_id": block_id,
                "source_row": start_row + offset,
                "time_s": start_time + offset,
                "Ecell_V": 3.8,
                "I_mA": -1000.0,
                "cycleNumber": 1,
                "Ns": ns_value,
            }
        )
    return pd.DataFrame(rows)


def _build_canonical(segments: list[tuple[int, int, int]]) -> pd.DataFrame:
    """``segments`` is a list of (ns_value, n_rows, block_id); rows/time are
    laid out contiguously in the order given."""
    frames = []
    row = 1
    time = 0.0
    for ns_value, n_rows, block_id in segments:
        frames.append(_segment(ns_value, n_rows, block_id=block_id, start_row=row, start_time=time))
        row += n_rows
        time += n_rows
    return pd.concat(frames, ignore_index=True)


def test_clean_triple_is_matched_with_correct_roles_and_row_ranges():
    canonical = _build_canonical(
        [
            (0, 3, 1),  # cc_charge (not part of a candidate)
            (4, 2, 1),  # take_off
            (5, 5, 1),  # cruise
            (6, 3, 1),  # landing
            (7, 2, 1),  # rest_1 (not part of a candidate)
        ]
    )
    runs = contiguous_ns_runs(canonical)

    manifest = enumerate_diagnostic_candidates(runs, PROTOCOL)

    assert len(manifest) == 1
    row = manifest.iloc[0]
    assert row["screening_status"] == "diagnostic_screening_match"
    # take_off = rows 4-5, cruise = rows 6-10, landing = rows 11-13
    assert row["take_off_source_row_start"] == 4
    assert row["take_off_source_row_end"] == 5
    assert row["cruise_source_row_start"] == 6
    assert row["cruise_source_row_end"] == 10
    assert row["landing_source_row_start"] == 11
    assert row["landing_source_row_end"] == 13
    assert row["start_source_row"] == 4
    assert row["end_source_row"] == 13


def test_ns_sequence_broken_by_unrelated_ns_is_not_matched():
    # 4, 8, 5, 6 -- the stray/RPT-artifact Ns=8 breaks the exact triple.
    canonical = _build_canonical(
        [
            (4, 2, 1),
            (8, 1, 1),
            (5, 5, 1),
            (6, 3, 1),
        ]
    )
    runs = contiguous_ns_runs(canonical)

    manifest = enumerate_diagnostic_candidates(runs, PROTOCOL)

    assert manifest.empty


def test_two_consecutive_valid_triples_both_matched_without_double_claim():
    canonical = _build_canonical(
        [
            (4, 2, 1),
            (5, 5, 1),
            (6, 3, 1),
            (4, 2, 1),
            (5, 5, 1),
            (6, 3, 1),
        ]
    )
    runs = contiguous_ns_runs(canonical)

    manifest = enumerate_diagnostic_candidates(runs, PROTOCOL)

    assert len(manifest) == 2
    assert (manifest["screening_status"] == "diagnostic_screening_match").all()
    # Non-overlapping: second candidate starts strictly after first ends.
    assert manifest.iloc[1]["start_source_row"] > manifest.iloc[0]["end_source_row"]
    # No run claimed by both candidates.
    ranges = [
        set(range(r["start_source_row"], r["end_source_row"] + 1))
        for _, r in manifest.iterrows()
    ]
    assert ranges[0].isdisjoint(ranges[1])


def test_triple_split_across_continuous_blocks_is_not_matched():
    canonical = pd.concat(
        [
            _segment(4, 2, block_id=1, start_row=1, start_time=0.0),
            _segment(5, 5, block_id=1, start_row=3, start_time=2.0),
            _segment(6, 3, block_id=2, start_row=8, start_time=7.0),
        ],
        ignore_index=True,
    )
    runs = contiguous_ns_runs(canonical)

    manifest = enumerate_diagnostic_candidates(runs, PROTOCOL)

    assert manifest.empty


def test_phase_roles_renaming_flows_into_build_candidate_rows_output():
    canonical = _build_canonical(
        [
            (0, 3, 1),
            (4, 2, 1),
            (5, 5, 1),
            (6, 3, 1),
            (7, 2, 1),
        ]
    )
    # build_candidate_rows selects the full CANONICAL_COLUMNS row set (it
    # slices canonical.loc[...] and later re-selects ROW_COLUMNS, which is
    # CANONICAL_COLUMNS plus derived fields), so the minimal
    # contiguous_ns_runs-only frame from _build_canonical needs the
    # remaining raw columns filled in with harmless placeholder values.
    canonical = canonical.assign(
        source_file="VAH01.csv",
        EnergyCharge_W_h=0.0,
        QCharge_mA_h=0.0,
        EnergyDischarge_W_h=0.0,
        QDischarge_mA_h=0.0,
        Temperature__C=25.0,
        continuous_block_boundary_reason="",
    )
    runs = contiguous_ns_runs(canonical)
    manifest = enumerate_diagnostic_candidates(runs, PROTOCOL)

    rows, updated_manifest = build_candidate_rows(canonical, manifest, capacity_ah=3.0)

    assert PHASE_ROLES == ("take_off", "cruise", "landing")
    assert set(rows["development_phase_role"].unique()) == {"take_off", "cruise", "landing"}
    assert (rows.loc[rows["Ns"] == 4, "development_phase_role"] == "take_off").all()
    assert (rows.loc[rows["Ns"] == 5, "development_phase_role"] == "cruise").all()
    assert (rows.loc[rows["Ns"] == 6, "development_phase_role"] == "landing").all()
    assert len(updated_manifest) == 1


def test_no_matches_returns_empty_manifest_with_expected_columns():
    canonical = _build_canonical([(0, 3, 1), (1, 2, 1), (7, 2, 1)])
    runs = contiguous_ns_runs(canonical)

    manifest = enumerate_diagnostic_candidates(runs, PROTOCOL)

    assert manifest.empty
    from battery_benchmark.datasets.cmu_evtol.development_pipeline_v0 import MANIFEST_COLUMNS

    assert list(manifest.columns) == list(MANIFEST_COLUMNS)
