from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from battery_benchmark.datasets.cmu_evtol.capacity_lookup import build_causal_capacity_lookup
from battery_benchmark.datasets.cmu_evtol.development_pipeline_v0 import (
    MANIFEST_COLUMNS,
    build_candidate_rows,
    contiguous_ns_runs,
    enumerate_diagnostic_candidates,
    load_diagnostic_protocol,
)
from battery_benchmark.datasets.cmu_evtol.dynamic_soc import (
    build_dynamic_soc_rows,
    validate_dynamic_soc,
)
from battery_benchmark.datasets.cmu_evtol.raw import load_cmu_v3_main_file


ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data/raw/cmu_evtol_vtc6/figshare_14226830"
LOOKUP_PATH = ROOT / "outputs/datasets/cmu_evtol/causal_capacity_lookup_v1/capacity_lookup.csv"
RPT_PATH = ROOT / "outputs/cmu_evtol_rpt_capacity_v1/capacity_observations.csv"
MISSION_PROTOCOL_PATH = ROOT / "configs/datasets/cmu_evtol/protocols/cmu_evtol_mission_segmentation_v0_1.yaml"


def _canonical(*, current_ma: float = -1000.0, cell_id: str = "VAH01") -> pd.DataFrame:
    source_rows = [1, 2, 3, 11, 12, 13, 21, 22, 23]
    times = [0.0, 3600.0, 7200.0] * 3
    blocks = [0] * 3 + [1] * 3 + [2] * 3
    return pd.DataFrame({
        "cell_id": cell_id, "source_file": f"{cell_id}.csv", "source_row": source_rows,
        "time_s": times, "Ecell_V": 3.7, "I_mA": current_ma,
        "EnergyCharge_W_h": 0.0, "QCharge_mA_h": 0.0,
        "EnergyDischarge_W_h": 0.0, "QDischarge_mA_h": 0.0,
        "Temperature__C": 25.0, "cycleNumber": [9, 9, 9, 1, 1, 1, 800, 2, 2],
        "Ns": [4, 5, 6] * 3, "continuous_block_id": blocks,
        "continuous_block_boundary_reason": ["file_start", "", ""] * 3,
    })


def _manifest(cell_id: str = "VAH01") -> pd.DataFrame:
    records = []
    for ordinal, start in enumerate((1, 11, 21)):
        record = {
            "development_candidate_id": f"candidate-{ordinal}", "cell_id": cell_id,
            "continuous_block_id": ordinal, "start_source_row": start,
            "end_source_row": start + 2, "start_time_s": 0.0, "end_time_s": 7200.0,
            "row_count": 3, "screening_status": "diagnostic_screening_match",
            "primary_reason": "all_accept_requirements_met", "all_reason_codes": "[]",
        }
        for offset, role in enumerate(("high_1", "lower", "high_2")):
            record[f"{role}_source_row_start"] = start + offset
            record[f"{role}_source_row_end"] = start + offset
            record[f"{role}_duration_s"] = 1.0
            record[f"{role}_median_power_W"] = 40.0 if role != "lower" else 15.0
        records.append(record)
    return pd.DataFrame(records, columns=MANIFEST_COLUMNS)


def _observations() -> pd.DataFrame:
    return pd.DataFrame({
        "rpt_id": ["r0", "r1"], "cell_id": ["VAH01", "VAH01"],
        "detector_status": ["accepted", "accepted"],
        "accepted_for_capacity_lookup": [True, True],
        "continuous_block_id": [0, 1], "end_source_row": [5, 15],
        "end_time_s": [8000.0, 16000.0], "capacity_net_ah": [2.0, 4.0],
    })


def _lookup() -> pd.DataFrame:
    return build_causal_capacity_lookup(_manifest(), _observations())


def test_three_ah_lookup_is_bitwise_equal_to_fixed_v0() -> None:
    canonical, manifest, lookup = _canonical(), _manifest(), _lookup()
    lookup["capacity_ah"] = 3.0
    dynamic, comparison, _ = build_dynamic_soc_rows(canonical, manifest, lookup)
    fixed, _ = build_candidate_rows(canonical, manifest, capacity_ah=3.0)
    pd.testing.assert_series_equal(
        dynamic["soc_dynamic"], fixed["soc_reference_v0"], check_exact=True,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        comparison["soc_dynamic"], comparison["soc_fixed_3ah"], check_exact=True,
        check_names=False,
    )


def test_capacity_is_constant_per_candidate_and_changes_only_after_rpt() -> None:
    dynamic, comparison, join_qc = build_dynamic_soc_rows(_canonical(), _manifest(), _lookup())
    assert not any(join_qc.values())
    capacities = dynamic.groupby("mission_cycle_id", sort=False)["capacity_ah"].unique()
    assert [values.tolist() for values in capacities] == [[3.0], [2.0], [4.0]]
    endpoints = comparison.groupby("mission_cycle_id", sort=False).tail(1).set_index("mission_cycle_id")
    assert endpoints.at["candidate-1", "soc_dynamic"] < endpoints.at["candidate-1", "soc_fixed_3ah"]
    assert endpoints.at["candidate-2", "soc_dynamic"] > endpoints.at["candidate-2", "soc_fixed_3ah"]
    assert not any(validate_dynamic_soc(
        dynamic, comparison, _lookup(), _manifest(), _observations()
    ).values())


def test_zero_current_soc_does_not_drift_with_capacity() -> None:
    dynamic, _, _ = build_dynamic_soc_rows(_canonical(current_ma=0.0), _manifest(), _lookup())
    assert dynamic["soc_dynamic"].eq(1.0).all()


def test_missing_duplicate_and_cross_cell_lookup_fail_closed() -> None:
    canonical, manifest, lookup = _canonical(), _manifest(), _lookup()
    with pytest.raises(ValueError, match="failed closed"):
        build_dynamic_soc_rows(canonical, manifest, lookup.iloc[:-1])
    with pytest.raises(ValueError, match="failed closed"):
        build_dynamic_soc_rows(canonical, manifest, pd.concat([lookup, lookup.iloc[[0]]]))
    wrong_cell = lookup.copy(); wrong_cell.loc[0, "cell_id"] = "VAH17"
    with pytest.raises(ValueError, match="failed closed"):
        build_dynamic_soc_rows(canonical, manifest, wrong_cell)


@pytest.mark.integration
def test_real_dynamic_soc_has_complete_alignment_and_zero_qc() -> None:
    if not RAW_ROOT.is_dir():
        pytest.skip("CMU Version 3 raw root is unavailable")
    protocol = load_diagnostic_protocol(MISSION_PROTOCOL_PATH)[0]
    lookup, observations = pd.read_csv(LOOKUP_PATH), pd.read_csv(RPT_PATH)
    counts = {}
    for cell_id in ("VAH01", "VAH17", "VAH27"):
        canonical = load_cmu_v3_main_file(RAW_ROOT / f"{cell_id}.csv").data
        manifest = enumerate_diagnostic_candidates(contiguous_ns_runs(canonical), protocol)
        cell_lookup = lookup.loc[lookup["cell_id"].eq(cell_id)]
        dynamic, comparison, join_qc = build_dynamic_soc_rows(canonical, manifest, cell_lookup)
        qc = validate_dynamic_soc(dynamic, comparison, cell_lookup, manifest, observations)
        assert not any(join_qc.values())
        assert not any(qc.values())
        counts[cell_id] = comparison["mission_cycle_id"].nunique()
    assert counts == {"VAH01": 828, "VAH17": 980, "VAH27": 572}
