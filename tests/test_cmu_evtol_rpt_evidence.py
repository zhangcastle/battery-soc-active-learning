from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from battery_benchmark.data.cmu_evtol_pipeline_v0 import contiguous_ns_runs
from scripts.audit_cmu_evtol_rpt_evidence import (
    audit_candidate,
    audit_cell,
    discover_broad_rpt_windows,
)
from battery_benchmark.data.cmu_evtol_pipeline_v0 import load_diagnostic_protocol


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data/raw/cmu_evtol_vtc6/figshare_14226830"
PROTOCOL = load_diagnostic_protocol(ROOT / "configs/protocols/cmu_evtol_mission_segmentation_v0_1.yaml")[0]


def _rpt(*, end_v: float = 2.5, start_v: float = 4.2, charge: bool = False, reset_cycle: bool = False) -> pd.DataFrame:
    # Three C/5 fragments separated by the two short resistance-measurement runs.
    times = [0, 1800, 3600, 3630, 7200, 10800, 10830, 14400, 18000]
    ns = [3, 3, 3, 4, 5, 5, 6, 7, 7]
    current = [-600, -600, -600, -50, -600, -600, -50, -600, -600]
    if charge: current[4] = 600
    voltage = [start_v + (end_v - start_v) * i / 8 for i in range(9)]
    cycles = [10, 10, 10, 10, 10, 1 if reset_cycle else 10, 1 if reset_cycle else 10, 1 if reset_cycle else 10, 1 if reset_cycle else 10]
    return pd.DataFrame({
        "cell_id": "VAH01", "source_file": "VAH01.csv", "source_row": range(1, 10),
        "time_s": times, "Ecell_V": voltage, "I_mA": current,
        "EnergyCharge_W_h": 0.0, "QCharge_mA_h": 0.0,
        "EnergyDischarge_W_h": 0.0, "QDischarge_mA_h": [0, 300, 600, 600, 1200, 1800, 1800, 2400, 3000],
        "Temperature__C": 25.0, "cycleNumber": cycles, "Ns": ns,
        "continuous_block_id": 0, "continuous_block_boundary_reason": "",
    })


def test_complete_c5_with_interruptions_is_strong_and_id_ignores_cycle_number() -> None:
    data = _rpt(reset_cycle=True)
    runs = contiguous_ns_runs(data)
    windows = discover_broad_rpt_windows(runs)
    assert len(windows) == 1
    result, _ = audit_candidate(data, runs, *windows[0])
    assert result["audit_status"] == "strong_rpt_evidence"
    assert result["resistance_interruption_count"] == 2
    assert result["preceding_full_charge_evidence"] is False
    assert result["candidate_ends_at_file_tail"] is True
    assert "exact_ns_structure_3_4_5_6_7" in json.loads(result["supporting_evidence"])
    assert "file_tail_without_following_charge" in json.loads(result["conflicting_evidence"])
    candidate_id = result["diagnostic_rpt_candidate_id"]
    changed = data.copy(); changed["cycleNumber"] = 999
    changed_result, _ = audit_candidate(changed, contiguous_ns_runs(changed), *windows[0])
    assert changed_result["diagnostic_rpt_candidate_id"] == candidate_id


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (_rpt(end_v=3.2), "uncertain_rpt_evidence"),
        (_rpt(start_v=3.8), "uncertain_rpt_evidence"),
        (_rpt(charge=True), "non_rpt_candidate"),
    ],
)
def test_incomplete_or_contaminated_rpt_is_not_strong(data: pd.DataFrame, expected: str) -> None:
    runs = contiguous_ns_runs(data)
    result, _ = audit_candidate(data, runs, *discover_broad_rpt_windows(runs)[0])
    assert result["audit_status"] == expected


def test_short_mission_and_empty_input_do_not_create_broad_candidate() -> None:
    data = _rpt().iloc[:3].copy()
    data["time_s"] = [0, 400, 800]
    data["I_mA"] = -12000
    assert discover_broad_rpt_windows(contiguous_ns_runs(data)) == []
    empty = contiguous_ns_runs(_rpt()).iloc[0:0]
    assert discover_broad_rpt_windows(empty) == []
    empty_audit, rows = audit_cell(_rpt().iloc[0:0], PROTOCOL)
    assert empty_audit.empty
    assert "diagnostic_rpt_candidate_id" in empty_audit.columns
    assert rows == {}


def test_cross_block_is_not_merged_and_overlapping_fragments_are_deduplicated() -> None:
    data = _rpt()
    data.loc[data["source_row"].ge(5), "continuous_block_id"] = 1
    windows = discover_broad_rpt_windows(contiguous_ns_runs(data))
    assert len(windows) == 2
    assert all(first <= last for first, last in windows)


def test_audit_output_has_no_formal_rpt_id() -> None:
    audit, _ = audit_cell(_rpt(), PROTOCOL)
    assert "diagnostic_rpt_candidate_id" in audit.columns
    assert "rpt_id" not in audit.columns


@pytest.mark.integration
def test_real_baseline_rpt_audit_when_raw_is_available() -> None:
    if not RAW_ROOT.is_dir():
        pytest.skip("CMU Version 3 raw root unavailable")
    for cell_id in ("VAH01", "VAH17", "VAH27"):
        from battery_benchmark.data.cmu_evtol_raw import load_cmu_v3_main_file
        audit, _ = audit_cell(load_cmu_v3_main_file(RAW_ROOT / f"{cell_id}.csv").data, PROTOCOL)
        assert not audit.empty
        assert "rpt_id" not in audit.columns
