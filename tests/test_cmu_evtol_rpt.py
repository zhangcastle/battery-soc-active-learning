from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from battery_benchmark.data.cmu_evtol_pipeline_v0 import load_diagnostic_protocol
from battery_benchmark.data.cmu_evtol_raw import load_cmu_v3_main_file
from battery_benchmark.data.cmu_evtol_rpt import (
    OBSERVATION_COLUMNS,
    detect_rpt_capacity_observations,
    discover_broad_rpt_windows,
    integrate_rpt_capacity,
    load_rpt_capacity_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data/raw/cmu_evtol_vtc6/figshare_14226830"
PROTOCOL = load_rpt_capacity_protocol(
    ROOT / "configs/protocols/cmu_evtol_rpt_capacity_v1.yaml"
)[0]
MISSION_PROTOCOL = load_diagnostic_protocol(
    ROOT / "configs/protocols/cmu_evtol_mission_segmentation_v0_1.yaml"
)[0]


def _rpt(
    *,
    start_v: float = 4.2,
    end_v: float = 2.5,
    include_charge: bool = True,
    include_rest: bool = True,
    include_following: bool = True,
) -> pd.DataFrame:
    times = [-300, -200, -100, -10, 0, 1800, 3600, 3630, 7200, 10800, 10830, 14400, 18000]
    ns = [0, 0, 2, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7]
    current = [1000, 100, 0, 0, -720, -720, -720, -70, -720, -720, -70, -720, -720]
    voltage = [4.0, 4.2, 4.2, 4.2] + [start_v + (end_v - start_v) * i / 8 for i in range(9)]
    if not include_charge:
        current[:2] = [0, 0]
    if not include_rest:
        current[2:4] = [-10, -10]
    if include_following:
        times += [18030, 18100, 18130, 18230]
        ns += [8, 8, 0, 0]
        current += [0, 0, 100, 1000]
        voltage += [2.5, 2.5, 2.6, 4.2]
    frame = pd.DataFrame({
        "cell_id": "VAH01", "source_file": "VAH01.csv",
        "source_row": range(1, len(times) + 1), "time_s": times,
        "Ecell_V": voltage, "I_mA": current,
        "EnergyCharge_W_h": 0.0, "QCharge_mA_h": 0.0,
        "EnergyDischarge_W_h": 0.0,
        "QDischarge_mA_h": 0.0,
        "Temperature__C": 25.0, "cycleNumber": 10, "Ns": ns,
        "continuous_block_id": 0, "continuous_block_boundary_reason": "",
    })
    discharge_a = np.maximum(-frame["I_mA"].to_numpy() / 1000, 0)
    increments = np.zeros(len(frame))
    increments[1:] = 0.5 * (discharge_a[:-1] + discharge_a[1:]) * np.diff(frame["time_s"]) / 3600
    frame["QDischarge_mA_h"] = np.cumsum(increments) * 1000
    return frame


def _detect(data: pd.DataFrame) -> pd.DataFrame:
    return detect_rpt_capacity_observations(data, PROTOCOL)


def test_complete_rpt_is_accepted_and_integral_includes_interruptions() -> None:
    data = _rpt()
    result = _detect(data)
    assert len(result) == 1
    row = result.iloc[0]
    candidate = data.loc[data["Ns"].isin([3, 4, 5, 6, 7])]
    expected_net = -np.trapz(candidate["I_mA"] / 1000, candidate["time_s"]) / 3600
    expected_discharge = np.trapz(
        np.maximum(-candidate["I_mA"] / 1000, 0), candidate["time_s"]
    ) / 3600
    assert row.detector_status == "accepted"
    assert row.interruption_count == 2
    assert row.capacity_net_ah == pytest.approx(expected_net)
    assert row.capacity_discharge_only_ah == pytest.approx(expected_discharge)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda d: d.assign(Ecell_V=d["Ecell_V"].where(d["Ns"].ne(7), 3.2)), "cutoff_not_reached"),
        (lambda d: d.assign(Ecell_V=d["Ecell_V"].where(d["Ns"].ne(3), 3.9)), "start_voltage_below_full_region"),
        (lambda d: _rpt(include_charge=False), "missing_preceding_full_charge"),
        (lambda d: _rpt(include_rest=False), "missing_preceding_rest"),
        (lambda d: d.assign(Ns=d["Ns"].replace({4: 9})), "unexpected_ns_sequence"),
        (lambda d: d.assign(Ns=d["Ns"].replace({4: 3})), "interruption_count_mismatch"),
    ],
)
def test_incomplete_rpt_requires_review(mutate, reason: str) -> None:
    result = _detect(mutate(_rpt()))
    assert not result.empty
    assert not result["detector_status"].eq("accepted").all()
    assert any(reason in json.loads(value) for value in result["all_reason_codes"])


def test_short_mission_and_cross_block_are_not_accepted() -> None:
    mission = _rpt().loc[lambda x: x["Ns"].eq(3)].copy()
    mission["time_s"] = np.arange(len(mission)) * 100
    mission["I_mA"] = -12000
    assert _detect(mission).empty
    crossed = _rpt()
    crossed.loc[crossed["Ns"].isin([5, 6, 7]), "continuous_block_id"] = 1
    assert not _detect(crossed)["detector_status"].eq("accepted").any()


def test_charge_contamination_rejects_and_broad_fragments_are_deduplicated() -> None:
    contaminated = _rpt()
    contaminated.loc[contaminated["Ns"].isin([4, 6]), "I_mA"] = 200
    result = _detect(contaminated)
    assert len(result) == 1
    assert result.iloc[0].detector_status == "rejected"
    assert "severe_charge_contamination" in json.loads(result.iloc[0].all_reason_codes)
    from battery_benchmark.data.cmu_evtol_pipeline_v0 import contiguous_ns_runs
    runs = contiguous_ns_runs(_rpt())
    assert len(discover_broad_rpt_windows(runs, PROTOCOL["broad_discovery"])) == 1


def test_time_and_current_validation_fail_closed() -> None:
    candidate = _rpt().loc[lambda x: x["Ns"].isin([3, 4, 5, 6, 7])].copy()
    duplicate = candidate.copy(); duplicate.iloc[2, duplicate.columns.get_loc("time_s")] = duplicate.iloc[1].time_s
    with pytest.raises(ValueError, match="strictly increasing"):
        integrate_rpt_capacity(duplicate)
    reversed_time = candidate.copy(); reversed_time.iloc[2, reversed_time.columns.get_loc("time_s")] = -1
    with pytest.raises(ValueError, match="strictly increasing"):
        integrate_rpt_capacity(reversed_time)
    nonfinite = candidate.copy(); nonfinite.iloc[2, nonfinite.columns.get_loc("I_mA")] = np.nan
    with pytest.raises(ValueError, match="non-finite current"):
        integrate_rpt_capacity(nonfinite)


def test_file_tail_complete_is_accepted_but_truncated_is_review() -> None:
    complete = _detect(_rpt(include_following=False)).iloc[0]
    assert complete.detector_status == "accepted"
    assert complete.candidate_ends_at_file_tail
    assert "file_tail_without_following_charge" in json.loads(complete.warning_codes)
    truncated = _detect(_rpt(end_v=3.2, include_following=False)).iloc[0]
    assert truncated.detector_status == "review_required"


def test_cycle_number_does_not_change_id_and_q_reset_is_only_warning() -> None:
    original = _detect(_rpt()).iloc[0]
    changed = _rpt(); changed["cycleNumber"] = np.arange(len(changed))[::-1]
    changed.loc[changed["Ns"].eq(5), "QDischarge_mA_h"] = 0
    result = _detect(changed).iloc[0]
    assert result.rpt_id == original.rpt_id
    assert result.detector_status == "accepted"
    assert "qdischarge_counter_reset" in json.loads(result.warning_codes)
    assert "cycleNumber" not in result.rpt_id


def test_empty_input_has_stable_schema_and_ids_change_with_boundaries() -> None:
    empty = _detect(_rpt().iloc[0:0])
    assert empty.empty and tuple(empty.columns) == OBSERVATION_COLUMNS
    original = _detect(_rpt()).iloc[0].rpt_id
    shifted = _rpt(); shifted["source_row"] += 100
    assert _detect(shifted).iloc[0].rpt_id != original


def _campaign(scales: list[float]) -> pd.DataFrame:
    parts = []
    for index, scale in enumerate(scales):
        part = _rpt()
        part["time_s"] += index * 20000
        part["source_row"] += index * 100
        part["I_mA"] = part["I_mA"].astype(float)
        part.loc[part["Ns"].isin([3, 5, 7]), "I_mA"] *= scale
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def test_small_rebound_warns_but_large_adjacent_jump_requires_review() -> None:
    rebound = _detect(_campaign([1.0, 1.005]))
    assert rebound.iloc[1].detector_status == "accepted"
    assert "small_capacity_rebound" in json.loads(rebound.iloc[1].warning_codes)
    jump = _detect(_campaign([1.0, 0.85]))
    assert jump.iloc[1].detector_status == "review_required"
    assert jump.iloc[1].primary_reason == "adjacent_capacity_jump"


@pytest.mark.integration
def test_real_baseline_rpt_capacity_regression() -> None:
    if not RAW_ROOT.is_dir():
        pytest.skip("CMU Version 3 raw root unavailable")
    expected = {"VAH01": 18, "VAH17": 21, "VAH27": 13}
    capacities = []
    for cell_id, count in expected.items():
        data = load_cmu_v3_main_file(RAW_ROOT / f"{cell_id}.csv").data
        first = detect_rpt_capacity_observations(data, PROTOCOL, mission_protocol=MISSION_PROTOCOL)
        changed_cycle = data.copy(); changed_cycle["cycleNumber"] = -999
        second = detect_rpt_capacity_observations(
            changed_cycle, PROTOCOL, mission_protocol=MISSION_PROTOCOL
        )
        assert len(first) == count
        assert first["rpt_id"].tolist() == second["rpt_id"].tolist()
        assert np.allclose(first["capacity_net_ah"], second["capacity_net_ah"])
        assert first["ns_sequence"].eq("[3,4,5,6,7]").all()
        assert first["interruption_count"].eq(2).all()
        assert first["continuous_block_id"].notna().all()
        assert first.iloc[0].diagnostic_missions_before_rpt == 0
        internal = first.loc[~first["candidate_ends_at_file_tail"]].iloc[1:]
        assert internal["missions_since_previous_accepted_rpt"].eq(50).all()
        capacities.extend(first["capacity_net_ah"])
    assert min(capacities) == pytest.approx(2.436, abs=0.01)
    assert max(capacities) == pytest.approx(3.121, abs=0.01)
