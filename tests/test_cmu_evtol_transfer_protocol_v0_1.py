from __future__ import annotations

import csv
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data/raw/cmu_evtol_vtc6/figshare_14226830"
MANIFEST_PATH = ROOT / "data/manifests/cmu_evtol_vtc6_cell_protocols.csv"
PROTOCOL_PATH = ROOT / "configs/protocols/cmu_evtol_transfer_benchmark_v0_1.yaml"

PROTOCOL_GROUPS = {
    "baseline",
    "discharge_power_shift",
    "cruise_duration_shift",
    "charge_current_shift",
    "thermal_or_cooling_shift",
    "end_of_charge_voltage_shift",
    "other_or_uncertain",
}

CHANGED_FACTORS = {
    "none",
    "discharge_power",
    "cruise_duration",
    "charge_current",
    "thermal_or_cooling",
    "end_of_charge_voltage",
    "uncertain",
}


def _manifest_rows() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _protocol() -> dict:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _as_bool(value: str) -> bool:
    assert value in {"true", "false"}
    return value == "true"


def test_manifest_contains_every_main_timeseries_cell_once() -> None:
    rows = _manifest_rows()
    raw_main_files = sorted(
        path.name
        for path in RAW_ROOT.glob("VAH[0-9][0-9].csv")
        if not path.name.endswith("_impedance.csv")
    )

    assert len(raw_main_files) == 22
    assert len(rows) == 22
    assert sorted(row["main_file"] for row in rows) == raw_main_files
    assert len({row["cell_id"] for row in rows}) == len(rows)
    assert all(row["main_file"] == f'{row["cell_id"]}.csv' for row in rows)


def test_impedance_flags_match_the_source_package() -> None:
    for row in _manifest_rows():
        expected = (RAW_ROOT / f'{row["cell_id"]}_impedance.csv').is_file()
        assert _as_bool(row["has_impedance"]) is expected


def test_baseline_condition_and_uncertain_fields_are_consistent() -> None:
    rows = _manifest_rows()
    baseline_ids = set()

    for row in rows:
        is_baseline = _as_bool(row["is_baseline"])
        group = row["protocol_group"]
        factor = row["changed_factor"]

        assert group in PROTOCOL_GROUPS
        assert factor in CHANGED_FACTORS
        assert is_baseline is (group == "baseline")

        if is_baseline:
            baseline_ids.add(row["cell_id"])
            assert factor == "none"
            assert row["changed_value"] == "baseline_protocol"
        else:
            assert factor != "none"

        if group == "other_or_uncertain" or factor == "uncertain":
            assert not is_baseline

    assert baseline_ids == {"VAH01", "VAH17", "VAH27"}


def test_protocol_prohibits_row_level_splitting_and_uses_fixed_budgets() -> None:
    protocol = _protocol()
    primary = protocol["primary_benchmark"]
    budgets = protocol["target_label_budgets"]

    assert primary["evaluation_strategy"] == "leave_one_cell_out"
    assert primary["split_unit"] == "complete_mission"
    assert primary["row_level_random_split"] is False
    assert primary["window_level_random_split_before_mission_assignment"] is False
    assert primary["same_mission_must_remain_in_one_split"] is True
    assert budgets["values_percent"] == [1, 5, 10]
    assert budgets["selection"] == "chronological_prefix_per_adaptation_cell"
    assert budgets["nested_subsets_required"] is True
    assert budgets["same_manifest_for_transfer_and_scratch"] is True


def test_leave_one_cell_out_folds_keep_test_cells_out_of_adaptation() -> None:
    protocol = _protocol()
    baseline_ids = {"VAH01", "VAH17", "VAH27"}
    observed_test_cells = set()

    for fold in protocol["folds"]:
        adaptation = set(fold["adaptation_cells"])
        test = set(fold["test_cells"])
        assert adaptation.isdisjoint(test)
        assert adaptation | test == baseline_ids
        assert len(adaptation) == 2
        assert len(test) == 1
        observed_test_cells.update(test)

    assert observed_test_cells == baseline_ids
    assert all(value is False for value in protocol["test_isolation"].values())


def test_protocol_does_not_claim_mission_ids_or_enable_condition_cells() -> None:
    protocol = _protocol()
    assert protocol["condition_stress_tests"]["enabled_in_primary_benchmark"] is False
    assert protocol["deferred"]["mission_ids"] == "not_defined_in_v0_1"
    assert "mission_ids" not in protocol["folds"][0]
