from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data/raw/cmu_evtol_vtc6/figshare_14226830"
MANIFEST_PATH = ROOT / "data/manifests/cmu_evtol/cmu_evtol_vtc6_cell_protocols.csv"
PROTOCOL_PATH = ROOT / "configs/datasets/cmu_evtol/protocols/cmu_evtol_transfer_benchmark_v0_1.yaml"

EXPECTED_CELL_IDS = {
    "VAH01",
    "VAH02",
    "VAH05",
    "VAH06",
    "VAH07",
    "VAH09",
    "VAH10",
    "VAH11",
    "VAH12",
    "VAH13",
    "VAH15",
    "VAH16",
    "VAH17",
    "VAH20",
    "VAH22",
    "VAH23",
    "VAH24",
    "VAH25",
    "VAH26",
    "VAH27",
    "VAH28",
    "VAH30",
}

EXPECTED_BASELINE_IDS = {"VAH01", "VAH17", "VAH27"}

EXPECTED_FOLDS = {
    "test_vah01": ({"VAH17", "VAH27"}, {"VAH01"}),
    "test_vah17": ({"VAH01", "VAH27"}, {"VAH17"}),
    "test_vah27": ({"VAH01", "VAH17"}, {"VAH27"}),
}

REQUIRED_TRACKS = {
    "zero_shot",
    "unsupervised_adaptation",
    "supervised_few_shot",
    "same_budget_scratch",
    "full_target_upper_bound",
}

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


def test_manifest_contains_expected_22_cells_without_raw_dataset() -> None:
    rows = _manifest_rows()
    cell_ids = [row["cell_id"] for row in rows]

    assert len(rows) == 22
    assert set(cell_ids) == EXPECTED_CELL_IDS
    assert len(set(cell_ids)) == len(rows)
    assert all(row["main_file"] == f'{row["cell_id"]}.csv' for row in rows)


def test_manifest_impedance_flags_without_raw_dataset() -> None:
    for row in _manifest_rows():
        expected = row["cell_id"] != "VAH26"
        assert _as_bool(row["has_impedance"]) is expected


@pytest.mark.integration
def test_optional_raw_package_matches_manifest() -> None:
    if not RAW_ROOT.is_dir():
        pytest.skip(f"Optional CMU raw package is not present: {RAW_ROOT}")

    rows = _manifest_rows()
    raw_main_files = {
        path.name
        for path in RAW_ROOT.glob("VAH[0-9][0-9].csv")
        if not path.name.endswith("_impedance.csv")
    }
    assert raw_main_files == {f"{cell_id}.csv" for cell_id in EXPECTED_CELL_IDS}

    for row in rows:
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

    assert baseline_ids == EXPECTED_BASELINE_IDS


def test_1_5c_charge_current_wording_is_neutral() -> None:
    rows_by_cell = {row["cell_id"]: row for row in _manifest_rows()}
    for cell_id in ("VAH16", "VAH20"):
        row = rows_by_cell[cell_id]
        assert row["protocol_group"] == "charge_current_shift"
        assert row["changed_factor"] == "charge_current"
        assert row["changed_value"] == "charge_current_1_5_c"
        assert "reduced" not in row["notes"].lower()


def test_protocol_domains_and_required_tracks() -> None:
    protocol = _protocol()
    source = protocol["source_domain"]
    target = protocol["target_domain"]

    assert source == {
        "dataset": "panasonic_18650pf",
        "cell_model": "NCR18650PF",
        "main_temperature_c": 25,
    }
    assert target["dataset"] == "cmu_evtol_vtc6"
    assert target["cell_model"] == "Sony-Murata 18650 VTC6"
    assert set(protocol["tracks"]) == REQUIRED_TRACKS


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
    assert budgets["unit"] == "percent_of_eligible_complete_missions_per_adaptation_cell"
    assert budgets["selection"] == "chronological_prefix_per_adaptation_cell"
    assert budgets["rounding"] == "ceil"
    assert budgets["rounding_scope"] == "per_adaptation_cell"
    assert budgets["merge_scope"] == "fold_after_per_cell_selection"
    assert budgets["nested_subsets_required"] is True
    assert budgets["same_budget_manifest_for_all_models"] is True
    assert budgets["same_manifest_for_transfer_and_scratch"] is True


def test_leave_one_cell_out_folds_keep_test_cells_out_of_adaptation() -> None:
    protocol = _protocol()
    observed_test_cells = set()

    for fold in protocol["folds"]:
        adaptation = set(fold["adaptation_cells"])
        test = set(fold["test_cells"])
        expected_adaptation, expected_test = EXPECTED_FOLDS[fold["fold_id"]]
        assert adaptation == expected_adaptation
        assert test == expected_test
        assert adaptation.isdisjoint(test)
        assert adaptation | test == EXPECTED_BASELINE_IDS
        assert len(adaptation) == 2
        assert len(test) == 1
        observed_test_cells.update(test)

    assert len(protocol["folds"]) == 3
    assert observed_test_cells == EXPECTED_BASELINE_IDS
    assert all(value is False for value in protocol["test_isolation"].values())


def test_protocol_does_not_claim_mission_ids_or_enable_condition_cells() -> None:
    protocol = _protocol()
    assert protocol["condition_stress_tests"]["enabled_in_primary_benchmark"] is False
    assert protocol["deferred"] == {
        "mission_ids": "not_defined_in_v0_1",
        "mission_reconstruction": "not_defined_in_v0_1",
        "soc_reference_version": "not_defined_in_v0_1",
        "rpt_and_capacity_policy": "not_defined_in_v0_1",
        "resampling_interval": "not_defined_in_v0_1",
        "physical_context_length": "not_defined_in_v0_1",
        "aging_stage_boundaries": "not_defined_in_v0_1",
    }
    assert "mission_ids" not in protocol["folds"][0]
