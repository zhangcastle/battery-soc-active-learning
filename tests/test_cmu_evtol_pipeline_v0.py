from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from battery_benchmark.data.cmu_evtol_pipeline_v0 import (
    MANIFEST_COLUMNS,
    PHASE_ROLES,
    ROW_COLUMNS,
    build_candidate_rows,
    contiguous_ns_runs,
    enumerate_diagnostic_candidates,
    load_diagnostic_protocol,
    write_pipeline_outputs,
)
from battery_benchmark.data.cmu_evtol_raw import RAW_COLUMNS, sha256_file


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/protocols/cmu_evtol_mission_segmentation_v0_1.yaml"
RAW_ROOT = ROOT / "data/raw/cmu_evtol_vtc6/figshare_14226830"


def _canonical(powers: list[float] | None = None, ns: list[int] | None = None) -> pd.DataFrame:
    powers = powers or [-50.0] * 3 + [-16.0] * 3 + [-50.0] * 3
    ns = ns or [4] * 3 + [5] * 3 + [6] * 3
    times = [0.0, 30.0, 60.0, 70.0, 170.0, 270.0, 280.0, 330.0, 380.0]
    return pd.DataFrame({
        "cell_id": "VAH01",
        "source_file": "VAH01.csv",
        "source_row": range(1, 10),
        "time_s": times,
        "Ecell_V": 4.0,
        "I_mA": [power / 4.0 * 1000.0 for power in powers],
        "Temperature__C": 25.0,
        "cycleNumber": 1,
        "Ns": ns,
        "continuous_block_id": 0,
        "continuous_block_boundary_reason": "",
    })


def _protocol() -> dict:
    return load_diagnostic_protocol(PROTOCOL_PATH)[0]


def test_candidate_screening_statuses_boundaries_overlap_and_stable_id() -> None:
    data = _canonical()
    runs = contiguous_ns_runs(data)
    matched = enumerate_diagnostic_candidates(runs, _protocol())
    assert matched["screening_status"].tolist() == ["diagnostic_screening_match", "uncertain"]
    candidate_id = matched.iloc[0]["development_candidate_id"]
    assert "mission_id" not in matched.columns
    changed_cycle = data.copy(); changed_cycle["cycleNumber"] = 999
    assert enumerate_diagnostic_candidates(contiguous_ns_runs(changed_cycle), _protocol()).iloc[0]["development_candidate_id"] == candidate_id

    crossed = runs.copy(); crossed.loc[2, "continuous_block_id"] = 1
    assert enumerate_diagnostic_candidates(crossed, _protocol()).iloc[0]["screening_status"] == "hard_reject"
    wrong = _canonical(powers=[-50.0] * 6 + [-16.0] * 3)
    assert enumerate_diagnostic_candidates(contiguous_ns_runs(wrong), _protocol()).iloc[0]["screening_status"] == "hard_reject"
    charge = _canonical(powers=[-50.0] * 3 + [16.0] * 3 + [-50.0] * 3)
    assert enumerate_diagnostic_candidates(contiguous_ns_runs(charge), _protocol()).iloc[0]["screening_status"] == "hard_reject"

    missing = enumerate_diagnostic_candidates(runs.iloc[:2], _protocol())
    assert missing.iloc[0]["screening_status"] == "uncertain"
    alone = enumerate_diagnostic_candidates(runs.iloc[:1], _protocol())
    assert alone.iloc[0]["screening_status"] != "diagnostic_screening_match"

    overlap = pd.concat([runs, runs.iloc[1:].copy()], ignore_index=True)
    overlap.loc[3:, "source_row_start"] += 9; overlap.loc[3:, "source_row_end"] += 9
    overlap.loc[3:, "time_start_s"] += 500; overlap.loc[3:, "time_end_s"] += 500
    overlap.loc[3:, "ns_value"] = [5, 6]
    # Explicit five-run high/lower/high/lower/high sequence.
    overlap = pd.concat([runs, runs.iloc[1:], runs.iloc[2:]], ignore_index=True).iloc[:5].copy()
    overlap.index = range(5)
    overlap.loc[3] = runs.iloc[1]; overlap.loc[4] = runs.iloc[2]
    overlapping = enumerate_diagnostic_candidates(overlap, _protocol())
    assert overlapping["screening_status"].tolist() == [
        "diagnostic_screening_match", "uncertain", "uncertain"
    ]
    assert "overlaps_earlier_candidate" in overlapping.iloc[1]["all_reason_codes"]


def test_rpt_like_run_is_hard_rejected() -> None:
    runs = contiguous_ns_runs(_canonical())
    runs.loc[1, ["discharge_power_median_W", "duration_s", "start_voltage_V", "end_voltage_V"]] = [2.0, 4000.0, 4.1, 2.5]
    result = enumerate_diagnostic_candidates(runs, _protocol()).iloc[0]
    assert result["screening_status"] == "hard_reject"
    assert "long_low_rate" in result["all_reason_codes"]


def test_all_triggered_reasons_are_unique_and_primary_uses_protocol_order() -> None:
    runs = contiguous_ns_runs(_canonical())
    runs.loc[1, ["discharge_fraction", "discharge_power_median_W"]] = [0.5, 50.0]
    runs.loc[2, "discharge_power_median_W"] = 16.0
    result = enumerate_diagnostic_candidates(runs, _protocol()).iloc[0]
    reasons = json.loads(result["all_reason_codes"])
    assert result["screening_status"] == "hard_reject"
    assert result["primary_reason"] == "severe_charge_or_rest_contamination"
    assert reasons == [
        "severe_charge_or_rest_contamination",
        "segment_order_definitively_incorrect",
        "power_or_duration_near_but_not_within_screening_range",
    ]
    assert len(reasons) == len(set(reasons))


def test_empty_anchor_and_no_match_outputs_have_stable_schema() -> None:
    canonical = _canonical(powers=[-16.0] * 9)
    empty_manifest = enumerate_diagnostic_candidates(contiguous_ns_runs(canonical), _protocol())
    rows, empty_manifest = build_candidate_rows(canonical, empty_manifest, capacity_ah=3.0)
    assert empty_manifest.empty
    assert tuple(empty_manifest.columns[: len(MANIFEST_COLUMNS)]) == MANIFEST_COLUMNS
    assert rows.empty and tuple(rows.columns) == ROW_COLUMNS

    contaminated = _canonical(powers=[-50.0] * 3 + [16.0] * 3 + [-50.0] * 3)
    no_match = enumerate_diagnostic_candidates(contiguous_ns_runs(contaminated), _protocol())
    rows, _ = build_candidate_rows(contaminated, no_match, capacity_ah=3.0)
    assert not no_match.empty
    assert not no_match["screening_status"].eq("diagnostic_screening_match").any()
    assert rows.empty and tuple(rows.columns) == ROW_COLUMNS


def test_synthetic_raw_to_candidate_soc_and_manifest(tmp_path: Path) -> None:
    raw_root, output_root = tmp_path / "raw", tmp_path / "out"
    raw_root.mkdir()
    canonical = _canonical()
    raw = pd.DataFrame({column: 0.0 for column in RAW_COLUMNS}, index=range(len(canonical)))
    for column in ("time_s", "Ecell_V", "I_mA", "Temperature__C", "cycleNumber", "Ns"):
        raw[column] = canonical[column].to_numpy()
    main = raw_root / "VAH01.csv"; raw.to_csv(main, index=False)
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame([{"cell_id": "VAH01", "sha256": sha256_file(main)}]).to_csv(inventory, index=False)
    metadata = write_pipeline_outputs(
        raw_root, output_root, PROTOCOL_PATH,
        asset_inventory_path=inventory, cells=["VAH01"], capacity_ah=3.0, repo_root=ROOT,
    )
    rows = pd.read_parquet(output_root / "VAH01_candidate_rows.parquet")
    manifest = pd.read_csv(output_root / "candidate_manifest.csv")
    assert set(rows["development_phase_role"]) == set(PHASE_ROLES)
    assert rows["soc_raw"].iloc[0] == 1.0
    assert manifest["screening_status"].tolist() == ["diagnostic_screening_match", "uncertain"]
    assert metadata["pipeline_status"] == "development_only"
    assert metadata["formal_benchmark_eligible"] is False
    assert metadata["label_is_ground_truth"] is False
    forbidden = {"mission_id", "phase_name", "split", "soh", "rpt_capacity"}
    assert forbidden.isdisjoint({column.lower() for column in rows.columns})
    qc = json.loads((output_root / "VAH01_qc_summary.json").read_text())
    assert qc["processed_row_count"] == 9
    assert qc["screening_unit"] == "anchor_window"
    assert "anchor_window_status_counts" in qc
    assert "candidate_status_counts" not in qc
    assert qc["capacity_mode"] == "constant" and qc["capacity_ah"] == 3.0
    assert qc["soc_initial_source"] == "protocol_full_charge_assumption"


def test_no_anchor_writes_empty_outputs_and_qc(tmp_path: Path) -> None:
    raw_root, output_root = tmp_path / "raw", tmp_path / "out"
    raw_root.mkdir()
    canonical = _canonical(powers=[-16.0] * 9)
    raw = pd.DataFrame({column: 0.0 for column in RAW_COLUMNS}, index=range(len(canonical)))
    for column in ("time_s", "Ecell_V", "I_mA", "Temperature__C", "cycleNumber", "Ns"):
        raw[column] = canonical[column].to_numpy()
    main = raw_root / "VAH01.csv"
    raw.to_csv(main, index=False)
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame([{"cell_id": "VAH01", "sha256": sha256_file(main)}]).to_csv(inventory, index=False)
    write_pipeline_outputs(
        raw_root, output_root, PROTOCOL_PATH,
        asset_inventory_path=inventory, cells=["VAH01"], repo_root=ROOT,
    )
    assert pd.read_parquet(output_root / "VAH01_candidate_rows.parquet").empty
    assert tuple(pd.read_csv(output_root / "candidate_manifest.csv").columns[: len(MANIFEST_COLUMNS)]) == MANIFEST_COLUMNS
    qc = json.loads((output_root / "VAH01_qc_summary.json").read_text())
    assert qc["anchor_window_status_counts"] == {}
    assert qc["processed_row_count"] == 0


def test_non_baseline_cell_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-baseline"):
        write_pipeline_outputs(
            tmp_path, tmp_path / "out", PROTOCOL_PATH,
            asset_inventory_path=tmp_path / "missing.csv", cells=["VAH02"], repo_root=ROOT,
        )


def test_duplicate_cells_and_non_3ah_api_and_cli_are_rejected(tmp_path: Path) -> None:
    common = dict(
        raw_root=tmp_path,
        output_root=tmp_path / "out",
        protocol_path=PROTOCOL_PATH,
        asset_inventory_path=tmp_path / "missing.csv",
        repo_root=ROOT,
    )
    with pytest.raises(ValueError, match="duplicate cell IDs"):
        write_pipeline_outputs(**common, cells=["VAH01", "VAH01"])
    with pytest.raises(ValueError, match="requires capacity_ah=3.0"):
        write_pipeline_outputs(**common, cells=["VAH01"], capacity_ah=2.9)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_cmu_evtol_pipeline_v0.py",
            "--raw-root", str(tmp_path),
            "--asset-inventory", str(tmp_path / "missing.csv"),
            "--capacity-ah", "2.9",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires capacity_ah=3.0" in result.stderr


@pytest.mark.integration
def test_real_baseline_candidate_counts_match_evidence_audit(tmp_path: Path) -> None:
    if not RAW_ROOT.is_dir():
        pytest.skip("CMU Version 3 raw root is unavailable")
    expected = {"VAH01": 828, "VAH17": 980, "VAH27": 572}
    protocol, protocol_sha = load_diagnostic_protocol(PROTOCOL_PATH)
    from battery_benchmark.data.cmu_evtol_pipeline_v0 import process_cmu_cell
    for cell_id, count in expected.items():
        rows, manifest, _ = process_cmu_cell(
            RAW_ROOT / f"{cell_id}.csv", protocol, protocol_sha, repo_root=ROOT
        )
        matched = manifest.loc[manifest["screening_status"].eq("diagnostic_screening_match")]
        assert len(matched) == count, f"{cell_id}: expected {count}, got {len(matched)}"
        assert not rows.empty
        assert rows.groupby("development_candidate_id")["continuous_block_id"].nunique().eq(1).all()
        inventory = pd.read_csv(ROOT / "data/manifests/cmu_evtol_vtc6_asset_inventory_v3.csv").set_index("cell_id")
        assert matched["input_sha256"].iloc[0] == inventory.at[cell_id, "sha256"]
