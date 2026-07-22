from pathlib import Path

import pandas as pd
import pytest

from battery_benchmark.datasets.cmu_evtol.development_pipeline_v0 import (
    SUPPORTED_CELLS,
    contiguous_ns_runs,
    enumerate_diagnostic_candidates,
    load_diagnostic_protocol,
)
from battery_benchmark.datasets.cmu_evtol.raw import load_cmu_v3_main_file
from battery_benchmark.datasets.cmu_evtol.rpt import (
    SUPPORTED_CELLS as RPT_SUPPORTED_CELLS,
    detect_rpt_capacity_observations,
    load_rpt_capacity_protocol,
)


ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data/raw/cmu_evtol_vtc6/figshare_14226830"
MISSION_PROTOCOL = ROOT / "configs/datasets/cmu_evtol/protocols/cmu_evtol_mission_segmentation_v0_1.yaml"
RPT_PROTOCOL = ROOT / "configs/datasets/cmu_evtol/protocols/cmu_evtol_rpt_capacity_v1_extension4.yaml"
RPT_OUTPUT = ROOT / "outputs/cmu_evtol_rpt_capacity_v1_extension4/capacity_observations.csv"
LOOKUP_OUTPUT = ROOT / "outputs/datasets/cmu_evtol/causal_capacity_lookup_v1_extension4/capacity_lookup.csv"
DYNAMIC_ROOT = ROOT / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1_extension4"
ADAPTER_INVENTORY = ROOT / "outputs/datasets/cmu_evtol/vidal_adapter_v1/core7_inventory.csv"
EXTENSION_CELLS = ("VAH30", "VAH15", "VAH12", "VAH11")


def test_extension_cells_use_existing_supported_pipeline() -> None:
    assert set(EXTENSION_CELLS) <= set(SUPPORTED_CELLS)
    assert set(EXTENSION_CELLS) <= set(RPT_SUPPORTED_CELLS)


@pytest.mark.integration
def test_real_extension4_raw_rpt_dynamic_and_adapter_outputs() -> None:
    required = [RPT_OUTPUT, LOOKUP_OUTPUT, ADAPTER_INVENTORY, *(
        DYNAMIC_ROOT / f"{cell}_dynamic_soc.parquet" for cell in EXTENSION_CELLS
    )]
    if not RAW_ROOT.is_dir() or not all(path.is_file() for path in required):
        pytest.skip("CMU extension4 raw or generated artifacts unavailable")

    mission_protocol = load_diagnostic_protocol(MISSION_PROTOCOL)[0]
    rpt_protocol = load_rpt_capacity_protocol(RPT_PROTOCOL)[0]
    expected = {
        "VAH30": (591239, 900, 19, 0, 91800),
        "VAH15": (365851, 542, 11, 0, 66124),
        "VAH12": (1331202, 2302, 39, 8, 142724),
        "VAH11": (1381245, 2202, 36, 9, 224604),
    }
    observations = pd.read_csv(RPT_OUTPUT)
    lookup = pd.read_csv(LOOKUP_OUTPUT)
    inventory = pd.read_csv(ADAPTER_INVENTORY).set_index("cell_id")

    for cell, (raw_rows, candidates, accepted, review, parquet_rows) in expected.items():
        result = load_cmu_v3_main_file(RAW_ROOT / f"{cell}.csv")
        assert result.metadata["input_row_count"] == raw_rows
        assert result.metadata["output_row_count"] == raw_rows
        manifest = enumerate_diagnostic_candidates(contiguous_ns_runs(result.data), mission_protocol)
        assert manifest["screening_status"].eq("diagnostic_screening_match").sum() == candidates
        detected = detect_rpt_capacity_observations(
            result.data, rpt_protocol, mission_protocol=mission_protocol
        )
        assert detected["detector_status"].eq("accepted").sum() == accepted
        assert detected["detector_status"].eq("review_required").sum() == review
        assert lookup.loc[lookup["cell_id"].eq(cell), "capacity_source"].eq("observed_rpt").all()
        dynamic = pd.read_parquet(DYNAMIC_ROOT / f"{cell}_dynamic_soc.parquet")
        assert len(dynamic) == parquet_rows
        assert dynamic[["voltage_v", "current_a", "Temperature__C", "soc_dynamic"]].notna().all().all()
        assert dynamic["soc_dynamic"].between(0.0, 1.0).all()
        assert inventory.at[cell, "vidal_records"] == candidates
        assert inventory.at[cell, "soc_clipping_count"] == 0


def test_core7_inventory_smoke() -> None:
    if not ADAPTER_INVENTORY.is_file():
        pytest.skip("CMU core7 adapter inventory unavailable")
    inventory = pd.read_csv(ADAPTER_INVENTORY)
    assert set(inventory["cell_id"]) == {
        "VAH01", "VAH17", "VAH27", "VAH30", "VAH15", "VAH12", "VAH11",
    }
    assert inventory["cell_id"].is_unique
    assert inventory["vidal_records"].eq(inventory["valid_candidates"]).all()
    assert inventory["vidal_windows"].gt(0).all()
    assert inventory["nonfinite_count"].eq(0).all()
    assert inventory["soc_clipping_count"].eq(0).all()
    assert inventory["blocking_issue"].isna().all()
