from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/benchmarks/build_cmu_supervised_budget_manifest.py"
MANIFEST = (
    ROOT
    / "data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1.csv"
)
PROTOCOL = (
    ROOT
    / "configs/datasets/cmu_evtol/protocols/cmu_evtol_transfer_benchmark_v0_1.yaml"
)

spec = importlib.util.spec_from_file_location("cmu_budget_manifest", SCRIPT)
assert spec and spec.loader
budget_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget_module)


def _synthetic_inventory(cell_id: str, count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mission_cycle_id": [f"{cell_id}_m{i:03d}" for i in range(count)],
            "cell_id": cell_id,
            "mission_start_source_row": [i * 10 for i in range(count)],
            "mission_end_source_row": [i * 10 + 9 for i in range(count)],
            "mission_start_time_s": [i * 100.0 for i in range(count)],
            "mission_end_time_s": [i * 100.0 + 90.0 for i in range(count)],
            "mission_row_count": [10] * count,
            "label_source": ["development_reference"] * count,
            "label_is_ground_truth": [False] * count,
            "formal_benchmark_eligible": [False] * count,
            "mission_duration_s": [90.0] * count,
            "chronological_rank_in_cell": range(1, count + 1),
        }
    )


def test_builder_creates_per_cell_chronological_splits_and_budgets(
    tmp_path: Path,
) -> None:
    inventories = {
        cell_id: _synthetic_inventory(cell_id, 101 + index)
        for index, cell_id in enumerate(budget_module.CELLS)
    }
    paths = {}
    for cell_id in inventories:
        path = tmp_path / f"{cell_id}.parquet"
        path.write_bytes(cell_id.encode())
        paths[cell_id] = path

    manifest = budget_module.build_budget_manifest(
        inventories, paths, {cell_id: "test" for cell_id in inventories}
    )
    for cell_id, rows in manifest.groupby("cell_id", sort=False):
        total = len(inventories[cell_id])
        n_train = math.floor(total * 0.60)
        n_validation = math.floor(total * 0.20)
        assert rows["split"].tolist() == (
            ["train"] * n_train
            + ["validation"] * n_validation
            + ["test"] * (total - n_train - n_validation)
        )
        for budget in budget_module.BUDGETS:
            assert rows[f"budget_{budget}pct"].sum() == math.ceil(
                n_train * budget / 100
            )
        assert rows["vidal_partial"].sum() == math.ceil(n_train * 0.40)
        assert rows["vidal_full"].sum() == n_train


def test_committed_manifest_is_core7_unique_nested_and_test_isolated() -> None:
    manifest = pd.read_csv(MANIFEST)
    assert len(manifest) == 8326
    assert set(manifest["cell_id"]) == set(budget_module.CELLS)
    assert not manifest.duplicated(["cell_id", "mission_cycle_id"]).any()

    for _, rows in manifest.groupby("cell_id"):
        one = set(rows.loc[rows["budget_1pct"], "mission_cycle_id"])
        five = set(rows.loc[rows["budget_5pct"], "mission_cycle_id"])
        ten = set(rows.loc[rows["budget_10pct"], "mission_cycle_id"])
        partial = set(rows.loc[rows["vidal_partial"], "mission_cycle_id"])
        full = set(rows.loc[rows["vidal_full"], "mission_cycle_id"])
        assert one < five < ten < partial < full
        assert not rows.loc[rows["split"] != "train", [
            "budget_1pct",
            "budget_5pct",
            "budget_10pct",
            "vidal_partial",
            "vidal_full",
        ]].any().any()


def test_committed_manifest_preserves_development_label_status() -> None:
    manifest = pd.read_csv(MANIFEST)
    assert set(manifest["manifest_version"]) == {
        "cmu_evtol_supervised_budget_v0_1"
    }
    assert set(manifest["benchmark_status"]) == {"development_only"}
    assert set(manifest["label_column"]) == {"soc_dynamic"}
    assert set(manifest["label_source"]) == {
        "cmu_causal_dynamic_capacity_reference_development_v1"
    }
    assert not manifest["label_is_ground_truth"].any()
    assert not manifest["formal_benchmark_eligible"].any()
    assert manifest["source_artifact_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_protocol_uses_shared_per_cell_split_manifest() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text())
    benchmark = protocol["development_supervised_benchmark"]
    budgets = protocol["target_label_budgets"]
    assert benchmark["target_cells"] == list(budget_module.CELLS)
    assert benchmark["split_percent"] == {
        "train": 60,
        "validation": 20,
        "test": 20,
    }
    assert benchmark["same_test_missions_for_all_tracks"] is True
    assert budgets["values_percent_of_train"] == [1, 5, 10]
    assert budgets["vidal_partial_percent_of_train"] == 40
    assert budgets["vidal_full_percent_of_train"] == 100
    assert ROOT / benchmark["manifest_path"] == MANIFEST
