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


def test_builder_uses_per_cell_ceil_and_chronological_prefix(tmp_path: Path) -> None:
    inventories = {
        "VAH01": _synthetic_inventory("VAH01", 101),
        "VAH17": _synthetic_inventory("VAH17", 51),
        "VAH27": _synthetic_inventory("VAH27", 21),
    }
    paths = {}
    for cell_id in inventories:
        path = tmp_path / f"{cell_id}.parquet"
        path.write_bytes(cell_id.encode())
        paths[cell_id] = path

    manifest = budget_module.build_budget_manifest(inventories, paths)
    for (fold_id, budget, cell_id), group in manifest.groupby(
        ["fold_id", "budget_percent", "adaptation_cell"]
    ):
        expected = math.ceil(len(inventories[cell_id]) * budget / 100)
        assert len(group) == expected
        assert group["chronological_rank_in_cell"].tolist() == list(
            range(1, expected + 1)
        )
        assert group["selected_missions_in_cell"].eq(expected).all()
        assert cell_id != group["held_out_test_cell"].iloc[0]


def test_committed_manifest_is_nested_isolated_and_unique() -> None:
    manifest = pd.read_csv(MANIFEST)
    assert len(manifest) == 768
    assert not manifest.duplicated(
        ["fold_id", "budget_percent", "adaptation_cell", "mission_cycle_id"]
    ).any()
    assert set(manifest["fold_id"]) == set(budget_module.FOLDS)
    assert set(manifest["budget_percent"]) == set(budget_module.BUDGETS)
    assert (manifest["adaptation_cell"] != manifest["held_out_test_cell"]).all()

    for (fold_id, cell_id), group in manifest.groupby(
        ["fold_id", "adaptation_cell"]
    ):
        selected = {
            budget: set(rows["mission_cycle_id"])
            for budget, rows in group.groupby("budget_percent")
        }
        assert selected[1] < selected[5] < selected[10]
        for budget, rows in group.groupby("budget_percent"):
            eligible = int(rows["eligible_missions_in_cell"].iloc[0])
            expected = math.ceil(eligible * budget / 100)
            assert len(rows) == expected
            assert sorted(rows["chronological_rank_in_cell"]) == list(
                range(1, expected + 1)
            )


def test_committed_manifest_preserves_development_label_status() -> None:
    manifest = pd.read_csv(MANIFEST)
    assert set(manifest["benchmark_status"]) == {"development_only"}
    assert set(manifest["label_column"]) == {"soc_dynamic"}
    assert set(manifest["label_source"]) == {
        "cmu_causal_dynamic_capacity_reference_development_v1"
    }
    assert not manifest["label_is_ground_truth"].any()
    assert not manifest["formal_benchmark_eligible"].any()
    assert manifest["source_artifact_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_protocol_points_all_models_to_one_development_manifest() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text())
    budgets = protocol["target_label_budgets"]
    development = protocol["development_budget_manifest"]
    assert budgets["same_budget_manifest_for_all_models"] is True
    assert budgets["same_manifest_for_transfer_and_scratch"] is True
    assert budgets["rounding_scope"] == "per_adaptation_cell"
    assert development["status"] == "development_only"
    assert ROOT / development["path"] == MANIFEST
    assert development["formal_benchmark_eligible"] is False
