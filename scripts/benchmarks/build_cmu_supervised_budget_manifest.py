#!/usr/bin/env python3
"""Build deterministic CMU development-label budgets for the three baseline folds."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.utils.hashing import sha256_file


BUDGETS = (1, 5, 10)
FOLDS = {
    "test_vah01": (("VAH17", "VAH27"), "VAH01"),
    "test_vah17": (("VAH01", "VAH27"), "VAH17"),
    "test_vah27": (("VAH01", "VAH17"), "VAH27"),
}
OUTPUT_COLUMNS = [
    "manifest_version",
    "benchmark_status",
    "fold_id",
    "budget_percent",
    "adaptation_cell",
    "held_out_test_cell",
    "mission_cycle_id",
    "chronological_rank_in_cell",
    "eligible_missions_in_cell",
    "selected_missions_in_cell",
    "mission_start_source_row",
    "mission_end_source_row",
    "mission_start_time_s",
    "mission_end_time_s",
    "mission_duration_s",
    "mission_row_count",
    "label_column",
    "label_source",
    "label_is_ground_truth",
    "formal_benchmark_eligible",
    "source_artifact",
    "source_artifact_sha256",
]


def load_mission_inventory(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=[
            "cell_id",
            "mission_cycle_id",
            "source_row",
            "time_s",
            "label_source",
            "label_is_ground_truth",
            "formal_benchmark_eligible",
        ],
    )
    if frame.empty or frame["cell_id"].nunique() != 1:
        raise ValueError(f"{path}: expected one non-empty cell")
    if frame[["mission_cycle_id", "source_row", "time_s"]].isna().any().any():
        raise ValueError(f"{path}: mission identity/order fields contain nulls")

    inventory = (
        frame.groupby("mission_cycle_id", sort=False)
        .agg(
            cell_id=("cell_id", "first"),
            mission_start_source_row=("source_row", "min"),
            mission_end_source_row=("source_row", "max"),
            mission_start_time_s=("time_s", "min"),
            mission_end_time_s=("time_s", "max"),
            mission_row_count=("source_row", "size"),
            label_source=("label_source", "first"),
            label_is_ground_truth=("label_is_ground_truth", "first"),
            formal_benchmark_eligible=("formal_benchmark_eligible", "first"),
        )
        .reset_index()
        .sort_values(["mission_start_source_row", "mission_cycle_id"], kind="stable")
        .reset_index(drop=True)
    )
    inventory["mission_duration_s"] = (
        inventory["mission_end_time_s"] - inventory["mission_start_time_s"]
    )
    inventory["chronological_rank_in_cell"] = range(1, len(inventory) + 1)
    return inventory


def build_budget_manifest(
    inventories: dict[str, pd.DataFrame],
    source_paths: dict[str, Path],
) -> pd.DataFrame:
    rows: list[dict] = []
    for fold_id, (adaptation_cells, test_cell) in FOLDS.items():
        for cell_id in adaptation_cells:
            inventory = inventories[cell_id]
            eligible_count = len(inventory)
            source_path = source_paths[cell_id]
            source_sha256 = sha256_file(source_path)
            for budget in BUDGETS:
                selected_count = math.ceil(eligible_count * budget / 100)
                for mission in inventory.iloc[:selected_count].to_dict("records"):
                    rows.append(
                        {
                            "manifest_version": "cmu_evtol_supervised_budget_v0_1",
                            "benchmark_status": "development_only",
                            "fold_id": fold_id,
                            "budget_percent": budget,
                            "adaptation_cell": cell_id,
                            "held_out_test_cell": test_cell,
                            "mission_cycle_id": mission["mission_cycle_id"],
                            "chronological_rank_in_cell": mission[
                                "chronological_rank_in_cell"
                            ],
                            "eligible_missions_in_cell": eligible_count,
                            "selected_missions_in_cell": selected_count,
                            "mission_start_source_row": mission[
                                "mission_start_source_row"
                            ],
                            "mission_end_source_row": mission["mission_end_source_row"],
                            "mission_start_time_s": mission["mission_start_time_s"],
                            "mission_end_time_s": mission["mission_end_time_s"],
                            "mission_duration_s": mission["mission_duration_s"],
                            "mission_row_count": mission["mission_row_count"],
                            "label_column": "soc_dynamic",
                            "label_source": mission["label_source"],
                            "label_is_ground_truth": mission[
                                "label_is_ground_truth"
                            ],
                            "formal_benchmark_eligible": mission[
                                "formal_benchmark_eligible"
                            ],
                            "source_artifact": str(
                                source_path.relative_to(REPO_ROOT)
                                if source_path.is_relative_to(REPO_ROOT)
                                else source_path.resolve()
                            ),
                            "source_artifact_sha256": source_sha256,
                        }
                    )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT
        / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1.csv",
    )
    args = parser.parse_args()

    source_paths = {
        cell_id: args.input_root / f"{cell_id}_dynamic_soc.parquet"
        for cell_id in ("VAH01", "VAH17", "VAH27")
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing baseline dynamic-SOC inputs: {missing}")

    inventories = {
        cell_id: load_mission_inventory(path)
        for cell_id, path in source_paths.items()
    }
    manifest = build_budget_manifest(inventories, source_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"Wrote {len(manifest)} rows: {args.output}")


if __name__ == "__main__":
    main()
