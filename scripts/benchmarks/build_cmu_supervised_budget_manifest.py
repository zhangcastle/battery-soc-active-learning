#!/usr/bin/env python3
"""Build the shared chronological CMU split and supervision-budget manifest."""

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


CELLS = ("VAH01", "VAH17", "VAH27", "VAH30", "VAH15", "VAH12", "VAH11")
BUDGETS = (1, 5, 10)
OUTPUT_COLUMNS = [
    "manifest_version",
    "benchmark_status",
    "cell_id",
    "condition",
    "mission_cycle_id",
    "chronological_rank_in_cell",
    "eligible_missions_in_cell",
    "split",
    "split_rank",
    "train_missions_in_cell",
    "validation_missions_in_cell",
    "test_missions_in_cell",
    "budget_1pct",
    "budget_5pct",
    "budget_10pct",
    "vidal_partial",
    "vidal_full",
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
    conditions: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for cell_id in CELLS:
        inventory = inventories[cell_id]
        total = len(inventory)
        n_train = math.floor(total * 0.60)
        n_validation = math.floor(total * 0.20)
        n_test = total - n_train - n_validation
        budget_counts = {
            budget: math.ceil(n_train * budget / 100) for budget in BUDGETS
        }
        partial_count = math.ceil(n_train * 0.40)
        source_path = source_paths[cell_id]
        source_sha256 = sha256_file(source_path)

        for index, mission in enumerate(inventory.to_dict("records")):
            if index < n_train:
                split = "train"
                split_rank = index + 1
            elif index < n_train + n_validation:
                split = "validation"
                split_rank = index - n_train + 1
            else:
                split = "test"
                split_rank = index - n_train - n_validation + 1

            rows.append(
                {
                    "manifest_version": "cmu_evtol_supervised_budget_v0_1",
                    "benchmark_status": "development_only",
                    "cell_id": cell_id,
                    "condition": conditions[cell_id],
                    "mission_cycle_id": mission["mission_cycle_id"],
                    "chronological_rank_in_cell": mission[
                        "chronological_rank_in_cell"
                    ],
                    "eligible_missions_in_cell": total,
                    "split": split,
                    "split_rank": split_rank,
                    "train_missions_in_cell": n_train,
                    "validation_missions_in_cell": n_validation,
                    "test_missions_in_cell": n_test,
                    "budget_1pct": split == "train"
                    and split_rank <= budget_counts[1],
                    "budget_5pct": split == "train"
                    and split_rank <= budget_counts[5],
                    "budget_10pct": split == "train"
                    and split_rank <= budget_counts[10],
                    "vidal_partial": split == "train"
                    and split_rank <= partial_count,
                    "vidal_full": split == "train",
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
                    "label_is_ground_truth": mission["label_is_ground_truth"],
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
        "--extension-input-root",
        type=Path,
        default=REPO_ROOT
        / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1_extension4",
    )
    parser.add_argument(
        "--cell-protocol-manifest",
        type=Path,
        default=REPO_ROOT
        / "data/manifests/cmu_evtol/cmu_evtol_vtc6_cell_protocols.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1.csv",
    )
    args = parser.parse_args()

    source_paths = {
        cell_id: (
            args.input_root if cell_id in {"VAH01", "VAH17", "VAH27"}
            else args.extension_input_root
        )
        / f"{cell_id}_dynamic_soc.parquet"
        for cell_id in CELLS
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing core7 dynamic-SOC inputs: {missing}")

    protocol_rows = pd.read_csv(args.cell_protocol_manifest).set_index("cell_id")
    conditions = {
        cell_id: str(protocol_rows.loc[cell_id, "changed_value"]) for cell_id in CELLS
    }
    inventories = {
        cell_id: load_mission_inventory(path)
        for cell_id, path in source_paths.items()
    }
    manifest = build_budget_manifest(inventories, source_paths, conditions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"Wrote {len(manifest)} rows: {args.output}")


if __name__ == "__main__":
    main()
