#!/usr/bin/env python3
"""Adapt the frozen Liu 2025 Panasonic estimator to selected CMU conditions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.reproductions.liu_2025.run_reproduction import (
    _legacy_source_protocol_sha256,
    adapt,
    feature_ranges,
    load_record,
    supervised_loader,
    unlabelled_loader,
)

from battery_benchmark.datasets.cmu_evtol.vidal_adapter import (
    load_cmu_dynamic_soc_for_vidal,
    load_cmu_features_for_uda,
)
from battery_benchmark.preprocessing.normalization import MinMaxNormalizer
from battery_benchmark.reproductions.liu_2025.cmu_protocol import (
    select_duration_matched_missions,
)
from battery_benchmark.reproductions.liu_2025.darm import DARM
from battery_benchmark.reproductions.liu_2025.protocol import (
    load_experiment_manifest,
    source_protocol_sha256,
)
from battery_benchmark.reproductions.liu_2025.source_artifact import (
    load_source_artifact,
)
from battery_benchmark.reproductions.liu_2025.windows import build_liu_windows
from battery_benchmark.utils.device import get_device
from battery_benchmark.utils.hashing import sha256_file

CELLS = {
    "VAH01": "baseline protocol",
    "VAH17": "baseline protocol",
    "VAH27": "baseline protocol",
    "VAH30": "35 °C chamber",
    "VAH15": "extended 1000 s cruise",
    "VAH12": "short 400 s cruise",
    "VAH11": "20% lower discharge power",
}
BASELINE_CELLS = {"VAH01", "VAH17", "VAH27"}
FEATURE_ORDER = ("current", "voltage", "temperature")


def _records_for_ids(records, mission_ids: set[str]):
    return [
        record
        for record in records
        if str(record.metadata["mission_cycle_id"]) in mission_ids
    ]


def _cell_path(cell_id: str, base_root: Path, condition_root: Path) -> Path:
    root = base_root if cell_id in BASELINE_CELLS else condition_root
    return root / f"{cell_id}_dynamic_soc.parquet"


def _labelled_windows(records) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    xs, ys, missions = [], [], []
    for record in records:
        x, y = build_liu_windows(
            [record], require_labels=True, feature_order=FEATURE_ORDER
        )
        xs.append(x)
        ys.append(y)
        missions.append((str(record.metadata["mission_cycle_id"]), len(y)))
    return np.concatenate(xs), np.concatenate(ys), missions


@torch.no_grad()
def _evaluate(
    model: DARM,
    x: np.ndarray,
    y: np.ndarray,
    missions: list[tuple[str, int]],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=min(batch_size, len(x)),
        shuffle=False,
    )
    model.eval()
    prediction = torch.cat([model(batch[0].to(device)).cpu() for batch in loader]).numpy()
    error = prediction - y
    mission_rmse, mission_mae, start = [], [], 0
    for _, length in missions:
        mission_error = error[start : start + length]
        mission_rmse.append(float(np.sqrt(np.mean(mission_error**2))))
        mission_mae.append(float(np.mean(np.abs(mission_error))))
        start += length
    return {
        "overall_rmse_percent": float(np.sqrt(np.mean(error**2)) * 100),
        "overall_mae_percent": float(np.mean(np.abs(error)) * 100),
        "mission_macro_rmse_percent": float(np.mean(mission_rmse) * 100),
        "mission_macro_mae_percent": float(np.mean(mission_mae) * 100),
        "test_missions": len(missions),
        "test_windows": len(y),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panasonic-data-root", type=Path, required=True)
    parser.add_argument("--cmu-base-root", type=Path, required=True)
    parser.add_argument("--cmu-condition-root", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=REPO_ROOT / "data/manifests/liu_2025/panasonic_25c_to_0c.csv",
    )
    parser.add_argument(
        "--cmu-manifest",
        type=Path,
        default=REPO_ROOT
        / "data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1.csv",
    )
    parser.add_argument("--cells", nargs="+", choices=tuple(CELLS), default=list(CELLS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.cmu_manifest)
    source_manifest = load_experiment_manifest(
        args.source_manifest, args.panasonic_data_root, verify_hashes=True
    )
    source_records = [
        load_record(row)
        for row in source_manifest.loc[
            source_manifest["role"] == "source_train"
        ].itertuples(index=False)
    ]
    source_x, source_y = build_liu_windows(
        source_records, require_labels=True, feature_order=FEATURE_ORDER
    )
    rows = []

    for cell_id in args.cells:
        cell_output = args.output_root / cell_id
        cell_output.mkdir(parents=True, exist_ok=True)
        result_path = cell_output / "results.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            rows.extend(result["metrics"])
            continue

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = DARM().to(device)
        normalizer = MinMaxNormalizer()
        source_metadata = load_source_artifact(
            args.source_artifact, model, normalizer, device=device
        )
        artifact_source_protocol = source_metadata.get(
            "source_protocol_sha256"
        ) or _legacy_source_protocol_sha256(source_metadata, args.source_manifest)
        if artifact_source_protocol != source_protocol_sha256(source_manifest):
            raise ValueError("Source artifact protocol does not match source manifest")
        if source_metadata.get("feature_order") != list(FEATURE_ORDER):
            raise ValueError("Source artifact feature order does not match paper_ivt")

        source_x_normalized = normalizer.transform(source_x).astype(np.float32)
        source_loader = supervised_loader(
            source_x_normalized, source_y, args.batch_size, True
        )
        source_path = _cell_path(
            cell_id, args.cmu_base_root, args.cmu_condition_root
        )
        input_sha256 = sha256_file(source_path)
        expected_hashes = set(
            manifest.loc[
                manifest["cell_id"] == cell_id, "source_artifact_sha256"
            ].astype(str)
        )
        if expected_hashes != {input_sha256}:
            raise ValueError(f"{cell_id}: CMU input hash does not match manifest")
        adaptation_ids, _, test_ids, budget = select_duration_matched_missions(
            manifest, cell_id
        )

        feature_records = load_cmu_features_for_uda([source_path], target_hz=1.0).records
        feature_ids = {
            str(record.metadata["mission_cycle_id"]) for record in feature_records
        }
        if not adaptation_ids <= feature_ids:
            raise ValueError(f"{cell_id}: adaptation missions missing from CMU input")
        adaptation_records = _records_for_ids(feature_records, adaptation_ids)
        target_x, target_y = build_liu_windows(
            adaptation_records, require_labels=False, feature_order=FEATURE_ORDER
        )
        if target_y is not None:
            raise RuntimeError("CMU adaptation unexpectedly exposed target labels")
        target_x = normalizer.transform(target_x).astype(np.float32)
        target_loader = unlabelled_loader(target_x, args.batch_size)

        labelled_records = load_cmu_dynamic_soc_for_vidal(
            [source_path], target_hz=1.0
        ).records
        labelled_ids = {
            str(record.metadata["mission_cycle_id"]) for record in labelled_records
        }
        if not test_ids <= labelled_ids:
            raise ValueError(f"{cell_id}: test missions missing from CMU input")
        test_records = _records_for_ids(labelled_records, test_ids)
        test_x, test_y, missions = _labelled_windows(test_records)
        test_x = normalizer.transform(test_x).astype(np.float32)

        source_only = _evaluate(
            model, test_x, test_y, missions, args.batch_size, device
        )
        training, _ = adapt(
            model,
            source_loader,
            target_loader,
            args.epochs,
            0.0001,
            1.0,
            10.0,
            1.0,
            20,
            device,
            cell_output,
            args.resume,
            "frozen_stats",
            "grl_joint",
            "constant",
            0.1,
            {},
        )
        darm = _evaluate(model, test_x, test_y, missions, args.batch_size, device)
        metrics = [
            {
                "cell_id": cell_id,
                "condition": CELLS[cell_id],
                "method": method,
                **values,
            }
            for method, values in (("Source only", source_only), ("DARM", darm))
        ]
        result = {
            "benchmark": "liu_2025_panasonic_25c_to_cmu_conditions",
            "cell_id": cell_id,
            "condition": CELLS[cell_id],
            "source_artifact": str(args.source_artifact),
            "source_artifact_sha256": sha256_file(args.source_artifact),
            "source_manifest_sha256": sha256_file(args.source_manifest),
            "cmu_manifest_sha256": sha256_file(args.cmu_manifest),
            "cmu_input_sha256": input_sha256,
            "source_windows": len(source_x),
            "adaptation_budget": budget,
            "adaptation_windows": len(target_x),
            "adaptation_feature_ranges": feature_ranges(target_x, FEATURE_ORDER),
            "target_training": training,
            "metrics": metrics,
            "target_soc_labels_used_for_adaptation": False,
            "target_soc_labels_used_for_checkpoint_selection": False,
            "target_soc_reference_used_for_posthoc_evaluation": True,
            "label_is_ground_truth": False,
            "formal_benchmark_eligible": False,
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        rows.extend(metrics)
        pd.DataFrame(rows).to_csv(args.output_root / "condition_metrics.csv", index=False)
        print(
            f"{cell_id}: source={source_only['overall_rmse_percent']:.3f}% "
            f"DARM={darm['overall_rmse_percent']:.3f}%"
        )

    table = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_root / "condition_metrics.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
