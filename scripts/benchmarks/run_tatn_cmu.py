#!/usr/bin/env python3
"""Run TATN from one Panasonic 25 C source model on the CMU core seven."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.datasets.cmu_evtol.vidal_adapter import (  # noqa: E402
    load_cmu_dynamic_soc_for_vidal,
    load_cmu_features_for_uda,
)
from battery_benchmark.reproductions.liu_2025.cmu_protocol import (  # noqa: E402
    select_duration_matched_missions,
)
from battery_benchmark.reproductions.shen_2023.data import (  # noqa: E402
    PROFILES,
    ShenCycle,
    load_temperature_cycles_and_range,
)
from battery_benchmark.reproductions.shen_tatn.train import (  # noqa: E402
    adapt,
    load_released_source,
    load_source,
    predict,
    train_source,
)
from battery_benchmark.utils.device import get_device  # noqa: E402
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty  # noqa: E402
from battery_benchmark.utils.hashing import sha256_file  # noqa: E402

CELLS = ("VAH01", "VAH17", "VAH27", "VAH30", "VAH15", "VAH12", "VAH11")
BASELINE_CELLS = {"VAH01", "VAH17", "VAH27"}
FEATURES = ("current", "voltage", "temperature")
OFFICIAL_PRE25_SHA256 = "f05475fc627bfb92304bc357b7f0ce81da4bceb724410409e51c65a3d0b969c5"

# TATN hyperparameters, unchanged from their original inline literal values
# below -- named here only so run_metadata.json/each cell's result.json can
# record the exact configuration a run used (for --resume verification and
# provenance), never to change what train_source()/adapt() are called with.
TATN_LEARNING_RATE = 1e-3  # used for both train_source() and adapt()
TATN_LAMBDAS = (1.0, 1.0, 1.0)
SOURCE_EARLY_STOPPING_PATIENCE = 30
SOURCE_MIN_EPOCHS = 60
TRANSFER_EARLY_STOPPING_PATIENCE = 15
TRANSFER_MIN_EPOCHS = 30

# _evaluate() scores every raw timestep inside each non-overlapping 90-step
# window (via predict()'s flattened truth/estimate arrays), not just a
# last-step-per-window summary -- same convention as DTNN/backbone/SFTNN.
EVALUATION_UNIT = "all_timesteps"

# Fields every cell's result.json records so a later --resume can verify the
# on-disk result was produced from the SAME data version and configuration
# this invocation is about to use, before ever treating it as reusable.
RESUME_CHECK_FIELDS = (
    "cmu_input_sha256",
    "manifest_sha256",
    "source_checkpoint_sha256",
    "source_kind",
    "scenario",
    "source_epochs",
    "transfer_epochs",
    "batch_size",
    "seed",
    "tensor_layout",
    "tatn_learning_rate",
    "tatn_lambdas",
    "source_early_stopping_patience",
    "source_min_epochs",
    "transfer_early_stopping_patience",
    "transfer_min_epochs",
)


def _cell_path(cell_id: str, base_root: Path, condition_root: Path) -> Path:
    root = base_root if cell_id in BASELINE_CELLS else condition_root
    return root / f"{cell_id}_dynamic_soc.parquet"


def _records_for_ids(records, mission_ids: set[str]):
    return [
        record
        for record in records
        if str(record.metadata["mission_cycle_id"]) in mission_ids
    ]


def _to_shen_cycles(
    records,
    feature_min: np.ndarray,
    feature_max: np.ndarray,
    *,
    require_labels: bool,
) -> list[ShenCycle]:
    scale = np.where(feature_max == feature_min, 1.0, feature_max - feature_min)
    cycles = []
    for record in records:
        frame = record.data
        if len(frame) < 90:
            raise ValueError(f"{record.metadata['mission_cycle_id']}: fewer than 90 samples")
        features = frame.loc[:, list(FEATURES)].to_numpy(dtype=np.float64)
        soc = (
            frame["soc"].to_numpy(dtype=np.float32)
            if require_labels
            else np.zeros(len(frame), dtype=np.float32)
        )
        cycles.append(
            ShenCycle(
                profile=str(record.metadata["mission_cycle_id"]),
                temperature=int(round(float(frame["temperature"].mean()))),
                features=((features - feature_min) / scale).astype(np.float32),
                soc=soc,
                paths=(Path(str(record.metadata.get("source_path", "<cmu>"))),),
            )
        )
    return cycles


def _evaluate(
    model,
    cycles: list[ShenCycle],
    device: torch.device,
    tensor_layout: str,
) -> dict[str, float | int]:
    truth, estimate = predict(model, cycles, device, tensor_layout)
    error = estimate - truth
    mission_rmse, mission_mae, start = [], [], 0
    for cycle in cycles:
        length = len(cycle.soc) // 90 * 90
        mission_error = error[start : start + length]
        mission_rmse.append(float(np.sqrt(np.mean(mission_error**2))))
        mission_mae.append(float(np.mean(np.abs(mission_error))))
        start += length
    return {
        "overall_rmse_percent": float(np.sqrt(np.mean(error**2)) * 100),
        "overall_mae_percent": float(np.mean(np.abs(error)) * 100),
        "mission_macro_rmse_percent": float(np.mean(mission_rmse) * 100),
        "mission_macro_mae_percent": float(np.mean(mission_mae) * 100),
        "test_missions": len(cycles),
        "test_points": len(truth),
    }


def _check_resume_tatn_result(
    *, result_path: Path, cell_output: Path, expected: dict[str, object]
) -> None:
    """Resume-safety check, run ONLY at the point where --resume is about to
    skip re-running a cell because ``result_path`` already exists -- never
    anywhere else, and never changes what a non-resumed run trains/
    predicts/evaluates.

    Every field in ``RESUME_CHECK_FIELDS`` (the current CMU parquet hash,
    manifest hash, source checkpoint hash, source_kind/scenario, epoch/
    batch-size/seed/tensor-layout configuration, and the TATN
    learning-rate/lambda/early-stopping/min-epochs configuration) must be
    present in the existing result JSON and equal this invocation's
    freshly-computed value exactly. A missing field (an old result
    predating this check) or ANY mismatch (a different CMU data version,
    manifest, source checkpoint, scenario, or hyperparameter) means the
    on-disk result cannot be trusted to represent the CURRENT
    configuration, and must never be silently resumed, overwritten, or
    deleted. The result's ``target.pt`` checkpoint sidecar must also exist.
    Raises ValueError immediately in every failing case instead."""
    existing = json.loads(result_path.read_text(encoding="utf-8"))
    for field in RESUME_CHECK_FIELDS:
        if field not in existing:
            raise ValueError(
                f"{result_path}: existing result has no {field!r} recorded -- "
                "cannot verify it was produced from the current configuration; "
                "refusing to --resume from it. Rerun without --resume, or use "
                "a fresh --output-root."
            )
        if existing[field] != expected[field]:
            raise ValueError(
                f"{result_path}: existing result's {field!r}={existing[field]!r} "
                f"does not match the current run's {field!r}={expected[field]!r} "
                "-- refusing to --resume from a different configuration. Rerun "
                "without --resume, or use a fresh --output-root."
            )
    target_checkpoint = cell_output / "target.pt"
    if not target_checkpoint.is_file():
        raise ValueError(
            f"{result_path}: --resume would skip this cell, but its target "
            f"checkpoint {target_checkpoint} is missing. Rerun without "
            "--resume, or use a fresh --output-root."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panasonic-data-root", type=Path, required=True)
    parser.add_argument(
        "--cmu-base-root", type=Path,
        default=REPO_ROOT
        / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1_ns_relabel",
    )
    parser.add_argument(
        "--cmu-condition-root", type=Path,
        default=REPO_ROOT
        / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1_extension4_ns_relabel",
    )
    parser.add_argument(
        "--output-root", type=Path,
        # Independent from any previously-used --output-root: TATN never had
        # a default output directory before (it was a required flag), so
        # this is simply the new, isolated default location -- never reads
        # from or writes into a caller-specified older directory.
        default=REPO_ROOT / "outputs/benchmarks/panasonic_to_cmu_evtol/tatn_ns_relabel_v1",
    )
    parser.add_argument("--source-kind", choices=("trained", "released"), required=True)
    parser.add_argument("--trained-checkpoint", type=Path)
    parser.add_argument("--released-checkpoint", type=Path)
    parser.add_argument(
        "--scenario",
        choices=("unsupervised", "semisupervised"),
        default="unsupervised",
    )
    parser.add_argument(
        "--cmu-manifest",
        type=Path,
        default=REPO_ROOT
        / "data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1_ns_relabel.csv",
    )
    parser.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS))
    parser.add_argument("--source-epochs", type=int, default=300)
    parser.add_argument("--transfer-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_cycles, feature_min, feature_max = load_temperature_cycles_and_range(
        args.panasonic_data_root, 25
    )

    if args.source_kind == "trained":
        tensor_layout = "paper"
        source_checkpoint = args.trained_checkpoint or args.output_root / "source_25C.pt"
        if source_checkpoint.is_file():
            source_model = load_source(source_checkpoint, device)
        else:
            source_model, _ = train_source(
                source_cycles,
                epochs=args.source_epochs,
                batch_size=args.batch_size,
                learning_rate=TATN_LEARNING_RATE,
                device=device,
                seed=args.seed,
                checkpoint=source_checkpoint,
                early_stopping_patience=SOURCE_EARLY_STOPPING_PATIENCE,
                min_epochs=SOURCE_MIN_EPOCHS,
                tensor_layout=tensor_layout,
            )
    else:
        tensor_layout = "released_repo"
        source_checkpoint = args.released_checkpoint
        if source_checkpoint is None or not source_checkpoint.is_file():
            raise FileNotFoundError("--released-checkpoint must point to pre-25.pt")
        if sha256_file(source_checkpoint) != OFFICIAL_PRE25_SHA256:
            raise ValueError("Released pre-25.pt SHA-256 mismatch")
        source_model = load_released_source(source_checkpoint, device)

    source_checkpoint_sha256 = sha256_file(source_checkpoint)
    source_truth, source_estimate = predict(
        source_model, list(source_cycles.values()), device, tensor_layout
    )
    source_sanity = {
        "mae_percent": float(np.mean(np.abs(source_estimate - source_truth)) * 100),
        "rmse_percent": float(np.sqrt(np.mean((source_estimate - source_truth) ** 2)) * 100),
    }
    manifest = pd.read_csv(args.cmu_manifest)
    manifest_sha256 = sha256_file(args.cmu_manifest)
    resume_expected = {
        "manifest_sha256": manifest_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_kind": args.source_kind,
        "scenario": args.scenario,
        "source_epochs": args.source_epochs if args.source_kind == "trained" else None,
        "transfer_epochs": args.transfer_epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "tensor_layout": tensor_layout,
        "tatn_learning_rate": TATN_LEARNING_RATE,
        "tatn_lambdas": list(TATN_LAMBDAS),
        "source_early_stopping_patience": (
            SOURCE_EARLY_STOPPING_PATIENCE if args.source_kind == "trained" else None
        ),
        "source_min_epochs": SOURCE_MIN_EPOCHS if args.source_kind == "trained" else None,
        "transfer_early_stopping_patience": TRANSFER_EARLY_STOPPING_PATIENCE,
        "transfer_min_epochs": TRANSFER_MIN_EPOCHS,
    }
    rows = []

    for cell_index, cell_id in enumerate(args.cells):
        cell_output = args.output_root / cell_id
        result_path = cell_output / "results.json"
        source_path = _cell_path(cell_id, args.cmu_base_root, args.cmu_condition_root)
        input_sha256 = sha256_file(source_path)
        expected_hashes = set(
            manifest.loc[
                manifest["cell_id"] == cell_id, "source_artifact_sha256"
            ].astype(str)
        )
        if expected_hashes != {input_sha256}:
            raise ValueError(f"{cell_id}: CMU input hash does not match manifest")

        if args.resume and result_path.is_file():
            _check_resume_tatn_result(
                result_path=result_path,
                cell_output=cell_output,
                expected={"cmu_input_sha256": input_sha256, **resume_expected},
            )
            rows.extend(json.loads(result_path.read_text(encoding="utf-8"))["metrics"])
            continue
        cell_output.mkdir(parents=True, exist_ok=True)

        adaptation_ids, _, test_ids, budget = select_duration_matched_missions(
            manifest, cell_id
        )
        feature_records = load_cmu_features_for_uda([source_path], target_hz=1.0).records
        labelled_records = load_cmu_dynamic_soc_for_vidal(
            [source_path], target_hz=1.0
        ).records
        adaptation_records = _records_for_ids(feature_records, adaptation_ids)
        adaptation_cycles = _to_shen_cycles(
            adaptation_records,
            feature_min,
            feature_max,
            require_labels=False,
        )
        labeled_profile = None
        if args.scenario == "semisupervised":
            selected = manifest.loc[
                (manifest["cell_id"] == cell_id)
                & manifest["mission_cycle_id"].astype(str).isin(adaptation_ids)
            ].sort_values("split_rank", kind="stable")
            labeled_profile = str(selected.iloc[0]["mission_cycle_id"])
            labeled_cycle = _to_shen_cycles(
                _records_for_ids(labelled_records, {labeled_profile}),
                feature_min,
                feature_max,
                require_labels=True,
            )[0]
            adaptation_cycles = [
                labeled_cycle if cycle.profile == labeled_profile else cycle
                for cycle in adaptation_cycles
            ]
        test_cycles = _to_shen_cycles(
            _records_for_ids(labelled_records, test_ids),
            feature_min,
            feature_max,
            require_labels=True,
        )
        if not adaptation_cycles or not test_cycles:
            raise ValueError(f"{cell_id}: missing adaptation or test missions")

        source_only = _evaluate(source_model, test_cycles, device, tensor_layout)
        transferred, history, target_info = adapt(
            source_model,
            source_cycles,
            adaptation_cycles,
            test_profile=None,
            scenario=args.scenario,
            epochs=args.transfer_epochs,
            batch_size=args.batch_size,
            learning_rate=TATN_LEARNING_RATE,
            lambdas=TATN_LAMBDAS,
            device=device,
            seed=args.seed + cell_index,
            labeled_profile=labeled_profile,
            early_stopping_patience=TRANSFER_EARLY_STOPPING_PATIENCE,
            min_epochs=TRANSFER_MIN_EPOCHS,
            tensor_layout=tensor_layout,
        )
        tatn = _evaluate(transferred, test_cycles, device, tensor_layout)
        torch.save(
            {
                "model_state_dict": transferred.state_dict(),
                "history": history,
                "target_info": target_info,
            },
            cell_output / "target.pt",
        )
        metrics = [
            {
                "source_kind": args.source_kind,
                "cell_id": cell_id,
                "condition": budget["condition"],
                "method": method,
                **values,
            }
            for method, values in (("Source only", source_only), ("TATN", tatn))
        ]
        result = {
            "benchmark": "shen_2023_tatn_panasonic_25c_to_cmu_core7",
            "source_checkpoint": str(source_checkpoint),
            "source_sanity": source_sanity,
            "source_profiles": list(PROFILES),
            "source_normalization": "Panasonic 25 C all-nine-profile min-max",
            "source_feature_min": feature_min.tolist(),
            "source_feature_max": feature_max.tolist(),
            "cell_id": cell_id,
            "cmu_input_sha256": input_sha256,
            "cmu_manifest_path": str(args.cmu_manifest),
            "adaptation_budget": budget,
            "adaptation_missions": len(adaptation_cycles),
            "labeled_target_mission": labeled_profile,
            "labeled_target_fraction": 0.5 if labeled_profile else 0.0,
            "test_missions": len(test_cycles),
            "target_info": target_info,
            "epochs_completed": len(history),
            "metrics": metrics,
            "target_soc_labels_used_for_adaptation": labeled_profile is not None,
            "target_soc_labels_used_for_checkpoint_selection": labeled_profile is not None,
            "target_soc_reference_used_for_posthoc_evaluation": True,
            "label_is_ground_truth": False,
            "formal_benchmark_eligible": False,
            "evaluation_unit": EVALUATION_UNIT,
            **resume_expected,
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        rows.extend(metrics)
        pd.DataFrame(rows).to_csv(args.output_root / "condition_metrics.csv", index=False)
        print(
            f"{cell_id}: source={source_only['overall_mae_percent']:.3f}% "
            f"TATN={tatn['overall_mae_percent']:.3f}% "
            f"pseudo_windows={target_info['pseudo_windows']}"
        )

    table = pd.DataFrame(rows)
    table.to_csv(args.output_root / "condition_metrics.csv", index=False)
    (args.output_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "benchmark": "shen_2023_tatn_panasonic_25c_to_cmu_core7",
                "source_checkpoint": str(source_checkpoint),
                "source_sanity": source_sanity,
                # Explicit record of which CMU dataset build and manifest this
                # run read from -- diagnostic/provenance-only, does not affect
                # model, training, or evaluation logic.
                "cmu_base_root": str(args.cmu_base_root),
                "cmu_condition_root": str(args.cmu_condition_root),
                "cmu_manifest_path": str(args.cmu_manifest),
                "phase_definition": "canonical Ns mapping: 4=take_off, 5=cruise, 6=landing",
                "evaluation_unit": EVALUATION_UNIT,
                "evaluation_unit_note": (
                    "predict()/_evaluate score every raw timestep inside each "
                    "non-overlapping 90-step window -- all_timesteps, same "
                    "convention as DTNN/backbone/SFTNN, unlike DARM's "
                    "last_step_per_window."
                ),
                "metric_units": {
                    "overall_rmse_percent": "percent",
                    "overall_mae_percent": "percent",
                    "mission_macro_rmse_percent": "percent",
                    "mission_macro_mae_percent": "percent",
                    "test_missions": "count",
                    "test_points": "count",
                },
                "git_commit": get_full_commit_hash(REPO_ROOT),
                "git_dirty": is_dirty(REPO_ROOT),
                **resume_expected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
