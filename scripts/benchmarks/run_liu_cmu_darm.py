#!/usr/bin/env python3
"""Adapt the frozen Liu 2025 Panasonic estimator to selected CMU conditions."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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

from battery_benchmark.benchmarks.cmu_diagnostics import (
    DIAGNOSTIC_METADATA_CASTERS,
    join_diagnostic_fields_by_source_row,
    load_raw_diagnostic_fields,
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
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty
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
WINDOW_LENGTH = 40

# DARM's `build_liu_windows` (src/battery_benchmark/reproductions/liu_2025/
# windows.py) uses stride-one overlapping windows, labelled by the value at
# the window's LAST row only -- i.e. this benchmark's real evaluation
# convention is "last_step_per_window", NOT DTNN/backbone's "all_timesteps".
# This is recorded truthfully in run metadata (see _build_run_metadata)
# rather than force-unified with the other frameworks' convention.
EVALUATION_UNIT = "last_step_per_window"

# Fixed schema for test_predictions.parquet, mirroring run_bian_cmu.py's
# TEST_PREDICTIONS_SCHEMA pattern so every chunk (any cell/method/repeat)
# has identical column order/types. One row = one window (its metadata is
# the window's LAST timestep, matching EVALUATION_UNIT above).
TEST_PREDICTIONS_SCHEMA = pa.schema(
    [
        ("framework", pa.string()),
        ("method", pa.string()),
        ("seed", pa.int64()),
        ("cell_id", pa.string()),
        ("condition", pa.string()),
        ("split", pa.string()),
        ("mission_cycle_id", pa.string()),
        ("window_index", pa.int64()),
        ("timestep_index", pa.int64()),
        ("mission_sequence_order", pa.int64()),
        ("elapsed_candidate_s", pa.float64()),
        ("original_time_s", pa.float64()),
        ("source_row_right", pa.int64()),
        ("voltage_v_raw", pa.float64()),
        ("current_a_raw", pa.float64()),
        ("temperature_c_raw", pa.float64()),
        ("capacity_ah", pa.float64()),
        ("capacity_source", pa.string()),
        ("source_rpt_id", pa.string()),
        ("ns_raw", pa.int64()),
        ("mission_phase", pa.string()),
        ("soc_reference", pa.float64()),
        ("soc_prediction", pa.float64()),
        ("signed_error", pa.float64()),
        ("absolute_error", pa.float64()),
    ]
)


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


def _labelled_windows_with_metadata(records, *, length: int = WINDOW_LENGTH):
    """Diagnostic-only companion to ``_labelled_windows``/``build_liu_windows``:
    builds the EXACT SAME stride-one, last-row-labelled windows (same
    ``range(len(values) - length + 1)`` loop, same
    ``start + length - 1`` label index), but also carries
    mission_sequence_order/elapsed_candidate_s/original_time_s/
    source_row_right for each window's last (labelled) row, sliced at the
    SAME index used for that window's SOC label -- never a separately
    re-derived alignment.

    Does not call or modify ``build_liu_windows`` itself (used elsewhere by
    the Panasonic 25C->0C reproduction and by DARM's own unlabelled target
    loader) -- this is a parallel, read-only path used only for building
    this benchmark's test_predictions.parquet rows. A unit test asserts the
    x/y this produces are identical to ``build_liu_windows``'s output for
    the same records."""
    xs, ys, missions = [], [], []
    metadata_lists = {
        "mission_sequence_order": [],
        "elapsed_candidate_s": [],
        "original_time_s": [],
        "source_row_right": [],
    }
    for record in records:
        frame = record.data
        values = frame[list(FEATURE_ORDER)].to_numpy(np.float32)
        soc = frame["soc"].to_numpy(dtype=np.float64)
        sequence_order = int(record.metadata["sequence_order"])
        elapsed = frame["elapsed_candidate_s"].to_numpy(dtype=np.float64)
        original_time = frame["original_time_s"].to_numpy(dtype=np.float64)
        source_row_right = frame["source_row_right"].to_numpy(dtype=np.int64)
        mission = str(record.metadata["mission_cycle_id"])
        count = 0
        for start in range(len(values) - length + 1):
            last = start + length - 1
            xs.append(values[start : start + length])
            ys.append(float(soc[last]))
            metadata_lists["mission_sequence_order"].append(sequence_order)
            metadata_lists["elapsed_candidate_s"].append(elapsed[last])
            metadata_lists["original_time_s"].append(original_time[last])
            metadata_lists["source_row_right"].append(source_row_right[last])
            count += 1
        missions.append((mission, count))
    if not xs:
        raise ValueError("No complete windows were produced")
    x = np.stack(xs)
    y = np.asarray(ys, dtype=np.float32)
    metadata = {
        "mission_sequence_order": np.asarray(
            metadata_lists["mission_sequence_order"], dtype=np.int64
        ),
        "elapsed_candidate_s": np.asarray(
            metadata_lists["elapsed_candidate_s"], dtype=np.float64
        ),
        "original_time_s": np.asarray(metadata_lists["original_time_s"], dtype=np.float64),
        "source_row_right": np.asarray(metadata_lists["source_row_right"], dtype=np.int64),
    }
    return x, y, missions, metadata


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


@torch.no_grad()
def _predict_all(
    model: DARM, x: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    """Diagnostic-only companion to ``_evaluate``: returns the raw
    per-window prediction array using the IDENTICAL loader construction
    (same TensorDataset, same batch_size, ``shuffle=False``) so predictions
    line up 1:1, in the same order, with both ``y``/``missions`` and the
    metadata arrays from ``_labelled_windows_with_metadata``. Never used to
    change ``_evaluate``'s own metrics -- a test recomputes
    overall_mae_percent/overall_rmse_percent from these predictions and
    asserts they match ``_evaluate``'s output exactly."""
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=min(batch_size, len(x)),
        shuffle=False,
    )
    model.eval()
    prediction = torch.cat([model(batch[0].to(device)).cpu() for batch in loader]).numpy()
    return prediction.astype(np.float64)


def _mission_id_array(missions: list[tuple[str, int]]) -> np.ndarray:
    return np.array(
        [mission for mission, count in missions for _ in range(count)], dtype=object
    )


def _build_prediction_rows(
    *,
    framework: str,
    method: str,
    seed: int,
    cell_id: str,
    condition: str,
    soc_reference: np.ndarray,
    soc_prediction: np.ndarray,
    metadata: dict[str, np.ndarray],
    mission_ids: np.ndarray,
    raw_diagnostic: pd.DataFrame,
) -> list[dict]:
    """Assembles one test_predictions.parquet row per window: this
    method's own soc_prediction, the shared soc_reference/metadata built
    once per cell (identical across Source only / DARM), and the raw-
    parquet diagnostic join (Ns/mission_phase/raw V-I-T/capacity
    provenance) keyed by this window's own last-row source_row_right --
    never reverse-derived from x, never re-guessed after the fact."""
    joined = join_diagnostic_fields_by_source_row(
        metadata["source_row_right"], raw_diagnostic
    )
    n = len(soc_reference)
    rows = []
    for i in range(n):
        row = {
            "framework": framework,
            "method": method,
            "seed": seed,
            "cell_id": cell_id,
            "condition": condition,
            "split": "test",
            "mission_cycle_id": str(mission_ids[i]),
            "window_index": i,
            "timestep_index": WINDOW_LENGTH - 1,
            "mission_sequence_order": int(metadata["mission_sequence_order"][i]),
            "elapsed_candidate_s": float(metadata["elapsed_candidate_s"][i]),
            "original_time_s": float(metadata["original_time_s"][i]),
            "source_row_right": int(metadata["source_row_right"][i]),
            "soc_reference": float(soc_reference[i]),
            "soc_prediction": float(soc_prediction[i]),
            "signed_error": float(soc_prediction[i] - soc_reference[i]),
            "absolute_error": float(abs(soc_prediction[i] - soc_reference[i])),
        }
        for column in (
            "voltage_v_raw",
            "current_a_raw",
            "temperature_c_raw",
            "capacity_ah",
            "capacity_source",
            "source_rpt_id",
            "ns_raw",
            "mission_phase",
        ):
            value = joined[column][i]
            row[column] = None if value is None else DIAGNOSTIC_METADATA_CASTERS[column](value)
        rows.append(row)
    return rows


def _flatten_training_history(
    epoch_diagnostics: list[dict],
    *,
    framework: str,
    method: str,
    seed: int,
    cell_id: str,
    learning_rate: float,
    target_bn_mode: str,
    adversarial_update: str,
    adversarial_schedule: str,
    grl_strength: float,
) -> list[dict]:
    """Exports ``adapt()``'s own returned per-epoch diagnostics list
    (previously discarded at the ``run_liu_cmu_darm.py`` call site --
    ``adapt()`` already computes and even persists this list to
    ``adaptation_diagnostics.json`` as a side effect, it was just never
    read back here) as flat training_history.csv rows. Does not retrain
    and does not call ``adapt()`` a second time -- purely a reshape of
    ``adapt()``'s existing return value.

    Nested per-epoch diagnostics that don't fit a flat numeric column
    (gradient_l2/gradient_cosine per-loss-term dicts, the per-layer BN
    buffer drift dict) are preserved verbatim as JSON-string columns
    rather than dropped, per the "never invent, never drop a real
    diagnostic field" instruction.

    DARM's real adaptation loop (see run_reproduction.py's ``adapt``) is
    called here with ``diagnostic_loaders={}`` (no target labels used for
    any per-epoch validation), so validation_mae/validation_rmse are
    genuinely null and validation_evaluated is genuinely False for every
    epoch -- this reports that real (label-free) behavior truthfully
    rather than forcing Vidal/DTNN's diagnostic-validation convention onto
    a method that doesn't do that."""
    rows = []
    for entry in epoch_diagnostics:
        training = entry["training"]
        rows.append(
            {
                "framework": framework,
                "method": method,
                "seed": seed,
                "cell_id": cell_id,
                "epoch": entry["epoch"],
                "learning_rate": learning_rate,
                "runtime_seconds": None,
                "validation_mae": None,
                "validation_rmse": None,
                "validation_evaluated": False,
                "target_bn_mode": target_bn_mode,
                "adversarial_update": adversarial_update,
                "adversarial_schedule": adversarial_schedule,
                "grl_strength": grl_strength,
                "adversarial_weight": training.get("adversarial_weight"),
                "adversarial_loss": training.get("adversarial"),
                "reconstruction_loss": training.get("reconstruction"),
                "mmd_loss": training.get("mmd"),
                "total_loss": training.get("total"),
                "discriminator_loss": training.get("discriminator"),
                "source_domain_accuracy": training.get("source_domain_accuracy"),
                "target_domain_accuracy": training.get("target_domain_accuracy"),
                "overall_domain_accuracy": training.get("overall_domain_accuracy"),
                "mean_p_source_given_source": training.get("mean_p_source_given_source"),
                "mean_p_source_given_target": training.get("mean_p_source_given_target"),
                "target_feature_rms_drift": training.get("target_feature_rms_drift"),
                "target_bn_running_stat_drift_json": json.dumps(
                    training.get("target_bn_running_stat_drift")
                ),
                "gradient_l2_json": json.dumps(training.get("gradient_l2")),
                "gradient_cosine_json": json.dumps(training.get("gradient_cosine")),
            }
        )
    return rows


def _check_resume_cmu_input_sha256(result_path: Path, expected_sha256: str) -> None:
    """Resume-safety check, run ONLY at the point where --resume is about to
    skip re-running a cell because ``result_path`` already exists -- never
    anywhere else, and never changes what a non-resumed run trains,
    adapts, or evaluates.

    An existing ``results.json`` must have been produced from the SAME CMU
    data version this invocation just hash-verified (against the manifest)
    for this cell. If it wasn't -- either because the hash differs, or
    because the field is entirely absent (an old results.json from before
    this check existed) -- resuming from it would silently reuse a stale
    data version's Source-only/DARM results, so this raises immediately
    rather than skipping, overwriting, or deleting anything.
    """
    existing = json.loads(result_path.read_text(encoding="utf-8"))
    existing_sha256 = existing.get("cmu_input_sha256")
    if existing_sha256 is None:
        raise ValueError(
            f"{result_path}: existing result has no cmu_input_sha256 recorded -- "
            "cannot verify it was produced from the current CMU data version; "
            "refusing to --resume from it. Rerun without --resume, or use a "
            "fresh --output-root."
        )
    if existing_sha256 != expected_sha256:
        raise ValueError(
            f"{result_path}: existing result's cmu_input_sha256={existing_sha256!r} "
            f"does not match the current CMU data version's hash "
            f"{expected_sha256!r} -- refusing to --resume from a different CMU "
            "data version. Rerun without --resume, or use a fresh --output-root."
        )


def _check_resume_diagnostic_sidecars(cell_output: Path) -> None:
    """Resume-safety companion to ``_check_resume_cmu_input_sha256``: an
    existing, hash-verified ``results.json`` is only actually safe to
    resume from if its diagnostic sidecars -- ``test_predictions.parquet``
    and ``training_history.csv`` -- are also still present on disk (both
    are read back by ``_rebuild_test_predictions``/``_rebuild_training_history``
    at the end of ``main()``). If either is missing, raise immediately
    rather than silently emitting an incomplete/empty aggregate or
    fabricating diagnostic rows for the skipped cell.
    """
    missing = [
        name
        for name in ("test_predictions.parquet", "training_history.csv")
        if not (cell_output / name).is_file()
    ]
    if missing:
        raise ValueError(
            f"{cell_output}: --resume requested but this cell's diagnostic "
            f"sidecar(s) {missing} are missing -- cannot safely resume from "
            "this old result without regenerating or fabricating diagnostics. "
            "Rerun without --resume, or use a fresh --output-root."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panasonic-data-root", type=Path, required=True)
    parser.add_argument(
        "--cmu-base-root",
        type=Path,
        default=REPO_ROOT / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1_ns_relabel",
    )
    parser.add_argument(
        "--cmu-condition-root",
        type=Path,
        default=REPO_ROOT
        / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1_extension4_ns_relabel",
    )
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        # Isolated from the pre-_ns_relabel results (previously
        # ".../liu_2025_darm") -- a --resume against this new directory can
        # never accidentally pick up a result produced from the old CMU
        # data version, since that directory is never written to or read
        # from by this default. The old directory itself is left exactly
        # as it was: never deleted, overwritten, or migrated.
        default=REPO_ROOT / "outputs/benchmarks/panasonic_to_cmu_evtol/liu_2025_darm_ns_relabel_v1",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=REPO_ROOT / "data/manifests/liu_2025/panasonic_25c_to_0c.csv",
    )
    parser.add_argument(
        "--cmu-manifest",
        type=Path,
        default=REPO_ROOT
        / "data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1_ns_relabel.csv",
    )
    parser.add_argument("--cells", nargs="+", choices=tuple(CELLS), default=list(CELLS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _build_run_metadata(args: argparse.Namespace) -> dict:
    return {
        "benchmark": "liu_2025_panasonic_25c_to_cmu_conditions",
        "framework": "liu_2025_darm",
        "methods": ["Source only", "DARM"],
        "architecture": "DARM",
        "feature_order": list(FEATURE_ORDER),
        "window_length": WINDOW_LENGTH,
        "evaluation_unit": EVALUATION_UNIT,
        "evaluation_unit_note": (
            "DARM windows are stride-one and overlapping (build_liu_windows); "
            "each window's SOC label/prediction is the value at the window's "
            "LAST row only, unlike DTNN/backbone's all_timesteps convention. "
            "Never force-unified across frameworks -- see spec Section III."
        ),
        "metric_units": {
            "soc_reference": "fraction",
            "soc_prediction": "fraction",
            "signed_error": "fraction",
            "absolute_error": "fraction",
            "overall_mae_percent": "percent",
            "overall_rmse_percent": "percent",
            "mission_macro_mae_percent": "percent",
            "mission_macro_rmse_percent": "percent",
        },
        "target_bn_mode": "frozen_stats",
        "adversarial_update": "grl_joint",
        "adversarial_schedule": "constant",
        "grl_strength": 0.1,
        "adaptation_learning_rate": 0.0001,
        "adaptation_epochs": args.epochs,
        "seed": args.seed,
        "target_soc_labels_used_for_training": False,
        "target_soc_labels_used_for_adaptation": False,
        "target_soc_labels_used_for_checkpoint_selection": False,
        "target_validation_labels_used_for_diagnostics": False,
        "target_validation_labels_used_for_training": False,
        "target_validation_labels_used_for_checkpoint_selection": False,
        "target_soc_reference_used_for_posthoc_evaluation": True,
        "label_is_ground_truth": False,
        "formal_benchmark_eligible": False,
        "git_commit": get_full_commit_hash(REPO_ROOT),
        "git_dirty": is_dirty(REPO_ROOT),
        # Explicit record of which CMU dataset build and manifest this run
        # read from, and which Ns -> mission-phase vocabulary its
        # diagnostic fields use -- diagnostic/provenance-only, does not
        # affect model, adaptation, training, or evaluation logic.
        "cmu_base_root": str(args.cmu_base_root),
        "cmu_condition_root": str(args.cmu_condition_root),
        "cmu_manifest_path": str(args.cmu_manifest),
        "cmu_manifest_sha256": sha256_file(args.cmu_manifest),
        "phase_definition": "canonical Ns mapping: 4=take_off, 5=cruise, 6=landing",
    }


def _rebuild_test_predictions(output_root: Path) -> None:
    """Rebuilds the top-level test_predictions.parquet from every per-cell
    ``<cell_id>/test_predictions.parquet`` sidecar found on disk -- never
    from this invocation's in-memory rows alone -- so a `--resume`'d
    invocation that skips already-completed cells still ends up with a
    complete merged file. Mirrors the merge-not-overwrite
    `_rebuild_cell_metrics` pattern already used elsewhere in this repo for
    cell_metrics.csv/run_metadata.json."""
    tables = [
        pq.read_table(path)
        for path in sorted(output_root.glob("*/test_predictions.parquet"))
    ]
    if not tables:
        return
    merged = pa.concat_tables(tables)
    pq.write_table(merged, output_root / "test_predictions.parquet")


def _rebuild_training_history(output_root: Path) -> None:
    """Rebuilds the top-level training_history.csv from every per-cell
    ``<cell_id>/training_history.csv`` sidecar found on disk, same
    merge-not-overwrite rationale as ``_rebuild_test_predictions``."""
    frames = [
        pd.read_csv(path) for path in sorted(output_root.glob("*/training_history.csv"))
    ]
    if not frames:
        return
    pd.concat(frames, ignore_index=True).to_csv(
        output_root / "training_history.csv", index=False
    )


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

    # Per-cell prediction/training-history sidecars (never a single
    # truncate-on-open writer for the whole run): a `--resume`'d
    # invocation that skips already-completed cells (see the
    # `result_path.is_file()` check below) still ends up with that cell's
    # predictions/history in the final merged files, rebuilt fresh from
    # disk at the end of main() via _rebuild_test_predictions/
    # _rebuild_training_history.
    for cell_id in args.cells:
        cell_output = args.output_root / cell_id
        cell_output.mkdir(parents=True, exist_ok=True)
        result_path = cell_output / "results.json"

        # Hoisted out of the "only proceed if not resumed" branch below
        # (unlike model construction/loading, which stays lazy and only
        # happens when this cell is actually re-run): computing/verifying
        # this cell's CMU input hash against the manifest is cheap, and the
        # resume-safety check below needs a freshly-verified `input_sha256`
        # regardless of whether this cell ends up being skipped.
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
            # Resume safety: only --resume may trigger a skip (previously
            # this skipped whenever results.json existed, regardless of
            # --resume -- fixed here). An existing result JSON must also
            # have been produced from the SAME CMU data version this
            # invocation just hash-verified above, and its diagnostic
            # sidecars must still be present, or this raises immediately
            # rather than silently reusing/overwriting/deleting anything.
            _check_resume_cmu_input_sha256(result_path, input_sha256)
            _check_resume_diagnostic_sidecars(cell_output)
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
        # source_path/input_sha256 already computed and manifest-verified
        # above (hoisted so the resume-safety check has them even when this
        # cell is skipped) -- reused here, not recomputed.
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

        # Diagnostic-only: identical windows/labels via a parallel (never
        # touching build_liu_windows) path that also carries per-window
        # metadata, plus the raw-parquet Ns/V/I/T/capacity join, used only
        # to build test_predictions.parquet rows below. Normalized with the
        # SAME fitted `normalizer` as test_x, so predictions from this
        # array are numerically identical to predictions from test_x for
        # the same window -- verified below via the diag_test_y == test_y
        # equality check (and by a dedicated unit test).
        _, diag_test_y, diag_missions, diag_metadata = _labelled_windows_with_metadata(
            test_records
        )
        diag_test_x, _ = build_liu_windows(
            test_records, require_labels=True, feature_order=FEATURE_ORDER
        )
        diag_test_x = normalizer.transform(diag_test_x).astype(np.float32)
        if not np.array_equal(diag_test_y, test_y):
            raise RuntimeError(
                f"{cell_id}: diagnostic window labels diverged from build_liu_windows"
            )
        diag_mission_ids = _mission_id_array(diag_missions)
        raw_diagnostic = load_raw_diagnostic_fields(source_path)

        source_only = _evaluate(
            model, test_x, test_y, missions, args.batch_size, device
        )
        source_only_prediction = _predict_all(model, diag_test_x, args.batch_size, device)
        source_only_prediction_rows = _build_prediction_rows(
            framework="liu_2025_darm",
            method="Source only",
            seed=args.seed,
            cell_id=cell_id,
            condition=CELLS[cell_id],
            soc_reference=diag_test_y.astype(np.float64),
            soc_prediction=source_only_prediction,
            metadata=diag_metadata,
            mission_ids=diag_mission_ids,
            raw_diagnostic=raw_diagnostic,
        )

        adapt_started = time.perf_counter()
        training, epoch_diagnostics = adapt(
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
        adaptation_wall_seconds = time.perf_counter() - adapt_started
        darm = _evaluate(model, test_x, test_y, missions, args.batch_size, device)
        darm_prediction = _predict_all(model, diag_test_x, args.batch_size, device)
        darm_prediction_rows = _build_prediction_rows(
            framework="liu_2025_darm",
            method="DARM",
            seed=args.seed,
            cell_id=cell_id,
            condition=CELLS[cell_id],
            soc_reference=diag_test_y.astype(np.float64),
            soc_prediction=darm_prediction,
            metadata=diag_metadata,
            mission_ids=diag_mission_ids,
            raw_diagnostic=raw_diagnostic,
        )

        cell_predictions_path = cell_output / "test_predictions.parquet"
        cell_prediction_rows = source_only_prediction_rows + darm_prediction_rows
        if cell_prediction_rows:
            pq.write_table(
                pa.Table.from_pylist(cell_prediction_rows, schema=TEST_PREDICTIONS_SCHEMA),
                cell_predictions_path,
            )

        cell_training_history_rows = _flatten_training_history(
            epoch_diagnostics,
            framework="liu_2025_darm",
            method="DARM",
            seed=args.seed,
            cell_id=cell_id,
            learning_rate=0.0001,
            target_bn_mode="frozen_stats",
            adversarial_update="grl_joint",
            adversarial_schedule="constant",
            grl_strength=0.1,
        )
        pd.DataFrame(cell_training_history_rows).to_csv(
            cell_output / "training_history.csv", index=False
        )

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
            "adaptation_wall_seconds": adaptation_wall_seconds,
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
    _rebuild_test_predictions(args.output_root)
    _rebuild_training_history(args.output_root)
    run_metadata_path = args.output_root / "run_metadata.json"
    run_metadata = _build_run_metadata(args)
    if run_metadata_path.is_file():
        existing = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        existing.update(run_metadata)
        run_metadata = existing
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
