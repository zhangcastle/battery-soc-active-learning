#!/usr/bin/env python3
"""Run TATN from one Panasonic 25 C source model on the CMU core seven."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.benchmarks.cmu_diagnostics import (  # noqa: E402
    DIAGNOSTIC_METADATA_CASTERS,
    join_diagnostic_fields_by_source_row,
    load_raw_diagnostic_fields,
)
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

# Fixed by the TATN architecture (same non-overlapping window length windows()
# uses inside predict()/_evaluate) -- diagnostic-only code below replays this
# EXACT usable = len(cycle.soc) // WINDOW * WINDOW truncation/reshape on
# per-mission metadata, never on features/SOC/predictions themselves.
WINDOW = 90
FRAMEWORK = "shen_tatn"
PHASE_SUMMARY_PHASES = ("take_off", "cruise", "landing")

# Fixed schema for test_predictions.parquet -- one row per raw CMU timestep
# (not per window), covering both "Source only" (frozen source model, no
# adaptation) and "TATN" (post-adapt()) methods per cell, distinguished by
# the method column. source_kind/scenario are TATN-specific additions
# (this script's own CLI configuration) alongside the columns shared with
# the other CMU-transfer benchmarks' prediction diagnostics.
TEST_PREDICTIONS_SCHEMA = pa.schema(
    [
        ("framework", pa.string()),
        ("method", pa.string()),
        ("source_kind", pa.string()),
        ("scenario", pa.string()),
        ("seed", pa.int64()),
        ("cell_id", pa.string()),
        ("condition", pa.string()),
        ("split", pa.string()),
        ("mission_cycle_id", pa.string()),
        ("mission_sequence_order", pa.int64()),
        ("window_index", pa.int64()),
        ("timestep_index", pa.int64()),
        ("elapsed_candidate_s", pa.float64()),
        ("original_time_s", pa.float64()),
        ("source_row_right", pa.int64()),
        ("ns_raw", pa.int64()),
        ("mission_phase", pa.string()),
        ("voltage_v", pa.float64()),
        ("current_a", pa.float64()),
        ("Temperature__C", pa.float64()),
        ("capacity_ah", pa.float64()),
        ("capacity_source", pa.string()),
        ("source_rpt_id", pa.string()),
        ("soc_reference", pa.float64()),
        ("soc_prediction", pa.float64()),
        ("signed_error", pa.float64()),
        ("absolute_error", pa.float64()),
    ]
)

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
    "validation_interval",
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
    *,
    return_predictions: bool = False,
) -> dict[str, float | int] | tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    """Unchanged primary evaluation (same predict() call, same per-mission
    truncation, same metrics) for every existing caller.

    ``return_predictions`` defaults to False, preserving the exact original
    dict-only return. When explicitly set True, this ALSO returns the raw
    flat ``truth``/``estimate`` arrays this call already computed internally
    -- never a second call to ``predict()``/the model -- so a caller
    building test_predictions.parquet reuses the SAME truth/prediction
    values the metrics dict above was computed from, rather than
    regenerating them."""
    truth, estimate = predict(model, cycles, device, tensor_layout)
    error = estimate - truth
    mission_rmse, mission_mae, start = [], [], 0
    for cycle in cycles:
        length = len(cycle.soc) // 90 * 90
        mission_error = error[start : start + length]
        mission_rmse.append(float(np.sqrt(np.mean(mission_error**2))))
        mission_mae.append(float(np.mean(np.abs(mission_error))))
        start += length
    metrics = {
        "overall_rmse_percent": float(np.sqrt(np.mean(error**2)) * 100),
        "overall_mae_percent": float(np.mean(np.abs(error)) * 100),
        "mission_macro_rmse_percent": float(np.mean(mission_rmse) * 100),
        "mission_macro_mae_percent": float(np.mean(mission_mae) * 100),
        "test_missions": len(cycles),
        "test_points": len(truth),
    }
    if not return_predictions:
        return metrics
    return metrics, truth, estimate


def _mission_metadata(records, mission_ids: set[str]) -> dict[str, dict[str, np.ndarray]]:
    """Diagnostic-only companion to ``_to_shen_cycles``: for each mission in
    ``mission_ids``, reads mission_sequence_order/elapsed_candidate_s/
    original_time_s/source_row_right straight from ``record.data`` -- the
    exact same frame ``_to_shen_cycles`` reads FEATURES/soc from, in the
    same per-record iteration order -- so these full-length (untruncated)
    per-row arrays can later be truncated/reshaped by ``_windows_metadata``
    with the IDENTICAL ``usable = len(...) // WINDOW * WINDOW`` logic
    ``windows()`` itself uses inside ``predict()``, keeping every value
    aligned to the exact SOC timestep it describes."""
    metadata = {}
    for record in records:
        mission = str(record.metadata["mission_cycle_id"])
        if mission not in mission_ids:
            continue
        frame = record.data
        metadata[mission] = {
            "mission_sequence_order": np.full(
                len(frame), int(record.metadata["sequence_order"]), dtype=np.int64
            ),
            "elapsed_candidate_s": frame["elapsed_candidate_s"].to_numpy(dtype=np.float64),
            "original_time_s": frame["original_time_s"].to_numpy(dtype=np.float64),
            "source_row_right": frame["source_row_right"].to_numpy(dtype=np.int64),
        }
    return metadata


def _windows_metadata(values: np.ndarray, length: int = WINDOW) -> np.ndarray:
    """Truncates/reshapes a full-length per-row metadata array with the SAME
    ``usable = len(values) // length * length`` logic ``windows()``
    (shen_2023/data.py) applies to features/soc for the same cycle."""
    usable = len(values) // length * length
    return values[:usable].reshape(-1, length)


def _flatten_prediction_metadata(
    cycles: list[ShenCycle],
    metadata_by_mission: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Diagnostic-only, model-free companion to ``_evaluate``: for the SAME
    ``cycles`` list, in the SAME order, that ``_evaluate``'s own
    ``predict()`` call iterates, replays ``windows()``'s identical
    ``usable = len(cycle.soc) // WINDOW * WINDOW`` truncation and
    window-major/timestep-minor reshape on each mission's raw metadata only
    -- never touches the model, never recomputes a prediction. The caller
    must pair this function's flat output 1:1, in order, against
    ``_evaluate(..., return_predictions=True)``'s own truth/estimate arrays
    for these same ``cycles`` -- both come from the exact same per-cycle,
    same-order, same-truncation iteration, so they always align."""
    mission_ids: list[str] = []
    window_indices: list[np.ndarray] = []
    timestep_indices: list[np.ndarray] = []
    metadata_out: dict[str, list[np.ndarray]] = {
        "mission_sequence_order": [],
        "elapsed_candidate_s": [],
        "original_time_s": [],
        "source_row_right": [],
    }
    for cycle in cycles:
        usable = len(cycle.soc) // WINDOW * WINDOW
        n_windows = usable // WINDOW
        if n_windows == 0:
            continue
        mission_metadata = metadata_by_mission[cycle.profile]
        for name in metadata_out:
            metadata_out[name].append(_windows_metadata(mission_metadata[name]).ravel())
        mission_ids.extend([cycle.profile] * usable)
        window_indices.append(np.repeat(np.arange(n_windows), WINDOW))
        timestep_indices.append(np.tile(np.arange(WINDOW), n_windows))
    if not mission_ids:
        empty_float = np.empty((0,), dtype=np.float64)
        empty_int = np.empty((0,), dtype=np.int64)
        return (
            np.empty((0,), dtype=object),
            empty_int,
            empty_int,
            {
                "mission_sequence_order": empty_int,
                "elapsed_candidate_s": empty_float,
                "original_time_s": empty_float,
                "source_row_right": empty_int,
            },
        )
    metadata_flat = {name: np.concatenate(values) for name, values in metadata_out.items()}
    metadata_flat["mission_sequence_order"] = metadata_flat["mission_sequence_order"].astype(np.int64)
    metadata_flat["source_row_right"] = metadata_flat["source_row_right"].astype(np.int64)
    return (
        np.array(mission_ids, dtype=object),
        np.concatenate(window_indices).astype(np.int64),
        np.concatenate(timestep_indices).astype(np.int64),
        metadata_flat,
    )


def _build_prediction_rows(
    *,
    framework: str,
    method: str,
    source_kind: str,
    scenario: str,
    seed: int,
    cell_id: str,
    condition: str,
    soc_reference: np.ndarray,
    soc_prediction: np.ndarray,
    mission_ids: np.ndarray,
    window_indices: np.ndarray,
    timestep_indices: np.ndarray,
    metadata: dict[str, np.ndarray],
    raw_diagnostic: pd.DataFrame,
) -> list[dict]:
    """Assembles one test_predictions.parquet row per raw timestep, joining
    Ns/mission_phase/raw V-I-T/capacity provenance by this timestep's own
    source_row_right via cmu_diagnostics' canonical join -- never a local
    phase map. ``soc_reference``/``soc_prediction`` must be the SAME arrays
    ``_evaluate(..., return_predictions=True)`` returned (never clipped,
    never recomputed), so signed_error/absolute_error here always match
    what a reader recomputing MAE/RMSE from this parquet gets against the
    primary metrics dict."""
    if not len(soc_reference):
        return []
    joined = join_diagnostic_fields_by_source_row(metadata["source_row_right"], raw_diagnostic)
    rows = []
    n = len(soc_reference)
    for i in range(n):
        signed_error = float(soc_prediction[i] - soc_reference[i])
        row = {
            "framework": framework,
            "method": method,
            "source_kind": source_kind,
            "scenario": scenario,
            "seed": seed,
            "cell_id": cell_id,
            "condition": condition,
            "split": "test",
            "mission_cycle_id": str(mission_ids[i]),
            "mission_sequence_order": int(metadata["mission_sequence_order"][i]),
            "window_index": int(window_indices[i]),
            "timestep_index": int(timestep_indices[i]),
            "elapsed_candidate_s": float(metadata["elapsed_candidate_s"][i]),
            "original_time_s": float(metadata["original_time_s"][i]),
            "source_row_right": int(metadata["source_row_right"][i]),
            "soc_reference": float(soc_reference[i]),
            "soc_prediction": float(soc_prediction[i]),
            "signed_error": signed_error,
            "absolute_error": float(abs(signed_error)),
        }
        for raw_column, output_column in (
            ("ns_raw", "ns_raw"),
            ("mission_phase", "mission_phase"),
            ("voltage_v_raw", "voltage_v"),
            ("current_a_raw", "current_a"),
            ("temperature_c_raw", "Temperature__C"),
            ("capacity_ah", "capacity_ah"),
            ("capacity_source", "capacity_source"),
            ("source_rpt_id", "source_rpt_id"),
        ):
            value = joined[raw_column][i]
            row[output_column] = (
                None if value is None else DIAGNOSTIC_METADATA_CASTERS[raw_column](value)
            )
        rows.append(row)
    return rows


def _write_test_predictions_atomic(
    rows_by_method: list[list[dict]], schema: pa.Schema, final_path: Path
) -> None:
    """Streams each method's chunk of prediction rows to a ``.incomplete``
    temp parquet file via one ``pq.ParquetWriter``, then atomically
    ``Path.replace``s it onto ``final_path`` ONLY once every chunk has been
    written without error. The writer is always closed in ``finally`` (so a
    failed ``.incomplete`` file still has a valid parquet footer, readable
    for inspection) but is renamed to the final name ONLY on success -- a
    failed or interrupted write is never silently promoted to, or mistaken
    for, a complete ``test_predictions.parquet``."""
    tmp_path = final_path.with_name(final_path.name + ".incomplete")
    writer = pq.ParquetWriter(tmp_path, schema)
    completed = False
    try:
        for rows in rows_by_method:
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        completed = True
    finally:
        writer.close()
        if completed:
            tmp_path.replace(final_path)


def _compute_phase_metrics(predictions_path: Path) -> pd.DataFrame:
    """Read-only recomputation from test_predictions.parquet alone --
    restricted to the three canonical flight phases (take_off/cruise/
    landing); grouping preserves source_kind and scenario so a released-
    checkpoint run and a trained-checkpoint run (or unsupervised vs
    semisupervised) are never averaged together."""
    frame = pd.read_parquet(
        predictions_path,
        columns=[
            "source_kind", "scenario", "cell_id", "condition", "method",
            "mission_phase", "mission_cycle_id", "soc_reference", "soc_prediction",
        ],
    )
    frame = frame.loc[frame["mission_phase"].isin(PHASE_SUMMARY_PHASES)].copy()
    frame["signed_error"] = frame["soc_prediction"] - frame["soc_reference"]
    frame["absolute_error"] = frame["signed_error"].abs()
    grouped = frame.groupby(
        ["source_kind", "scenario", "cell_id", "condition", "method", "mission_phase"],
        sort=False,
        as_index=False,
    )
    return grouped.agg(
        n_points=("soc_reference", "size"),
        n_missions=("mission_cycle_id", "nunique"),
        mae_percent=("absolute_error", lambda s: float(s.mean()) * 100),
        rmse_percent=("signed_error", lambda s: float(np.sqrt((s**2).mean())) * 100),
        mean_signed_error_percent=("signed_error", lambda s: float(s.mean()) * 100),
    )


_TRANSFER_EFFECT_COLUMNS = (
    "source_kind", "scenario", "cell_id", "condition", "method", "reference_method",
    "source_only_mae_percent", "tatn_mae_percent", "delta_mae_percent",
    "source_only_rmse_percent", "tatn_rmse_percent", "delta_rmse_percent",
    "negative_transfer_mae", "negative_transfer_rmse",
)


def _pair_source_only_vs_tatn(
    frame: pd.DataFrame, join_keys: list[str], mae_column: str, rmse_column: str
) -> pd.DataFrame:
    """Shared pairing logic for both phase- and cell-level transfer-effect
    tables: pairs each (join_keys) group's "Source only" row against its
    "TATN" row, never against another group's row (an inner merge on
    join_keys, which always includes source_kind/scenario/cell_id/
    condition). delta = TATN - Source only; negative_transfer_* flags a
    positive delta (TATN worse than the frozen source baseline)."""
    columns = ["mission_phase", *_TRANSFER_EFFECT_COLUMNS] if "mission_phase" in join_keys else list(
        _TRANSFER_EFFECT_COLUMNS
    )
    if frame.empty:
        return pd.DataFrame(columns=columns)
    source_only = frame.loc[frame["method"] == "Source only"]
    tatn = frame.loc[frame["method"] == "TATN"]
    if source_only.empty or tatn.empty:
        return pd.DataFrame(columns=columns)
    merged = source_only.merge(
        tatn, on=join_keys, suffixes=("_source_only", "_tatn"), how="inner"
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged["method"] = "TATN"
    merged["reference_method"] = "Source only"
    merged["source_only_mae_percent"] = merged[f"{mae_column}_source_only"]
    merged["tatn_mae_percent"] = merged[f"{mae_column}_tatn"]
    merged["delta_mae_percent"] = merged["tatn_mae_percent"] - merged["source_only_mae_percent"]
    merged["source_only_rmse_percent"] = merged[f"{rmse_column}_source_only"]
    merged["tatn_rmse_percent"] = merged[f"{rmse_column}_tatn"]
    merged["delta_rmse_percent"] = merged["tatn_rmse_percent"] - merged["source_only_rmse_percent"]
    merged["negative_transfer_mae"] = merged["delta_mae_percent"] > 0
    merged["negative_transfer_rmse"] = merged["delta_rmse_percent"] > 0
    return merged[columns]


def _compute_phase_transfer_effects(phase_metrics: pd.DataFrame) -> pd.DataFrame:
    """delta = TATN - Source only per (source_kind, scenario, cell_id,
    condition, mission_phase); negative_transfer_* = delta > 0."""
    return _pair_source_only_vs_tatn(
        phase_metrics,
        ["source_kind", "scenario", "cell_id", "condition", "mission_phase"],
        "mae_percent",
        "rmse_percent",
    )


def _compute_cell_transfer_effects(condition_metrics_path: Path) -> pd.DataFrame:
    """Overall (non-phase-restricted) counterpart to
    ``_compute_phase_transfer_effects``, read back from condition_metrics.csv
    alone: delta = TATN - Source only per (source_kind, scenario, cell_id,
    condition), using each row's own overall_mae_percent/
    overall_rmse_percent."""
    frame = pd.read_csv(condition_metrics_path)
    return _pair_source_only_vs_tatn(
        frame,
        ["source_kind", "scenario", "cell_id", "condition"],
        "overall_mae_percent",
        "overall_rmse_percent",
    )


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
    deleted. The result's ``target.pt`` checkpoint, ``training_history.csv``,
    AND ``test_predictions.parquet`` sidecars must also exist -- a prior run
    could have been interrupted after writing ``results.json`` but before
    one of its sidecars finished (``test_predictions.parquet`` is only ever
    the FINAL name of an atomically-renamed write; a leftover
    ``test_predictions.parquet.incomplete`` from an interrupted write is
    never treated as this file). Raises ValueError immediately in every
    failing case instead."""
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
    training_history_path = cell_output / "training_history.csv"
    if not training_history_path.is_file():
        raise ValueError(
            f"{result_path}: --resume would skip this cell, but its "
            f"training_history.csv sidecar {training_history_path} is "
            "missing. Rerun without --resume, or use a fresh --output-root."
        )
    test_predictions_path = cell_output / "test_predictions.parquet"
    if not test_predictions_path.is_file():
        raise ValueError(
            f"{result_path}: --resume would skip this cell, but its "
            f"test_predictions.parquet sidecar {test_predictions_path} is "
            "missing. Rerun without --resume, or use a fresh --output-root."
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
    parser.add_argument(
        "--validation-interval", type=int, default=10,
        help="Run the read-only validation diagnostic every Nth adaptation "
        "epoch, plus epoch 1 and the actual last epoch (default: 10). "
        "Purely diagnostic -- never affects training, pseudo-labeling, "
        "early stopping, the scheduler, or checkpoint selection.",
    )
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
            # Loaded from an existing checkpoint -- no training happened
            # this run, so no source_training_history.csv is written (never
            # fabricate a training history for a checkpoint we didn't train).
            source_model = load_source(source_checkpoint, device)
        else:
            source_model, _, source_epoch_diagnostics = train_source(
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
                record_epoch_seconds=True,
            )
            pd.DataFrame(source_epoch_diagnostics).to_csv(
                args.output_root / "source_training_history.csv", index=False
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
        "validation_interval": args.validation_interval,
    }
    rows = []
    training_history_frames: list[pd.DataFrame] = []
    prediction_tables: list[pa.Table] = []

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
            training_history_frames.append(
                pd.read_csv(cell_output / "training_history.csv")
            )
            prediction_tables.append(
                pq.read_table(cell_output / "test_predictions.parquet")
            )
            continue
        cell_output.mkdir(parents=True, exist_ok=True)

        adaptation_ids, validation_ids, test_ids, budget = select_duration_matched_missions(
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

        # Diagnostic-only, read-only validation probe input: the manifest's
        # EXISTING "validation"-split missions for this cell (never used
        # for training/pseudo-labeling/early-stopping/checkpoint selection
        # -- see adapt()'s validation_cycles= docstring). An empty
        # validation_ids set for this cell is not an error; the probe is
        # simply skipped (validation_cycles left as None) and
        # validation_evaluated stays honestly False for every epoch.
        validation_cycles = (
            _to_shen_cycles(
                _records_for_ids(labelled_records, validation_ids),
                feature_min,
                feature_max,
                require_labels=True,
            )
            if validation_ids
            else None
        )

        source_only, source_only_truth, source_only_estimate = _evaluate(
            source_model, test_cycles, device, tensor_layout, return_predictions=True
        )
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
            record_epoch_seconds=True,
            validation_cycles=validation_cycles,
            validation_interval=args.validation_interval,
        )
        tatn, tatn_truth, tatn_estimate = _evaluate(
            transferred, test_cycles, device, tensor_layout, return_predictions=True
        )
        cell_training_history = pd.DataFrame(history)
        if not cell_training_history.empty:
            cell_training_history.insert(0, "cell_id", cell_id)
        cell_training_history.to_csv(cell_output / "training_history.csv", index=False)
        training_history_frames.append(cell_training_history)

        # Diagnostic-only: predictions/metadata for test_predictions.parquet,
        # built from the SAME truth/estimate arrays _evaluate just returned
        # for "Source only"/"TATN" above -- never a second predict() call,
        # never a different clip/rounding, so signed_error/absolute_error
        # here always match what a reader recomputes against the primary
        # metrics dict. Metadata (mission id/window/timestep/sequence order/
        # elapsed & original time/source_row_right) replays predict()'s own
        # usable = len(cycle.soc)//WINDOW*WINDOW truncation and flatten
        # order without touching the model.
        prediction_metadata = _mission_metadata(labelled_records, test_ids)
        raw_diagnostic = load_raw_diagnostic_fields(source_path)
        mission_ids_arr, window_idx, timestep_idx, pred_metadata = _flatten_prediction_metadata(
            test_cycles, prediction_metadata
        )
        source_only_rows = _build_prediction_rows(
            framework=FRAMEWORK,
            method="Source only",
            source_kind=args.source_kind,
            scenario=args.scenario,
            seed=args.seed,
            cell_id=cell_id,
            condition=budget["condition"],
            soc_reference=source_only_truth,
            soc_prediction=source_only_estimate,
            mission_ids=mission_ids_arr,
            window_indices=window_idx,
            timestep_indices=timestep_idx,
            metadata=pred_metadata,
            raw_diagnostic=raw_diagnostic,
        )
        tatn_rows = _build_prediction_rows(
            framework=FRAMEWORK,
            method="TATN",
            source_kind=args.source_kind,
            scenario=args.scenario,
            seed=args.seed,
            cell_id=cell_id,
            condition=budget["condition"],
            soc_reference=tatn_truth,
            soc_prediction=tatn_estimate,
            mission_ids=mission_ids_arr,
            window_indices=window_idx,
            timestep_indices=timestep_idx,
            metadata=pred_metadata,
            raw_diagnostic=raw_diagnostic,
        )
        test_predictions_path = cell_output / "test_predictions.parquet"
        _write_test_predictions_atomic(
            [source_only_rows, tatn_rows], TEST_PREDICTIONS_SCHEMA, test_predictions_path
        )
        prediction_tables.append(pq.read_table(test_predictions_path))
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
                "scenario": args.scenario,
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
    condition_metrics_path = args.output_root / "condition_metrics.csv"
    table.to_csv(condition_metrics_path, index=False)
    if training_history_frames:
        pd.concat(training_history_frames, ignore_index=True).to_csv(
            args.output_root / "training_history.csv", index=False
        )

    # Diagnostic-only: rebuild the top-level test_predictions.parquet by
    # concatenating every cell's own (already atomically-written) sidecar --
    # including any --resume'd cell's, read back from disk -- never from a
    # second evaluation pass. phase_metrics.csv/phase_transfer_effects.csv
    # are then read back from THIS just-rebuilt file (never from in-memory
    # rows), and cell_transfer_effects.csv from condition_metrics.csv --
    # skipped (never writing empty/placeholder files) if there is nothing to
    # summarize yet.
    test_predictions_path = args.output_root / "test_predictions.parquet"
    if prediction_tables:
        pq.write_table(pa.concat_tables(prediction_tables), test_predictions_path)
    if test_predictions_path.is_file():
        phase_metrics = _compute_phase_metrics(test_predictions_path)
        if not phase_metrics.empty:
            phase_metrics.to_csv(args.output_root / "phase_metrics.csv", index=False)
            phase_transfer_effects = _compute_phase_transfer_effects(phase_metrics)
            phase_transfer_effects.to_csv(
                args.output_root / "phase_transfer_effects.csv", index=False
            )
    if condition_metrics_path.is_file():
        cell_transfer_effects = _compute_cell_transfer_effects(condition_metrics_path)
        if not cell_transfer_effects.empty:
            cell_transfer_effects.to_csv(
                args.output_root / "cell_transfer_effects.csv", index=False
            )

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
                    "validation_mae": "percent",
                    "validation_rmse": "percent",
                    "runtime_seconds": "seconds",
                    "validation_runtime_seconds": "seconds",
                    "soc_reference": "fraction",
                    "soc_prediction": "fraction",
                    "signed_error": "fraction",
                    "absolute_error": "fraction",
                },
                "prediction_and_phase_protocol": (
                    "test_predictions.parquet (per cell, then aggregated at "
                    "the top level) has one row per raw CMU test-split "
                    "timestep for both the 'Source only' (frozen source "
                    "model) and 'TATN' (post-adapt()) methods, built from "
                    "the SAME truth/estimate arrays _evaluate(..., "
                    "return_predictions=True) already computed for the "
                    "metrics above -- never a second predict() call, never "
                    "reclipped or otherwise altered. Metadata (window/"
                    "timestep index, mission sequence order, elapsed/"
                    "original time, source_row_right) replays predict()'s "
                    "own usable = len(cycle.soc)//90*90 truncation and "
                    "window-major/timestep-minor flatten order without "
                    "touching the model. voltage_v/current_a/Temperature__C/"
                    "capacity_ah/capacity_source/source_rpt_id/ns_raw/"
                    "mission_phase are joined by source_row_right via "
                    "battery_benchmark.benchmarks.cmu_diagnostics' canonical "
                    "Ns mapping and raw-parquet join -- no locally-defined "
                    "phase map. Each cell's parquet is written to a "
                    "'.incomplete' temp file via a streaming ParquetWriter "
                    "and atomically renamed onto test_predictions.parquet "
                    "only once the write completes without error; a failed "
                    "or interrupted write leaves the '.incomplete' file and "
                    "never produces a falsely-complete test_predictions."
                    "parquet, and --resume refuses to skip a cell whose "
                    "test_predictions.parquet is missing. phase_metrics.csv/"
                    "phase_transfer_effects.csv are recomputed read-only "
                    "from the aggregated test_predictions.parquet, "
                    "restricted to the take_off/cruise/landing phases, "
                    "grouped by (source_kind, scenario, cell_id, condition, "
                    "method, mission_phase); cell_transfer_effects.csv is "
                    "the same delta/negative-transfer pairing over overall "
                    "(non-phase-restricted) metrics from condition_metrics."
                    "csv. Both effects tables define delta_*_percent = "
                    "TATN - 'Source only', and negative_transfer_* = "
                    "delta_*_percent > 0. None of this changes predictions, "
                    "metrics, training, pseudo-labeling, or model/optimizer/"
                    "early-stopping/checkpoint-selection logic."
                ),
                "validation_protocol": (
                    "Diagnostic-only: when the CMU manifest's 'validation' "
                    "split has missions for a cell, adapt() probes "
                    "validation_mae/validation_rmse (percent) on epoch 1, "
                    "every --validation-interval-th epoch, and the actual "
                    "last epoch, via a read-only, no_grad forward pass "
                    "through a LOCAL DataLoader generator, with every "
                    "submodule's train/eval mode snapshotted and restored "
                    "around the probe. Never used for early stopping, the "
                    "scheduler, checkpoint selection, or pseudo-labeling -- "
                    "unchanged by this diagnostic. training_history.csv's "
                    "runtime_seconds covers ONLY the training batch loop; "
                    "the validation probe's own wall-clock is recorded "
                    "separately as validation_runtime_seconds (None when "
                    "validation wasn't run this epoch) -- the two are never "
                    "summed together."
                ),
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
