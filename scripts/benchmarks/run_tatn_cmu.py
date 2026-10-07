#!/usr/bin/env python3
"""Run TATN from Panasonic NCR18650PF 25 C onto the CMU eVTOL core seven.

Unified-benchmark-protocol runner (output schema ``tatn_cmu_v2_diagnostic``):

* deterministic ``--repeats`` (independent source state per repeat; every
  target cell of a repeat starts from that repeat's frozen source state),
* hash-level protocol verification against the manifest and, optionally,
  finished Backbone/DARM/SFTNN/DTNN reference runs,
* per-epoch source/adaptation monitoring with training-only wall-clock,
* per-timestep test predictions, overall/phase metrics, negative-transfer
  tables, a full-mission diagnostic, computational-cost metadata, and
* strict ``--resume`` that rejects legacy or mismatching outputs.

Primary setting: unsupervised TATN (no CMU SOC label is used for training,
adaptation, pseudo-labels or checkpoint selection; the SOC reference is used
only for post-hoc evaluation and the diagnostic-only validation probe).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from copy import deepcopy
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
from battery_benchmark.benchmarks.cmu_protocol_check import (  # noqa: E402
    ProtocolMismatchError,
    verify_cmu_protocol,
)
from battery_benchmark.datasets.cmu_evtol.ns_phase_labeling_v1 import (  # noqa: E402
    NS_MISSION_PHASE_MAP,
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
    PSEUDO_LABEL_THRESHOLD,
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

# Version of every file/column contract written by this script. Anything on
# disk with a different (or missing) value is a legacy output and is never
# resumed from, overwritten or deleted.
OUTPUT_SCHEMA_VERSION = "tatn_cmu_v2_diagnostic"
PHASE_MAPPING_VERSION = "cmu_evtol.ns_phase_labeling_v1.NS_MISSION_PHASE_MAP"
PHASE_MAPPING_SHA256 = hashlib.sha256(
    json.dumps(
        {str(k): v for k, v in sorted(NS_MISSION_PHASE_MAP.items())}, sort_keys=True
    ).encode()
).hexdigest()

METHOD_SOURCE_ONLY = "Source only"
METHOD_TATN = "TATN"

# TATN hyperparameters (paper.yaml / verified repo implementation), named
# only so metadata and resume checks can record the exact configuration.
TATN_LEARNING_RATE = 1e-3  # used for both train_source() and adapt()
TATN_LAMBDAS = (1.0, 1.0, 1.0)
SOURCE_EARLY_STOPPING_PATIENCE = 30
SOURCE_MIN_EPOCHS = 60
TRANSFER_EARLY_STOPPING_PATIENCE = 15
TRANSFER_MIN_EPOCHS = 30
PSEUDO_LABEL_LOOKAHEAD = 200  # shen_2023.data.selected_windows default
SEED_STRIDE_PER_REPEAT = 1000

# _evaluate() scores every raw timestep inside each non-overlapping 90-step
# window (via predict()'s flattened truth/estimate arrays), not a
# last-step-per-window summary -- same convention as DTNN/backbone/SFTNN.
EVALUATION_UNIT = "all_timesteps"

WINDOW = 90
FRAMEWORK = "shen_tatn"
PHASE_SUMMARY_PHASES = ("take_off", "cruise", "landing")

# test_predictions.parquet: one row per raw CMU test timestep, for both
# methods, every repeat and cell. SOC columns are FRACTIONS (0-1); metrics
# files report percent (explicit *_percent column names).
TEST_PREDICTIONS_SCHEMA = pa.schema(
    [
        ("framework", pa.string()),
        ("method", pa.string()),
        ("source_kind", pa.string()),
        ("scenario", pa.string()),
        ("repeat", pa.int64()),
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
        ("voltage_v_raw", pa.float64()),
        ("current_a_raw", pa.float64()),
        ("temperature_c_raw", pa.float64()),
        ("capacity_ah", pa.float64()),
        ("capacity_source", pa.string()),
        ("source_rpt_id", pa.string()),
        ("soc_reference", pa.float64()),
        ("soc_prediction", pa.float64()),
        ("signed_error", pa.float64()),
        ("absolute_error", pa.float64()),
    ],
    metadata={
        b"output_schema_version": OUTPUT_SCHEMA_VERSION.encode(),
        b"soc_unit": b"fraction",
        b"error_unit": b"fraction (soc_prediction - soc_reference)",
        b"evaluation_unit": EVALUATION_UNIT.encode(),
        b"voltage_unit": b"V",
        b"current_unit": b"A",
        b"temperature_unit": b"degC",
    },
)

LABEL_FLAGS_UNSUPERVISED = {
    "target_soc_labels_used_for_training": False,
    "target_soc_labels_used_for_adaptation": False,
    "target_soc_labels_used_for_pseudo_labels": False,
    "target_soc_labels_used_for_checkpoint_selection": False,
    "target_soc_reference_used_for_posthoc_evaluation": True,
}

# Fields every cell's results.json records so a later --resume can verify the
# on-disk result was produced from the SAME data version and configuration.
RESUME_CHECK_FIELDS = (
    "output_schema_version",
    "cmu_input_sha256",
    "manifest_sha256",
    "source_checkpoint_sha256",
    "source_kind",
    "scenario",
    "repeat",
    "seed",
    "base_seed",
    "source_epochs",
    "transfer_epochs",
    "batch_size",
    "tensor_layout",
    "tatn_learning_rate",
    "tatn_lambdas",
    "source_early_stopping_patience",
    "source_min_epochs",
    "transfer_early_stopping_patience",
    "transfer_min_epochs",
    "validation_interval",
)

# Required per-epoch columns of training_history.csv /
# source_training_history.csv (``stage`` is "source" or "adaptation").
HISTORY_COLUMNS = (
    "repeat",
    "seed",
    "cell_id",
    "stage",
    "epoch",
    "learning_rate",
    "runtime_seconds",
    "optimizer_steps",
    "total_loss",
    "source_soc_regression_loss",
    "adversarial_loss",
    "discriminator_loss",
    "mmd_loss",
    "pseudo_label_loss",
    "pseudo_accepted_windows",
    "pseudo_candidate_windows",
    "pseudo_accepted_fraction",
    "pseudo_threshold",
    "validation_overall_mae",
    "validation_overall_rmse",
    "validation_mission_macro_mae",
    "validation_mission_macro_rmse",
    "validation_evaluated",
    "validation_runtime_seconds",
)

SOURCE_CHECKPOINT_PREFIX = "source_25C_repeat"


def repeat_seed(base_seed: int, repeat: int) -> int:
    """Seed of repeat ``repeat`` (1-based); repeat 1 equals the legacy seed."""
    return int(base_seed) + SEED_STRIDE_PER_REPEAT * (int(repeat) - 1)


def cell_seed(base_seed: int, repeat: int, cell_id: str) -> int:
    """Adaptation seed for one (repeat, cell): independent of ``--cells``."""
    return repeat_seed(base_seed, repeat) + CELLS.index(cell_id)


def _repeat_dir(output_root: Path, repeat: int) -> Path:
    return output_root / f"repeat_{repeat:02d}"


def _source_checkpoint_path(output_root: Path, repeat: int) -> Path:
    return output_root / f"{SOURCE_CHECKPOINT_PREFIX}{repeat:02d}.pt"


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



def _state_fingerprint(model: torch.nn.Module) -> str:
    """SHA-256 over every parameter/buffer (name + raw bytes) -- used to prove
    a frozen source state is byte-identical before and after each target cell."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _bias_metrics(
    truth: np.ndarray, estimate: np.ndarray, cycles: list[ShenCycle]
) -> dict[str, float]:
    """Mean signed error (prediction - reference) in percent, pooled and
    mission-macro, over the same per-mission truncation ``_evaluate`` uses."""
    error = estimate - truth
    per_mission, start = [], 0
    for cycle in cycles:
        length = len(cycle.soc) // WINDOW * WINDOW
        per_mission.append(float(np.mean(error[start : start + length])))
        start += length
    return {
        "overall_bias_percent": float(np.mean(error) * 100),
        "mission_macro_bias_percent": float(np.mean(per_mission) * 100),
    }


def _assert_no_target_label_leak(
    adaptation_cycles: list[ShenCycle],
    *,
    scenario: str,
    labeled_profile: str | None,
    adaptation_ids: set[str],
    validation_ids: set[str],
    test_ids: set[str],
) -> None:
    """Defensive guard: unsupervised adaptation cycles must carry NO target
    SOC (only the zero placeholder ``_to_shen_cycles(require_labels=False)``
    writes), and the adaptation/validation/test mission sets are disjoint."""
    if adaptation_ids & test_ids or adaptation_ids & validation_ids or validation_ids & test_ids:
        raise ValueError("mission split leakage between adaptation/validation/test")
    if scenario != "unsupervised":
        return
    if labeled_profile is not None:
        raise ValueError("unsupervised scenario must not name a labelled target mission")
    for cycle in adaptation_cycles:
        if np.any(cycle.soc != 0):
            raise ValueError(
                f"{cycle.profile}: unsupervised adaptation cycle carries non-placeholder SOC"
            )
        if cycle.profile in test_ids or cycle.profile in validation_ids:
            raise ValueError(f"{cycle.profile}: adaptation mission also in test/validation")


def _object_column(values: np.ndarray, caster, arrow_type: pa.DataType) -> pa.Array:
    flat = [None if v is None else caster(v) for v in values.tolist()]
    return pa.array(flat, type=arrow_type)


def _build_prediction_table(
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
    repeat: int = 1,
) -> pa.Table:
    """Vectorized builder of the per-timestep prediction table. Joins
    Ns/mission_phase/raw V-I-T/capacity provenance by each timestep's own
    ``source_row_right`` via cmu_diagnostics' canonical join -- never a local
    phase map. ``soc_reference``/``soc_prediction`` must be the SAME arrays
    ``_evaluate(..., return_predictions=True)`` returned (never clipped), so
    signed_error/absolute_error recompute to the primary metrics."""
    n = len(soc_reference)
    reference = np.asarray(soc_reference, dtype=np.float64)
    prediction = np.asarray(soc_prediction, dtype=np.float64)
    signed = prediction - reference
    if n:
        joined = join_diagnostic_fields_by_source_row(metadata["source_row_right"], raw_diagnostic)
    else:
        joined = {c: np.empty((0,), dtype=object) for c in DIAGNOSTIC_METADATA_CASTERS}

    def const(value, arrow_type):
        return pa.array([value] * n, type=arrow_type)

    columns = {
        "framework": const(framework, pa.string()),
        "method": const(method, pa.string()),
        "source_kind": const(source_kind, pa.string()),
        "scenario": const(scenario, pa.string()),
        "repeat": const(int(repeat), pa.int64()),
        "seed": const(int(seed), pa.int64()),
        "cell_id": const(cell_id, pa.string()),
        "condition": const(condition, pa.string()),
        "split": const("test", pa.string()),
        "mission_cycle_id": pa.array([str(m) for m in mission_ids], type=pa.string()),
        "mission_sequence_order": pa.array(metadata["mission_sequence_order"], type=pa.int64()),
        "window_index": pa.array(window_indices, type=pa.int64()),
        "timestep_index": pa.array(timestep_indices, type=pa.int64()),
        "elapsed_candidate_s": pa.array(metadata["elapsed_candidate_s"], type=pa.float64()),
        "original_time_s": pa.array(metadata["original_time_s"], type=pa.float64()),
        "source_row_right": pa.array(metadata["source_row_right"], type=pa.int64()),
        "soc_reference": pa.array(reference, type=pa.float64()),
        "soc_prediction": pa.array(prediction, type=pa.float64()),
        "signed_error": pa.array(signed, type=pa.float64()),
        "absolute_error": pa.array(np.abs(signed), type=pa.float64()),
    }
    for name in (
        "ns_raw", "mission_phase", "voltage_v_raw", "current_a_raw",
        "temperature_c_raw", "capacity_ah", "capacity_source", "source_rpt_id",
    ):
        columns[name] = _object_column(
            joined[name], DIAGNOSTIC_METADATA_CASTERS[name], TEST_PREDICTIONS_SCHEMA.field(name).type
        )
    return pa.table(columns, schema=TEST_PREDICTIONS_SCHEMA)


def _build_prediction_rows(**kwargs) -> list[dict]:
    """List-of-dict view of ``_build_prediction_table`` (same arguments)."""
    return _build_prediction_table(**kwargs).to_pylist()


def _write_test_predictions_atomic(
    chunks: list, schema: pa.Schema, final_path: Path
) -> None:
    """Streams each chunk (a list of row dicts or a ``pa.Table``) to a
    ``.incomplete`` temp parquet via one ``pq.ParquetWriter``, then atomically
    renames it onto ``final_path`` ONLY once every chunk was written without
    error. A failed or interrupted write is never promoted to, or mistaken
    for, a complete ``test_predictions.parquet``."""
    tmp_path = final_path.with_name(final_path.name + ".incomplete")
    writer = pq.ParquetWriter(tmp_path, schema)
    completed = False
    try:
        for chunk in chunks:
            if isinstance(chunk, pa.Table):
                if chunk.num_rows:
                    writer.write_table(chunk.cast(schema))
            elif chunk:
                writer.write_table(pa.Table.from_pylist(chunk, schema=schema))
        completed = True
    finally:
        writer.close()
        if completed:
            tmp_path.replace(final_path)


def _concat_parquets_atomic(paths: list[Path], schema: pa.Schema, final_path: Path) -> None:
    """Atomically concatenate completed per-cell parquet files (streamed)."""

    def chunks():
        for path in paths:
            yield pq.read_table(path).cast(schema)

    _write_test_predictions_atomic(chunks(), schema, final_path)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".incomplete")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _write_json_atomic(payload: object, path: Path) -> None:
    tmp = path.with_name(path.name + ".incomplete")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------
# Phase metrics
# --------------------------------------------------------------------------
_PHASE_GROUP = ["source_kind", "scenario", "cell_id", "condition", "method", "mission_phase"]


def _compute_phase_metrics(
    predictions_path: Path,
    *,
    phases: tuple[str, ...] | None = PHASE_SUMMARY_PHASES,
) -> pd.DataFrame:
    """Read-only recomputation from test_predictions.parquet alone. Default:
    the three canonical flight phases (take_off/cruise/landing). With
    ``phases=None`` every mapped phase present is kept and a missing/unmapped
    phase is labelled ``unmapped`` (full-mission diagnostic). Grouping keeps
    source_kind/scenario (and ``repeat`` when present) so runs are never
    averaged together."""
    available = set(pq.read_schema(predictions_path).names)
    has_repeat = "repeat" in available
    columns = [
        "source_kind", "scenario", "cell_id", "condition", "method", "mission_phase",
        "mission_cycle_id", "soc_reference", "soc_prediction",
    ] + (["repeat"] if has_repeat else [])
    frame = pd.read_parquet(predictions_path, columns=columns)
    if phases is None:
        frame["mission_phase"] = frame["mission_phase"].fillna("unmapped")
    else:
        frame = frame.loc[frame["mission_phase"].isin(phases)].copy()
    frame["signed_error"] = frame["soc_prediction"] - frame["soc_reference"]
    frame["absolute_error"] = frame["signed_error"].abs()
    frame["squared_error"] = frame["signed_error"] ** 2
    keys = (["repeat"] if has_repeat else []) + _PHASE_GROUP
    grouped = frame.groupby(keys, sort=False, as_index=False).agg(
        n_points=("soc_reference", "size"),
        n_missions=("mission_cycle_id", "nunique"),
        _mae=("absolute_error", "mean"),
        _mse=("squared_error", "mean"),
        _bias=("signed_error", "mean"),
    )
    grouped["mae_percent"] = grouped.pop("_mae") * 100
    grouped["rmse_percent"] = np.sqrt(grouped.pop("_mse")) * 100
    grouped["mean_signed_error_percent"] = grouped.pop("_bias") * 100
    return grouped


_TRANSFER_EFFECT_COLUMNS = (
    "source_kind", "scenario", "cell_id", "condition", "method", "reference_method",
    "source_only_mae_percent", "tatn_mae_percent", "delta_mae_percent",
    "source_only_rmse_percent", "tatn_rmse_percent", "delta_rmse_percent",
    "negative_transfer_mae", "negative_transfer_rmse",
)


def _pair_source_only_vs_tatn(
    frame: pd.DataFrame,
    join_keys: list[str],
    mae_column: str,
    rmse_column: str,
    bias_column: str | None = None,
) -> pd.DataFrame:
    """Pair each (join_keys) group's "Source only" row against its "TATN" row
    (inner merge -- never across groups). delta = TATN - Source only;
    negative_transfer_* flags a positive delta (TATN worse than frozen source)."""
    lead = [k for k in ("repeat",) if k in join_keys]
    phase = ["mission_phase"] if "mission_phase" in join_keys else []
    extra = (
        ["source_only_bias_percent", "tatn_bias_percent", "delta_abs_bias_percent"]
        if bias_column
        else []
    )
    columns = lead + phase + list(_TRANSFER_EFFECT_COLUMNS) + extra
    if frame.empty:
        return pd.DataFrame(columns=columns)
    source_only = frame.loc[frame["method"] == METHOD_SOURCE_ONLY]
    tatn = frame.loc[frame["method"] == METHOD_TATN]
    if source_only.empty or tatn.empty:
        return pd.DataFrame(columns=columns)
    merged = source_only.merge(tatn, on=join_keys, suffixes=("_source_only", "_tatn"), how="inner")
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged["method"] = METHOD_TATN
    merged["reference_method"] = METHOD_SOURCE_ONLY
    merged["source_only_mae_percent"] = merged[f"{mae_column}_source_only"]
    merged["tatn_mae_percent"] = merged[f"{mae_column}_tatn"]
    merged["delta_mae_percent"] = merged["tatn_mae_percent"] - merged["source_only_mae_percent"]
    merged["source_only_rmse_percent"] = merged[f"{rmse_column}_source_only"]
    merged["tatn_rmse_percent"] = merged[f"{rmse_column}_tatn"]
    merged["delta_rmse_percent"] = merged["tatn_rmse_percent"] - merged["source_only_rmse_percent"]
    merged["negative_transfer_mae"] = merged["delta_mae_percent"] > 0
    merged["negative_transfer_rmse"] = merged["delta_rmse_percent"] > 0
    if bias_column:
        merged["source_only_bias_percent"] = merged[f"{bias_column}_source_only"]
        merged["tatn_bias_percent"] = merged[f"{bias_column}_tatn"]
        merged["delta_abs_bias_percent"] = (
            merged["tatn_bias_percent"].abs() - merged["source_only_bias_percent"].abs()
        )
    return merged[columns]


def _compute_phase_transfer_effects(phase_metrics: pd.DataFrame) -> pd.DataFrame:
    """delta = TATN - Source only per (repeat, source_kind, scenario, cell_id,
    condition, mission_phase); negative_transfer_* = delta > 0."""
    keys = (["repeat"] if "repeat" in phase_metrics.columns else []) + [
        "source_kind", "scenario", "cell_id", "condition", "mission_phase",
    ]
    return _pair_source_only_vs_tatn(
        phase_metrics, keys, "mae_percent", "rmse_percent",
        bias_column=(
            "mean_signed_error_percent"
            if "mean_signed_error_percent" in phase_metrics.columns
            else None
        ),
    )


def _compute_cell_transfer_effects(condition_metrics_path: Path) -> pd.DataFrame:
    """Overall (non-phase) counterpart read back from condition_metrics.csv."""
    frame = pd.read_csv(condition_metrics_path)
    keys = (["repeat"] if "repeat" in frame.columns else []) + [
        "source_kind", "scenario", "cell_id", "condition",
    ]
    return _pair_source_only_vs_tatn(
        frame, keys, "overall_mae_percent", "overall_rmse_percent",
        bias_column="overall_bias_percent" if "overall_bias_percent" in frame.columns else None,
    )


# --------------------------------------------------------------------------
# Cell-level (overall) negative-transfer table and summaries
# --------------------------------------------------------------------------
_OVERALL_METRICS = (
    "overall_rmse_percent", "overall_mae_percent",
    "mission_macro_rmse_percent", "mission_macro_mae_percent",
    "overall_bias_percent", "mission_macro_bias_percent",
)


def build_cell_metrics(condition_metrics: pd.DataFrame, cell_info: pd.DataFrame) -> pd.DataFrame:
    """One row per (repeat, cell): Source only vs TATN overall/mission-macro
    RMSE/MAE and bias, deltas (TATN - Source only), negative-transfer flags,
    plus adaptation epochs completed, runtimes and pseudo-label statistics."""
    keys = ["repeat", "cell_id"]
    source = condition_metrics.loc[condition_metrics["method"] == METHOD_SOURCE_ONLY]
    tatn = condition_metrics.loc[condition_metrics["method"] == METHOD_TATN]
    lead = ["repeat", "seed", "source_kind", "scenario", "cell_id", "condition",
            "test_missions", "test_points"]
    merged = source[lead + list(_OVERALL_METRICS)].merge(
        tatn[keys + list(_OVERALL_METRICS)], on=keys, suffixes=("_source_only", "_tatn")
    )
    out = merged[lead].copy()
    for metric in _OVERALL_METRICS:
        out[f"source_only_{metric}"] = merged[f"{metric}_source_only"]
        out[f"tatn_{metric}"] = merged[f"{metric}_tatn"]
        out[f"delta_{metric}"] = merged[f"{metric}_tatn"] - merged[f"{metric}_source_only"]
    for metric in _OVERALL_METRICS[:4]:
        out[f"negative_transfer_{metric.removesuffix('_percent')}"] = (
            out[f"delta_{metric}"] > 0
        )
    out["delta_abs_overall_bias_percent"] = (
        out["tatn_overall_bias_percent"].abs() - out["source_only_overall_bias_percent"].abs()
    )
    return out.merge(cell_info, on=keys, how="left")


_SUMMARY_VALUE_PREFIXES = ("source_only_", "tatn_", "delta_")


def _summary_value_columns(frame: pd.DataFrame) -> list[str]:
    columns = [c for c in frame.columns if c.startswith(_SUMMARY_VALUE_PREFIXES)]
    columns += [
        c for c in ("adaptation_runtime_seconds", "adaptation_wall_clock_seconds",
                    "epochs_completed", "pseudo_accepted_fraction")
        if c in frame.columns
    ]
    return [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]


def _mean_std_frame(frame: pd.DataFrame, group_keys: list[str], values: list[str]) -> pd.DataFrame:
    flags = [c for c in frame.columns if c.startswith("negative_transfer_")]
    if not group_keys:
        grouped = [((), frame)]
    else:
        grouped = list(frame.groupby(group_keys, sort=False))
    rows = []
    for key, part in grouped:
        row = dict(zip(group_keys, key if isinstance(key, tuple) else (key,)))
        row["n"] = len(part)
        for column in values:
            row[f"{column}_mean"] = float(part[column].mean())
            row[f"{column}_std"] = float(part[column].std(ddof=1)) if len(part) > 1 else float("nan")
        for column in flags:
            row[f"{column}_rate"] = float(part[column].astype(float).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    """summary.csv: mean/std over repeats per cell (``cell_across_repeats``),
    over cells per repeat then across repeats (``core7_across_repeats``: each
    repeat's seven-cell mean is one sample) and over the per-cell repeat means
    (``core7_across_cells``). Std is the sample std (ddof=1); NaN when n=1."""
    values = _summary_value_columns(cell_metrics)
    flags = [c for c in cell_metrics.columns if c.startswith("negative_transfer_")]
    per_cell = _mean_std_frame(cell_metrics, ["cell_id"], values)
    per_cell.insert(0, "scope", "cell_across_repeats")
    per_repeat = cell_metrics.groupby("repeat", sort=True)[values + flags].mean().reset_index()
    across_repeats = _mean_std_frame(per_repeat, [], values)
    across_repeats.insert(0, "scope", "core7_across_repeats")
    cell_means = cell_metrics.groupby("cell_id", sort=False)[values + flags].mean().reset_index()
    across_cells = _mean_std_frame(cell_means, [], values)
    across_cells.insert(0, "scope", "core7_across_cells")
    return pd.concat([per_cell, across_repeats, across_cells], ignore_index=True)


def build_phase_summary(table: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    """Mean/std across repeats per (cell, phase, method) plus the core-seven
    aggregate (per-repeat mean over cells, then mean/std across repeats)."""
    keys = [k for k in ("method", "mission_phase") if k in table.columns]
    flags = [c for c in table.columns if c.startswith("negative_transfer_")]
    per_cell = _mean_std_frame(table, ["cell_id", *keys], value_columns)
    per_cell.insert(0, "scope", "cell_across_repeats")
    per_repeat = (
        table.groupby(["repeat", *keys], sort=False)[value_columns + flags].mean().reset_index()
    )
    core = _mean_std_frame(per_repeat, keys, value_columns)
    core.insert(0, "scope", "core7_across_repeats")
    core.insert(1, "cell_id", "core7")
    return pd.concat([per_cell, core], ignore_index=True)


# --------------------------------------------------------------------------
# History helpers
# --------------------------------------------------------------------------
def _normalize_history(
    rows: list[dict], *, repeat: int, seed: int, cell_id: str, stage: str
) -> pd.DataFrame:
    """Map shen_tatn.train per-epoch dicts onto ``HISTORY_COLUMNS`` (missing
    values stay NaN, never fabricated) and keep legacy adaptation columns."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(HISTORY_COLUMNS))
    out = pd.DataFrame(index=frame.index)
    out["repeat"], out["seed"], out["cell_id"], out["stage"] = repeat, seed, cell_id, stage
    for column in HISTORY_COLUMNS[4:]:
        out[column] = frame[column] if column in frame.columns else np.nan
    if stage == "source":
        out["validation_evaluated"] = False
    legacy = [c for c in frame.columns if c not in out.columns]
    return pd.concat([out, frame[legacy]], axis=1)


def _check_resume_tatn_result(
    *, result_path: Path, cell_output: Path, expected: dict[str, object]
) -> None:
    """Resume-safety check, run ONLY where --resume is about to skip a
    cell/repeat because ``result_path`` already exists.

    Every field in ``RESUME_CHECK_FIELDS`` (output schema version, manifest /
    CMU artifact / source-checkpoint hashes, source kind, scenario, repeat,
    seeds, epochs, batch size, tensor layout, all hyperparameters) must be
    present and equal this invocation's value exactly. A missing field (a
    legacy output predating the check, e.g. the August runs) or any mismatch
    raises ``ValueError``. All sidecars (``target.pt``,
    ``training_history.csv``, ``test_predictions.parquet``) must exist AND be
    readable; a leftover ``*.incomplete`` file is never treated as the final
    file. Nothing is ever overwritten or deleted."""
    try:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{result_path}: unreadable result ({error}); refusing to --resume") from error
    for field in RESUME_CHECK_FIELDS:
        if field not in existing:
            raise ValueError(
                f"{result_path}: existing result has no {field!r} recorded -- cannot "
                "verify it was produced from the current configuration (legacy "
                "output?); refusing to --resume. Use a fresh --output-root."
            )
        if existing[field] != expected[field]:
            raise ValueError(
                f"{result_path}: existing result's {field!r}={existing[field]!r} does not "
                f"match the current run's {field!r}={expected[field]!r} -- refusing to "
                "--resume from a different configuration. Use a fresh --output-root."
            )
    target_checkpoint = cell_output / "target.pt"
    if not target_checkpoint.is_file() or target_checkpoint.stat().st_size == 0:
        raise ValueError(
            f"{result_path}: --resume would skip this cell, but its target checkpoint "
            f"{target_checkpoint} is missing or empty."
        )
    try:
        torch.load(target_checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - any unreadable checkpoint
        raise ValueError(f"{target_checkpoint}: unreadable ({error})") from error
    history_path = cell_output / "training_history.csv"
    if not history_path.is_file():
        raise ValueError(
            f"{result_path}: --resume would skip this cell, but its training_history.csv "
            f"sidecar {history_path} is missing."
        )
    try:
        pd.read_csv(history_path, nrows=1)
    except (pd.errors.EmptyDataError, OSError, pd.errors.ParserError) as error:
        raise ValueError(f"{history_path}: unreadable ({error})") from error
    predictions_path = cell_output / "test_predictions.parquet"
    if not predictions_path.is_file():
        raise ValueError(
            f"{result_path}: --resume would skip this cell, but its test_predictions.parquet "
            f"sidecar {predictions_path} is missing (a leftover .incomplete file is never "
            "treated as complete)."
        )
    try:
        pq.read_metadata(predictions_path)
    except Exception as error:  # noqa: BLE001
        raise ValueError(f"{predictions_path}: unreadable ({error})") from error


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
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
        # New, isolated default: never reads from or writes into an older
        # (e.g. August legacy) output directory.
        default=REPO_ROOT
        / "outputs/benchmarks/panasonic_to_cmu_evtol/tatn_ns_relabel_v2_unsupervised_formal",
    )
    parser.add_argument("--source-kind", choices=("trained", "released"), required=True)
    parser.add_argument("--trained-checkpoint", type=Path)
    parser.add_argument("--released-checkpoint", type=Path)
    parser.add_argument(
        "--scenario", choices=("unsupervised", "semisupervised"), default="unsupervised",
        help="'unsupervised' is the primary protocol (no CMU SOC label used for "
        "adaptation). 'semisupervised' is retained only for legacy comparisons.",
    )
    parser.add_argument(
        "--cmu-manifest", type=Path,
        default=REPO_ROOT
        / "data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1_ns_relabel.csv",
    )
    parser.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS))
    parser.add_argument("--source-epochs", type=int, default=300)
    parser.add_argument("--transfer-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=100, help="Base seed (repeat 1 uses it unchanged).")
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Independent repeats; repeat r uses seed base + 1000*(r-1) and its own "
        "source state (trained source only).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--validation-interval", type=int, default=10,
        help="Run the read-only validation diagnostic every Nth adaptation epoch, plus "
        "epoch 1 and the actual last epoch. Diagnostic only -- never affects training, "
        "pseudo-labels, early stopping, the scheduler, or checkpoint selection.",
    )
    parser.add_argument(
        "--reference-run-root", type=Path, nargs="*", default=[],
        help="Finished Backbone/DARM/SFTNN/DTNN run roots whose recorded manifest/CMU "
        "artifact hashes must equal this run's.",
    )
    parser.add_argument(
        "--skip-reference-check", action="store_true",
        help="Explicitly skip the cross-run hash comparison (recorded in metadata; the "
        "run is then not formal-benchmark eligible). The manifest check always runs.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.trained_checkpoint is not None and args.repeats > 1:
        parser.error("--trained-checkpoint is a single fixed source state; use --repeats 1")
    return args


# --------------------------------------------------------------------------
# Output root safety
# --------------------------------------------------------------------------
def _prepare_output_root(output_root: Path, resume: bool) -> dict | None:
    """Return previous run metadata (resume) or None (fresh). Never deletes or
    overwrites: a non-empty root without --resume, a root without a
    schema-versioned run_metadata.json (legacy), or one holding the legacy
    top-level per-cell layout is refused."""
    if not output_root.exists() or not any(output_root.iterdir()):
        output_root.mkdir(parents=True, exist_ok=True)
        return None
    if not resume:
        raise FileExistsError(
            f"{output_root} is not empty. Use a fresh --output-root, or --resume to "
            "continue a run of the same configuration (nothing is ever overwritten)."
        )
    metadata_path = output_root / "run_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"{output_root}: no run_metadata.json -- legacy/foreign output; refusing --resume")
    try:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{metadata_path}: unreadable ({error})") from error
    if previous.get("output_schema_version") != OUTPUT_SCHEMA_VERSION:
        raise ValueError(
            f"{output_root}: output_schema_version={previous.get('output_schema_version')!r} "
            f"!= {OUTPUT_SCHEMA_VERSION!r} (legacy output); refusing --resume"
        )
    legacy_layout = [c for c in CELLS if (output_root / c / "results.json").is_file()]
    if legacy_layout:
        raise ValueError(f"{output_root}: legacy per-cell layout {legacy_layout}; refusing --resume")
    return previous


# --------------------------------------------------------------------------
# Source model per repeat
# --------------------------------------------------------------------------
def _source_protocol(args: argparse.Namespace, seed: int) -> dict[str, object]:
    return {
        "epochs": args.source_epochs,
        "batch_size": args.batch_size,
        "learning_rate": TATN_LEARNING_RATE,
        "seed": seed,
        "early_stopping_patience": SOURCE_EARLY_STOPPING_PATIENCE,
    }


def _obtain_source(args, repeat, device, source_cycles, output_root):
    """Return ``(model, info)``. Trained: a per-repeat independent source state
    (``source_25C_repeatNN.pt``), trained here unless a checkpoint with a
    matching recorded protocol already exists. Released: the official
    pre-25.pt (hash-verified), shared by every repeat."""
    seed = repeat_seed(args.seed, repeat)
    if args.source_kind == "released":
        checkpoint = args.released_checkpoint
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError("--released-checkpoint must point to pre-25.pt")
        if sha256_file(checkpoint) != OFFICIAL_PRE25_SHA256:
            raise ValueError("Released pre-25.pt SHA-256 mismatch")
        model = load_released_source(checkpoint, device)
        return model, {
            "repeat": repeat, "path": str(checkpoint), "sha256": sha256_file(checkpoint),
            "trained_this_run": False, "independent_per_repeat": False,
            "training_seconds_sum": 0.0, "wall_clock_seconds": 0.0, "epochs_completed": None,
        }
    user_checkpoint = args.trained_checkpoint
    checkpoint = Path(user_checkpoint) if user_checkpoint else _source_checkpoint_path(output_root, repeat)
    repeat_dir = _repeat_dir(output_root, repeat)
    summary_path = repeat_dir / "source_training_summary.json"
    if checkpoint.is_file():
        if user_checkpoint is None:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            recorded = payload.get("protocol", {})
            expected = _source_protocol(args, seed)
            for key, value in expected.items():
                if recorded.get(key) != value:
                    raise ValueError(
                        f"{checkpoint}: recorded source protocol {key}={recorded.get(key)!r} "
                        f"!= current {value!r}; refusing to reuse it"
                    )
        model = load_source(checkpoint, device)
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        )
        return model, {
            "repeat": repeat, "path": str(checkpoint), "sha256": sha256_file(checkpoint),
            "trained_this_run": False, "independent_per_repeat": user_checkpoint is None,
            "training_seconds_sum": summary.get("training_seconds_sum"),
            "wall_clock_seconds": summary.get("wall_clock_seconds"),
            "epochs_completed": summary.get("epochs_completed"),
        }
    started = time.perf_counter()
    model, history, diagnostics = train_source(
        source_cycles,
        epochs=args.source_epochs,
        batch_size=args.batch_size,
        learning_rate=TATN_LEARNING_RATE,
        device=device,
        seed=seed,
        checkpoint=checkpoint,
        early_stopping_patience=SOURCE_EARLY_STOPPING_PATIENCE,
        min_epochs=SOURCE_MIN_EPOCHS,
        tensor_layout="paper",
        record_epoch_seconds=True,
    )
    wall = time.perf_counter() - started
    repeat_dir.mkdir(parents=True, exist_ok=True)
    history_frame = _normalize_history(diagnostics, repeat=repeat, seed=seed, cell_id="", stage="source")
    _write_csv_atomic(history_frame, repeat_dir / "source_training_history.csv")
    training_seconds = float(sum(row["runtime_seconds"] for row in diagnostics))
    summary = {
        "repeat": repeat, "seed": seed, "epochs_completed": len(history),
        "training_seconds_sum": training_seconds, "wall_clock_seconds": wall,
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    _write_json_atomic(summary, summary_path)
    return model, {
        "repeat": repeat, "path": str(checkpoint), "sha256": sha256_file(checkpoint),
        "trained_this_run": True, "independent_per_repeat": True,
        "training_seconds_sum": training_seconds, "wall_clock_seconds": wall,
        "epochs_completed": len(history),
    }


# --------------------------------------------------------------------------
# Per-cell data (loaded once, reused by every repeat)
# --------------------------------------------------------------------------
def _load_cell_data(args, cell_id, source_path, manifest, feature_min, feature_max) -> dict:
    adaptation_ids, validation_ids, test_ids, budget = select_duration_matched_missions(
        manifest, cell_id
    )
    feature_records = load_cmu_features_for_uda([source_path], target_hz=1.0).records
    labelled_records = load_cmu_dynamic_soc_for_vidal([source_path], target_hz=1.0).records
    adaptation_cycles = _to_shen_cycles(
        _records_for_ids(feature_records, adaptation_ids),
        feature_min, feature_max, require_labels=False,
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
            feature_min, feature_max, require_labels=True,
        )[0]
        adaptation_cycles = [
            labeled_cycle if cycle.profile == labeled_profile else cycle
            for cycle in adaptation_cycles
        ]
    test_cycles = _to_shen_cycles(
        _records_for_ids(labelled_records, test_ids), feature_min, feature_max, require_labels=True
    )
    if not adaptation_cycles or not test_cycles:
        raise ValueError(f"{cell_id}: missing adaptation or test missions")
    # Diagnostic-only validation probe input (manifest "validation" split).
    validation_cycles = (
        _to_shen_cycles(
            _records_for_ids(labelled_records, validation_ids),
            feature_min, feature_max, require_labels=True,
        )
        if validation_ids
        else None
    )
    _assert_no_target_label_leak(
        adaptation_cycles,
        scenario=args.scenario,
        labeled_profile=labeled_profile,
        adaptation_ids=adaptation_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
    )
    return {
        "budget": budget,
        "adaptation_cycles": adaptation_cycles,
        "labeled_profile": labeled_profile,
        "test_cycles": test_cycles,
        "validation_cycles": validation_cycles,
        "prediction_metadata": _mission_metadata(labelled_records, test_ids),
        "raw_diagnostic": load_raw_diagnostic_fields(source_path),
    }


def _parameter_counts(model) -> dict[str, int]:
    probe = deepcopy(model).cpu()
    if getattr(probe, "target_extractor", None) is None or not list(probe.target_extractor.parameters()):
        probe.initialize_target()
    counts = {
        name: int(sum(p.numel() for p in getattr(probe, name).parameters()))
        for name in ("source_extractor", "predictor", "target_extractor", "discriminator")
        if getattr(probe, name, None) is not None
    }
    counts["total"] = int(sum(counts.values()))
    counts["trainable_in_adaptation"] = int(
        counts.get("target_extractor", 0) + counts.get("discriminator", 0)
    )
    counts["trainable_in_source_training"] = int(
        counts.get("source_extractor", 0) + counts.get("predictor", 0)
    )
    return counts


# --------------------------------------------------------------------------
# One (repeat, cell)
# --------------------------------------------------------------------------
def _run_cell(
    args, *, repeat, cell_id, cell_index_seed, data, source_model, source_cycles, device,
    tensor_layout, cell_output, resume_expected, input_sha256, source_info, feature_min,
    feature_max, source_fingerprint,
):
    cell_output.mkdir(parents=True, exist_ok=True)
    cell = data
    budget = cell["budget"]
    source_only, so_truth, so_estimate = _evaluate(
        source_model, cell["test_cycles"], device, tensor_layout, return_predictions=True
    )
    source_only.update(_bias_metrics(so_truth, so_estimate, cell["test_cycles"]))

    wall_start = time.perf_counter()
    transferred, history, target_info = adapt(
        source_model,
        source_cycles,
        cell["adaptation_cycles"],
        test_profile=None,
        scenario=args.scenario,
        epochs=args.transfer_epochs,
        batch_size=args.batch_size,
        learning_rate=TATN_LEARNING_RATE,
        lambdas=TATN_LAMBDAS,
        device=device,
        seed=cell_index_seed,
        labeled_profile=cell["labeled_profile"],
        early_stopping_patience=TRANSFER_EARLY_STOPPING_PATIENCE,
        min_epochs=TRANSFER_MIN_EPOCHS,
        tensor_layout=tensor_layout,
        record_epoch_seconds=True,
        validation_cycles=cell["validation_cycles"],
        validation_interval=args.validation_interval,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    adaptation_wall = time.perf_counter() - wall_start
    # Cell independence: adapt() works on a deep copy; the frozen source state
    # shared by every cell of this repeat must be byte-identical afterwards.
    if _state_fingerprint(source_model) != source_fingerprint:
        raise RuntimeError(f"{cell_id}: frozen source state changed during adaptation (leakage)")

    tatn, tatn_truth, tatn_estimate = _evaluate(
        transferred, cell["test_cycles"], device, tensor_layout, return_predictions=True
    )
    tatn.update(_bias_metrics(tatn_truth, tatn_estimate, cell["test_cycles"]))

    history_frame = _normalize_history(
        history, repeat=repeat, seed=cell_index_seed, cell_id=cell_id, stage="adaptation"
    )
    _write_csv_atomic(history_frame, cell_output / "training_history.csv")

    mission_ids, window_idx, timestep_idx, pred_metadata = _flatten_prediction_metadata(
        cell["test_cycles"], cell["prediction_metadata"]
    )
    tables = []
    for method, truth, estimate in (
        (METHOD_SOURCE_ONLY, so_truth, so_estimate),
        (METHOD_TATN, tatn_truth, tatn_estimate),
    ):
        tables.append(
            _build_prediction_table(
                framework=FRAMEWORK, method=method, source_kind=args.source_kind,
                scenario=args.scenario, repeat=repeat, seed=cell_index_seed,
                cell_id=cell_id, condition=budget["condition"], soc_reference=truth,
                soc_prediction=estimate, mission_ids=mission_ids, window_indices=window_idx,
                timestep_indices=timestep_idx, metadata=pred_metadata,
                raw_diagnostic=cell["raw_diagnostic"],
            )
        )
    _write_test_predictions_atomic(tables, TEST_PREDICTIONS_SCHEMA, cell_output / "test_predictions.parquet")
    torch.save(
        {"model_state_dict": transferred.state_dict(), "history": history, "target_info": target_info},
        cell_output / "target.pt",
    )

    metrics = [
        {
            "repeat": repeat, "seed": cell_index_seed, "source_kind": args.source_kind,
            "scenario": args.scenario, "cell_id": cell_id, "condition": budget["condition"],
            "method": method, **values,
        }
        for method, values in ((METHOD_SOURCE_ONLY, source_only), (METHOD_TATN, tatn))
    ]
    runtime_sum = float(sum(row.get("runtime_seconds") or 0.0 for row in history))
    validation_sum = float(sum(row.get("validation_runtime_seconds") or 0.0 for row in history))
    last = history[-1] if history else {}
    cell_info = {
        "repeat": repeat, "cell_id": cell_id,
        "epochs_completed": len(history),
        "adaptation_runtime_seconds": runtime_sum,
        "adaptation_wall_clock_seconds": adaptation_wall,
        "validation_probe_seconds": validation_sum,
        "pseudo_windows": target_info.get("pseudo_windows"),
        "pseudo_candidate_windows": last.get("pseudo_candidate_windows"),
        "pseudo_accepted_fraction": last.get("pseudo_accepted_fraction"),
        "adaptation_status": target_info.get("status"),
    }
    result = {
        "benchmark": "shen_2023_tatn_panasonic_25c_to_cmu_core7",
        "source_checkpoint": source_info["path"],
        "source_profiles": list(PROFILES),
        "source_normalization": "Panasonic 25 C all-nine-profile min-max",
        "source_feature_min": feature_min.tolist(),
        "source_feature_max": feature_max.tolist(),
        "cell_id": cell_id,
        "cmu_manifest_path": str(args.cmu_manifest),
        "adaptation_budget": budget,
        "adaptation_missions": len(cell["adaptation_cycles"]),
        "labeled_target_mission": cell["labeled_profile"],
        "labeled_target_fraction": 0.5 if cell["labeled_profile"] else 0.0,
        "test_missions": len(cell["test_cycles"]),
        "validation_missions": len(cell["validation_cycles"] or []),
        "target_info": target_info,
        "epochs_completed": len(history),
        "metrics": metrics,
        "cell_info": cell_info,
        "source_state_sha256": source_fingerprint,
        "formal_benchmark_eligible": False,  # decided at run level (run_metadata.json)
        "evaluation_unit": EVALUATION_UNIT,
        **_label_flags(args.scenario),
        "label_is_ground_truth": False,
        **resume_expected,
        "cmu_input_sha256": input_sha256,
    }
    # results.json is written LAST: its presence marks every sidecar complete.
    _write_json_atomic(result, cell_output / "results.json")
    return result


def _label_flags(scenario: str) -> dict[str, bool]:
    if scenario == "unsupervised":
        return dict(LABEL_FLAGS_UNSUPERVISED)
    return {
        "target_soc_labels_used_for_training": True,
        "target_soc_labels_used_for_adaptation": True,
        "target_soc_labels_used_for_pseudo_labels": False,
        "target_soc_labels_used_for_checkpoint_selection": False,
        "target_soc_reference_used_for_posthoc_evaluation": True,
    }


# --------------------------------------------------------------------------
# Aggregation (always from the files on disk, fresh and resumed alike)
# --------------------------------------------------------------------------
def _aggregate(output_root: Path, repeats: int, cells: list[str]) -> dict[str, object]:
    results, history_frames, prediction_paths, source_history = [], [], [], []
    for repeat in range(1, repeats + 1):
        source_path = _repeat_dir(output_root, repeat) / "source_training_history.csv"
        if source_path.is_file():
            source_history.append(pd.read_csv(source_path))
        for cell_id in cells:
            cell_output = _repeat_dir(output_root, repeat) / cell_id
            result_path = cell_output / "results.json"
            if not result_path.is_file():
                continue
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
            history_frames.append(pd.read_csv(cell_output / "training_history.csv"))
            prediction_paths.append(cell_output / "test_predictions.parquet")
    summary: dict[str, object] = {"n_cell_results": len(results), "results": results}
    if not results:
        return summary
    condition_metrics = pd.DataFrame([row for r in results for row in r["metrics"]])
    _write_csv_atomic(condition_metrics, output_root / "condition_metrics.csv")
    cell_info = pd.DataFrame([r["cell_info"] for r in results])
    cell_metrics = build_cell_metrics(condition_metrics, cell_info)
    _write_csv_atomic(cell_metrics, output_root / "cell_metrics.csv")
    _write_csv_atomic(build_summary(cell_metrics), output_root / "summary.csv")
    _write_csv_atomic(pd.concat(history_frames, ignore_index=True), output_root / "training_history.csv")
    if source_history:
        _write_csv_atomic(pd.concat(source_history, ignore_index=True), output_root / "source_training_history.csv")
    predictions_path = output_root / "test_predictions.parquet"
    _concat_parquets_atomic(prediction_paths, TEST_PREDICTIONS_SCHEMA, predictions_path)

    phase_metrics = _compute_phase_metrics(predictions_path)
    phase_effects = _compute_phase_transfer_effects(phase_metrics)
    _write_csv_atomic(phase_metrics, output_root / "phase_metrics.csv")
    _write_csv_atomic(phase_effects, output_root / "phase_transfer_effects.csv")
    if not phase_metrics.empty:
        metric_cols = ["n_points", "n_missions", "mae_percent", "rmse_percent", "mean_signed_error_percent"]
        _write_csv_atomic(build_phase_summary(phase_metrics, metric_cols), output_root / "phase_metrics_summary.csv")
    if not phase_effects.empty:
        effect_cols = [c for c in phase_effects.columns if c.startswith(("source_only_", "tatn_", "delta_"))]
        _write_csv_atomic(
            build_phase_summary(phase_effects, effect_cols), output_root / "phase_transfer_effects_summary.csv"
        )
    # Full-mission diagnostic: independent recomputation over EVERY phase label
    # present in the held-out test missions (unmapped -> "unmapped"), derived
    # from the same primary parquet. Never used in training/adaptation/early
    # stopping/checkpoint selection and never replaces the primary metrics.
    full_metrics = _compute_phase_metrics(predictions_path, phases=None)
    full_effects = _compute_phase_transfer_effects(full_metrics)
    for frame in (full_metrics, full_effects):
        frame.insert(0, "evaluation_scope", "full_mission_diagnostic")
    _write_csv_atomic(full_metrics, output_root / "phase_metrics_full_mission_diagnostic.csv")
    _write_csv_atomic(full_effects, output_root / "phase_transfer_effects_full_mission_diagnostic.csv")
    summary["cell_metrics"] = cell_metrics
    return summary


def _build_run_metadata(
    args, *, status, protocol_record, source_infos, device, tensor_layout, feature_min,
    feature_max, aggregate, parameter_counts, invocation_seconds, previous, cell_artifacts,
) -> dict[str, object]:
    results = aggregate.get("results", [])
    expected_cells = len(args.cells) * args.repeats
    reasons = []
    if args.scenario != "unsupervised":
        reasons.append("scenario is not unsupervised")
    if args.source_kind != "trained":
        reasons.append("source_kind is not the paper-layout trained primary source")
    if protocol_record.get("reference_check") != "passed":
        reasons.append("cross-run reference hash check was skipped")
    if args.repeats < 5:
        reasons.append("fewer than 5 repeats (screening)")
    if set(args.cells) != set(CELLS):
        reasons.append("not all seven core cells")
    if len(results) != expected_cells or status != "completed":
        reasons.append("run incomplete")
    if args.trained_checkpoint is not None:
        reasons.append("externally supplied source checkpoint (single fixed source state)")
    is_dirty_tree = is_dirty(REPO_ROOT)
    adaptation_sum = float(sum(r["cell_info"]["adaptation_runtime_seconds"] for r in results))
    adaptation_wall = float(sum(r["cell_info"]["adaptation_wall_clock_seconds"] for r in results))
    source_sum = [i["training_seconds_sum"] for i in source_infos]
    source_wall = [i["wall_clock_seconds"] for i in source_infos]
    previous_wall = float((previous or {}).get("cumulative_wall_clock_seconds") or 0.0)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    return {
        "benchmark": "shen_2023_tatn_panasonic_25c_to_cmu_core7",
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "status": status,
        "git_commit": get_full_commit_hash(REPO_ROOT),
        "git_dirty": is_dirty_tree,
        "protocol_verification": protocol_record,
        "cmu_manifest_path": str(args.cmu_manifest),
        "cmu_manifest_sha256": protocol_record["manifest_sha256"],
        "cmu_base_root": str(args.cmu_base_root),
        "cmu_condition_root": str(args.cmu_condition_root),
        "cmu_artifacts": {
            cell: {"path": str(path), "sha256": protocol_record["cmu_artifact_sha256"][cell]}
            for cell, path in cell_artifacts.items()
        },
        "source_kind": args.source_kind,
        "scenario": args.scenario,
        "source_dataset": "Panasonic NCR18650PF, 25 C, all nine profiles",
        "source_checkpoints": source_infos,
        "source_checkpoint_sha256": (
            source_infos[0]["sha256"]
            if source_infos and len({i["sha256"] for i in source_infos}) == 1 else None
        ),
        "source_independent_per_repeat": all(i["independent_per_repeat"] for i in source_infos),
        "tensor_layout": tensor_layout,
        "source_normalization": "Panasonic 25 C all-nine-profile min-max (source-global)",
        "target_normalization": (
            "source-global: the Panasonic range is applied unchanged to CMU; no target "
            "statistics are fitted"
        ),
        "source_feature_min": feature_min.tolist(),
        "source_feature_max": feature_max.tolist(),
        "feature_order": list(FEATURES),
        "source_hyperparameters": {
            "epochs": args.source_epochs if args.source_kind == "trained" else None,
            "batch_size": args.batch_size, "learning_rate": TATN_LEARNING_RATE,
            "early_stopping_patience": SOURCE_EARLY_STOPPING_PATIENCE,
            "min_epochs": SOURCE_MIN_EPOCHS, "optimizer": "Adam", "scheduler": "ReduceLROnPlateau",
        },
        "transfer_hyperparameters": {
            "epochs": args.transfer_epochs, "batch_size": args.batch_size,
            "learning_rate": TATN_LEARNING_RATE, "lambdas": list(TATN_LAMBDAS),
            "lambda_order": ["estimate(pseudo-label regression)", "adversarial", "mmd"],
            "early_stopping_patience": TRANSFER_EARLY_STOPPING_PATIENCE,
            "min_epochs": TRANSFER_MIN_EPOCHS,
            "pseudo_label_threshold": PSEUDO_LABEL_THRESHOLD,
            "pseudo_label_lookahead": PSEUDO_LABEL_LOOKAHEAD,
            "checkpoint_selection": "lowest training total loss (label-safe)",
        },
        "adaptation_components": "target extractor + discriminator (DANN), gaussian MMD, "
        "pseudo-label selection (original TATN; nothing simplified)",
        "base_seed": args.seed,
        "repeats": args.repeats,
        "repeat_seeds": [repeat_seed(args.seed, r) for r in range(1, args.repeats + 1)],
        "cell_seeds": {
            str(r): {c: cell_seed(args.seed, r, c) for c in args.cells}
            for r in range(1, args.repeats + 1)
        },
        "seed_rule": "repeat seed = base + 1000*(repeat-1); cell adaptation seed = repeat seed + "
        "index of the cell in the canonical core-seven order",
        "cells": list(args.cells),
        "evaluation_unit": EVALUATION_UNIT,
        "evaluation_unit_note": (
            "predict()/_evaluate score every raw timestep inside each non-overlapping 90-step "
            "window (all_timesteps), same as DTNN/backbone/SFTNN, unlike DARM's "
            "last_step_per_window."
        ),
        "metric_units": {
            "*_percent": "percent (SOC points x 100)",
            "soc_reference": "fraction", "soc_prediction": "fraction",
            "signed_error": "fraction (prediction - reference)", "absolute_error": "fraction",
            "runtime_seconds": "seconds (training-only wall clock per epoch)",
            "validation_runtime_seconds": "seconds", "voltage_v_raw": "V",
            "current_a_raw": "A", "temperature_c_raw": "degC",
        },
        **_label_flags(args.scenario),
        "target_soc_reference_note": (
            "CMU SOC is used only for post-hoc evaluation and the diagnostic-only validation "
            "probe; never for training, adaptation, pseudo-labels, early stopping, scheduler or "
            "checkpoint selection."
        ),
        "validation_probe": {
            "diagnostic_only": True,
            "interval_epochs": args.validation_interval,
            "split": "manifest validation split",
            "used_for_early_stopping": False,
            "used_for_checkpoint_selection": False,
            "used_for_pseudo_labels": False,
            "used_for_hyperparameter_selection": False,
            "test_set_used_for_selection": False,
        },
        "phase_mapping_version": PHASE_MAPPING_VERSION,
        "phase_mapping_sha256": PHASE_MAPPING_SHA256,
        "phase_definition": "canonical Ns mapping: 4=take_off, 5=cruise, 6=landing",
        "phase_files": {
            "primary": "phase_metrics.csv / phase_transfer_effects.csv "
            "(take_off/cruise/landing; repeat x cell x phase x method)",
            "summaries": "phase_metrics_summary.csv / phase_transfer_effects_summary.csv "
            "(cell x phase mean/std across repeats; core7 aggregate)",
            "full_mission_diagnostic": "phase_*_full_mission_diagnostic.csv: every phase label "
            "present in the test missions, derived from the same primary parquet "
            "(CMU mission candidates are flight-only, so it equals the primary test unless an "
            "unmapped Ns appears)",
        },
        "history_columns": list(HISTORY_COLUMNS),
        "test_predictions_columns": TEST_PREDICTIONS_SCHEMA.names,
        "formal_benchmark_eligible": not reasons,
        "formal_benchmark_ineligibility_reasons": reasons,
        "runtime": {
            "total_source_training_seconds": (
                float(sum(s for s in source_sum if s is not None)) if args.source_kind == "trained" else 0.0
            ),
            "source_training_seconds_per_repeat": source_sum,
            "source_training_wall_clock_seconds_per_repeat": source_wall,
            "total_adaptation_training_seconds": adaptation_sum,
            "total_adaptation_wall_clock_seconds": adaptation_wall,
            "invocation_wall_clock_seconds": invocation_seconds,
        },
        "cumulative_wall_clock_seconds": previous_wall + invocation_seconds,
        "compute_cost": {
            "device": str(device), "gpu_name": gpu, "torch_version": torch.__version__,
            "python_version": platform.python_version(), "platform": platform.platform(),
            "cuda_version": torch.version.cuda,
            "peak_cuda_memory_allocated_mb": (
                torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
            ),
            "parameter_counts": parameter_counts,
        },
        "outputs": [
            "run_metadata.json", "condition_metrics.csv", "cell_metrics.csv", "summary.csv",
            "training_history.csv", "source_training_history.csv", "test_predictions.parquet",
            "phase_metrics.csv", "phase_transfer_effects.csv", "phase_metrics_summary.csv",
            "phase_transfer_effects_summary.csv",
            "phase_metrics_full_mission_diagnostic.csv",
            "phase_transfer_effects_full_mission_diagnostic.csv",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    device = get_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    previous = _prepare_output_root(args.output_root, args.resume)

    cell_artifacts = {
        cell: _cell_path(cell, args.cmu_base_root, args.cmu_condition_root) for cell in args.cells
    }
    protocol_record = verify_cmu_protocol(
        args.cmu_manifest,
        cell_artifacts,
        reference_run_roots=args.reference_run_root,
        require_reference=not args.skip_reference_check,
    )
    manifest = pd.read_csv(args.cmu_manifest)
    source_cycles, feature_min, feature_max = load_temperature_cycles_and_range(
        args.panasonic_data_root, 25
    )
    tensor_layout = "paper" if args.source_kind == "trained" else "released_repo"

    base_expected = {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "manifest_sha256": protocol_record["manifest_sha256"],
        "source_kind": args.source_kind,
        "scenario": args.scenario,
        "base_seed": args.seed,
        "source_epochs": args.source_epochs if args.source_kind == "trained" else None,
        "transfer_epochs": args.transfer_epochs,
        "batch_size": args.batch_size,
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
    # Early, in-progress metadata so an interrupted run is self-describing.
    _write_json_atomic(
        {
            "output_schema_version": OUTPUT_SCHEMA_VERSION, "status": "in_progress",
            "benchmark": "shen_2023_tatn_panasonic_25c_to_cmu_core7",
            "protocol_verification": protocol_record, "cumulative_wall_clock_seconds":
            float((previous or {}).get("cumulative_wall_clock_seconds") or 0.0),
            **{k: v for k, v in base_expected.items()},
        },
        args.output_root / "run_metadata.json",
    )

    cell_cache: dict[str, dict] = {}
    source_infos: list[dict] = []
    parameter_counts: dict[str, int] = {}
    for repeat in range(1, args.repeats + 1):
        source_model, source_info = _obtain_source(args, repeat, device, source_cycles, args.output_root)
        source_infos.append(source_info)
        source_fingerprint = _state_fingerprint(source_model)
        if not parameter_counts:
            parameter_counts = _parameter_counts(source_model)
        for cell_id in args.cells:
            cell_output = _repeat_dir(args.output_root, repeat) / cell_id
            result_path = cell_output / "results.json"
            seed = cell_seed(args.seed, repeat, cell_id)
            input_sha256 = protocol_record["cmu_artifact_sha256"][cell_id]
            expected = {
                **base_expected, "repeat": repeat, "seed": seed,
                "source_checkpoint_sha256": source_info["sha256"],
                "cmu_input_sha256": input_sha256,
            }
            if args.resume and result_path.is_file():
                _check_resume_tatn_result(result_path=result_path, cell_output=cell_output, expected=expected)
                print(f"repeat {repeat} {cell_id}: resumed (verified)")
                continue
            if cell_id not in cell_cache:
                cell_cache[cell_id] = _load_cell_data(
                    args, cell_id, cell_artifacts[cell_id], manifest, feature_min, feature_max
                )
            result = _run_cell(
                args, repeat=repeat, cell_id=cell_id, cell_index_seed=seed,
                data=cell_cache[cell_id], source_model=source_model, source_cycles=source_cycles,
                device=device, tensor_layout=tensor_layout, cell_output=cell_output,
                resume_expected={k: v for k, v in expected.items() if k != "cmu_input_sha256"},
                input_sha256=input_sha256, source_info=source_info, feature_min=feature_min,
                feature_max=feature_max, source_fingerprint=source_fingerprint,
            )
            so, ta = result["metrics"]
            print(
                f"repeat {repeat} {cell_id}: Source only MAE={so['overall_mae_percent']:.3f}% "
                f"TATN MAE={ta['overall_mae_percent']:.3f}% "
                f"pseudo_windows={result['target_info']['pseudo_windows']}"
            )
        del source_model

    aggregate = _aggregate(args.output_root, args.repeats, list(args.cells))
    metadata = _build_run_metadata(
        args, status="completed", protocol_record=protocol_record, source_infos=source_infos,
        device=device, tensor_layout=tensor_layout, feature_min=feature_min,
        feature_max=feature_max, aggregate=aggregate, parameter_counts=parameter_counts,
        invocation_seconds=time.perf_counter() - started, previous=previous,
        cell_artifacts=cell_artifacts,
    )
    _write_json_atomic(metadata, args.output_root / "run_metadata.json")
    if "cell_metrics" in aggregate:
        print(aggregate["cell_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
