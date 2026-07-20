#!/usr/bin/env python3
"""Audit CMU Version 3 evidence relevant to future mission reconstruction.

This script emits descriptive evidence only. It does not assign mission IDs,
phase names, RPT labels, capacity references, splits, or model sequences.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.data.cmu_evtol_raw import (  # noqa: E402
    ARTICLE_ID,
    CMU_V3_CELL_IDS,
    DATASET_VERSION,
    LOADER_SCHEMA_VERSION,
    load_cmu_v3_main_file,
)

AUDIT_CELLS = ("VAH01", "VAH17", "VAH27", "VAH02", "VAH12", "VAH05")
GAP_CONTEXT_CELLS = ("VAH09", "VAH10", "VAH27", "VAH30")
ALL_INPUT_CELLS = tuple(dict.fromkeys((*AUDIT_CELLS, *GAP_CONTEXT_CELLS)))
GAP_THRESHOLD_S = 120.0
POWER_FORMULA = "electrical_power_W = Ecell_V * I_mA / 1000"

# Descriptive audit filters only; none is a production segmentation rule.
CANDIDATE_THRESHOLDS = {
    "high_discharge_power_W": [35.0, 70.0],
    "lower_discharge_power_W": [5.0, 30.0],
    "first_high_duration_s": [45.0, 120.0],
    "lower_duration_s": [200.0, 1500.0],
    "last_high_duration_s": [60.0, 150.0],
    "minimum_discharge_fraction": 0.9,
    "rpt_like_discharge_power_W": [0.5, 4.0],
    "rpt_like_minimum_duration_s": 3600.0,
    "rpt_like_minimum_voltage_drop_V": 0.8,
    "rpt_like_maximum_end_voltage_V": 2.7,
}

EVIDENCE_COLUMNS = (
    "article_id",
    "dataset_version",
    "loader_schema_version",
    "generated_at_utc",
    "git_revision",
    "git_dirty",
    "gap_threshold_s",
    "input_sha256",
    "power_formula",
    "candidate_thresholds_json",
    "evidence_type",
    "cell_id",
    "ns_value",
    "row_count",
    "contiguous_run_count",
    "charge_fraction",
    "discharge_fraction",
    "zero_current_fraction",
    "run_duration_min_s",
    "run_duration_median_s",
    "run_duration_max_s",
    "electrical_power_median_W",
    "electrical_power_p10_W",
    "electrical_power_p90_W",
    "common_previous_ns",
    "common_next_ns",
    "candidate_count",
    "representative_source_row_ranges",
    "observed_ns_sequences",
    "first_segment_duration_median_s",
    "middle_segment_duration_median_s",
    "last_segment_duration_median_s",
    "first_segment_discharge_power_median_W",
    "middle_segment_discharge_power_median_W",
    "last_segment_discharge_power_median_W",
    "start_voltage_median_V",
    "end_voltage_median_V",
    "voltage_drop_median_V",
    "impedance_cycle_match_count",
    "impedance_cycle_candidate_count",
    "previous_source_row",
    "current_source_row",
    "previous_time_s",
    "current_time_s",
    "delta_s",
    "previous_cycle_number",
    "current_cycle_number",
    "previous_ns",
    "current_ns",
    "previous_current_mA",
    "current_current_mA",
    "previous_voltage_V",
    "current_voltage_V",
    "previous_electrical_power_W",
    "current_electrical_power_W",
    "duplicate_boundary_count",
    "duplicate_representative_source_rows",
    "assessment",
)


def add_power_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with explicitly signed electrical/discharge power."""
    result = data.copy()
    result["electrical_power_W"] = result["Ecell_V"] * result["I_mA"] / 1000.0
    result["discharge_power_W"] = np.maximum(-result["electrical_power_W"], 0.0)
    return result


def contiguous_ns_runs(data: pd.DataFrame) -> pd.DataFrame:
    """Describe contiguous Ns runs without crossing canonical block IDs."""
    required = {
        "cell_id",
        "continuous_block_id",
        "source_row",
        "time_s",
        "Ecell_V",
        "I_mA",
        "cycleNumber",
        "Ns",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Ns run evidence missing canonical column(s): {missing}")
    frame = add_power_columns(data)
    starts = frame["continuous_block_id"].ne(frame["continuous_block_id"].shift())
    starts |= frame["Ns"].ne(frame["Ns"].shift())
    run_key = starts.cumsum()
    records: list[dict[str, object]] = []
    for _, group in frame.groupby(run_key, sort=False):
        start_time = float(group["time_s"].iloc[0])
        end_time = float(group["time_s"].iloc[-1])
        duration = (
            end_time - start_time
            if np.isfinite(start_time) and np.isfinite(end_time)
            else np.nan
        )
        current = group["I_mA"]
        records.append(
            {
                "cell_id": group["cell_id"].iloc[0],
                "continuous_block_id": int(group["continuous_block_id"].iloc[0]),
                "ns_value": group["Ns"].iloc[0],
                "source_row_start": int(group["source_row"].iloc[0]),
                "source_row_end": int(group["source_row"].iloc[-1]),
                "time_start_s": start_time,
                "time_end_s": end_time,
                "duration_s": duration,
                "row_count": len(group),
                "charge_fraction": float((current > 0).mean()),
                "discharge_fraction": float((current < 0).mean()),
                "zero_current_fraction": float((current == 0).mean()),
                "electrical_power_median_W": float(
                    group["electrical_power_W"].median()
                ),
                "electrical_power_p10_W": float(
                    group["electrical_power_W"].quantile(0.10)
                ),
                "electrical_power_p90_W": float(
                    group["electrical_power_W"].quantile(0.90)
                ),
                "discharge_power_median_W": float(
                    group["discharge_power_W"].median()
                ),
                "start_voltage_V": float(group["Ecell_V"].iloc[0]),
                "end_voltage_V": float(group["Ecell_V"].iloc[-1]),
                "cycle_number_start": group["cycleNumber"].iloc[0],
                "cycle_number_end": group["cycleNumber"].iloc[-1],
            }
        )
    runs = pd.DataFrame(records)
    runs["previous_ns"] = np.nan
    runs["next_ns"] = np.nan
    for _, indices in runs.groupby("continuous_block_id", sort=False).groups.items():
        ordered = list(indices)
        runs.loc[ordered, "previous_ns"] = runs.loc[ordered, "ns_value"].shift().to_numpy()
        runs.loc[ordered, "next_ns"] = runs.loc[ordered, "ns_value"].shift(-1).to_numpy()
    return runs


def _common_values(values: Iterable[object], limit: int = 3) -> str:
    finite = [value for value in values if pd.notna(value)]
    counts = Counter(finite)
    return json.dumps(
        [{"value": value, "count": count} for value, count in counts.most_common(limit)],
        separators=(",", ":"),
    )


def ns_evidence(data: pd.DataFrame, runs: pd.DataFrame) -> list[dict[str, object]]:
    frame = add_power_columns(data)
    records: list[dict[str, object]] = []
    for ns_value, rows in frame.groupby("Ns", sort=True, dropna=False):
        ns_runs = runs.loc[runs["ns_value"].eq(ns_value)]
        durations = ns_runs["duration_s"].dropna()
        current = rows["I_mA"]
        records.append(
            {
                "evidence_type": "ns_summary",
                "cell_id": rows["cell_id"].iloc[0],
                "ns_value": ns_value,
                "row_count": len(rows),
                "contiguous_run_count": len(ns_runs),
                "charge_fraction": float((current > 0).mean()),
                "discharge_fraction": float((current < 0).mean()),
                "zero_current_fraction": float((current == 0).mean()),
                "run_duration_min_s": float(durations.min()),
                "run_duration_median_s": float(durations.median()),
                "run_duration_max_s": float(durations.max()),
                "electrical_power_median_W": float(
                    rows["electrical_power_W"].median()
                ),
                "electrical_power_p10_W": float(
                    rows["electrical_power_W"].quantile(0.10)
                ),
                "electrical_power_p90_W": float(
                    rows["electrical_power_W"].quantile(0.90)
                ),
                "common_previous_ns": _common_values(ns_runs["previous_ns"]),
                "common_next_ns": _common_values(ns_runs["next_ns"]),
                "assessment": "descriptive_Ns_evidence_only",
            }
        )
    return records


def _between(value: float, bounds: Sequence[float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def mission_waveform_evidence(runs: pd.DataFrame) -> dict[str, object]:
    """Find descriptive high/lower/high triples without assigning missions."""
    candidates: list[tuple[pd.Series, pd.Series, pd.Series]] = []
    for index in range(len(runs) - 2):
        first, middle, last = runs.iloc[index : index + 3].itertuples(index=False)
        if not (
            first.continuous_block_id
            == middle.continuous_block_id
            == last.continuous_block_id
        ):
            continue
        if min(
            first.discharge_fraction,
            middle.discharge_fraction,
            last.discharge_fraction,
        ) < CANDIDATE_THRESHOLDS["minimum_discharge_fraction"]:
            continue
        checks = (
            _between(
                first.discharge_power_median_W,
                CANDIDATE_THRESHOLDS["high_discharge_power_W"],
            ),
            _between(
                middle.discharge_power_median_W,
                CANDIDATE_THRESHOLDS["lower_discharge_power_W"],
            ),
            _between(
                last.discharge_power_median_W,
                CANDIDATE_THRESHOLDS["high_discharge_power_W"],
            ),
            _between(
                first.duration_s, CANDIDATE_THRESHOLDS["first_high_duration_s"]
            ),
            _between(middle.duration_s, CANDIDATE_THRESHOLDS["lower_duration_s"]),
            _between(last.duration_s, CANDIDATE_THRESHOLDS["last_high_duration_s"]),
        )
        if all(checks):
            candidates.append((first, middle, last))

    def med(attribute: str, position: int) -> float:
        return float(np.median([getattr(item[position], attribute) for item in candidates]))

    representative = [
        f"{item[0].source_row_start}-{item[2].source_row_end}"
        for item in candidates[:3]
    ]
    sequences = [
        [item[0].ns_value, item[1].ns_value, item[2].ns_value]
        for item in candidates[:3]
    ]
    record: dict[str, object] = {
        "evidence_type": "candidate_high_lower_high_summary",
        "cell_id": runs["cell_id"].iloc[0],
        "candidate_count": len(candidates),
        "representative_source_row_ranges": json.dumps(representative),
        "observed_ns_sequences": json.dumps(sequences),
        "assessment": (
            "repeated_candidate_waveform_not_a_mission_or_phase_assignment"
            if candidates
            else "insufficient_or_uncertain"
        ),
    }
    if candidates:
        record.update(
            {
                "first_segment_duration_median_s": med("duration_s", 0),
                "middle_segment_duration_median_s": med("duration_s", 1),
                "last_segment_duration_median_s": med("duration_s", 2),
                "first_segment_discharge_power_median_W": med(
                    "discharge_power_median_W", 0
                ),
                "middle_segment_discharge_power_median_W": med(
                    "discharge_power_median_W", 1
                ),
                "last_segment_discharge_power_median_W": med(
                    "discharge_power_median_W", 2
                ),
            }
        )
    return record


def rpt_like_evidence(runs: pd.DataFrame, impedance_path: Path) -> dict[str, object]:
    power_bounds = CANDIDATE_THRESHOLDS["rpt_like_discharge_power_W"]
    voltage_drop = runs["start_voltage_V"] - runs["end_voltage_V"]
    mask = (
        runs["discharge_power_median_W"].between(*power_bounds)
        & runs["duration_s"].ge(
            CANDIDATE_THRESHOLDS["rpt_like_minimum_duration_s"]
        )
        & voltage_drop.ge(CANDIDATE_THRESHOLDS["rpt_like_minimum_voltage_drop_V"])
        & runs["end_voltage_V"].le(
            CANDIDATE_THRESHOLDS["rpt_like_maximum_end_voltage_V"]
        )
        & runs["discharge_fraction"].ge(
            CANDIDATE_THRESHOLDS["minimum_discharge_fraction"]
        )
    )
    candidates = runs.loc[mask].copy()
    candidate_cycles = set(candidates["cycle_number_start"].dropna().astype(int))
    impedance_cycles: set[int] = set()
    if impedance_path.is_file():
        impedance = pd.read_csv(impedance_path)
        impedance_cycles = set(impedance["cycle numbers"].dropna().astype(int))
    matches = candidate_cycles & impedance_cycles
    record: dict[str, object] = {
        "evidence_type": "candidate_long_low_rate_discharge_summary",
        "cell_id": runs["cell_id"].iloc[0],
        "candidate_count": len(candidates),
        "representative_source_row_ranges": json.dumps(
            [
                f"{row.source_row_start}-{row.source_row_end}"
                for row in candidates.head(3).itertuples()
            ]
        ),
        "impedance_cycle_match_count": len(matches),
        "impedance_cycle_candidate_count": len(candidate_cycles),
        "assessment": (
            "rpt_like_candidate_with_possible_impedance_cycle_correspondence"
            if len(candidates)
            else "insufficient_or_uncertain"
        ),
    }
    if len(candidates):
        record.update(
            {
                "run_duration_median_s": float(candidates["duration_s"].median()),
                "electrical_power_median_W": float(
                    candidates["electrical_power_median_W"].median()
                ),
                "start_voltage_median_V": float(
                    candidates["start_voltage_V"].median()
                ),
                "end_voltage_median_V": float(candidates["end_voltage_V"].median()),
                "voltage_drop_median_V": float(
                    (
                        candidates["start_voltage_V"]
                        - candidates["end_voltage_V"]
                    ).median()
                ),
            }
        )
    return record


def discontinuity_evidence(data: pd.DataFrame) -> list[dict[str, object]]:
    frame = add_power_columns(data).reset_index(drop=True)
    records: list[dict[str, object]] = []
    gap_indices = frame.index[
        frame["continuous_block_boundary_reason"].str.contains(
            "gap_exceeded", na=False
        )
    ]
    for index in gap_indices:
        if index == 0:
            continue
        previous = frame.iloc[index - 1]
        current = frame.iloc[index]
        records.append(
            {
                "evidence_type": "gap_context",
                "cell_id": current["cell_id"],
                "previous_source_row": int(previous["source_row"]),
                "current_source_row": int(current["source_row"]),
                "previous_time_s": previous["time_s"],
                "current_time_s": current["time_s"],
                "delta_s": current["time_s"] - previous["time_s"],
                "previous_cycle_number": previous["cycleNumber"],
                "current_cycle_number": current["cycleNumber"],
                "previous_ns": previous["Ns"],
                "current_ns": current["Ns"],
                "previous_current_mA": previous["I_mA"],
                "current_current_mA": current["I_mA"],
                "previous_voltage_V": previous["Ecell_V"],
                "current_voltage_V": current["Ecell_V"],
                "previous_electrical_power_W": previous["electrical_power_W"],
                "current_electrical_power_W": current["electrical_power_W"],
                "assessment": "canonical_block_boundary_may_interrupt_future_reconstruction",
            }
        )
    duplicate_rows = frame.loc[
        frame["continuous_block_boundary_reason"].str.contains(
            "time_duplicate", na=False
        ),
        "source_row",
    ]
    records.append(
        {
            "evidence_type": "duplicate_boundary_summary",
            "cell_id": frame["cell_id"].iloc[0],
            "duplicate_boundary_count": len(duplicate_rows),
            "duplicate_representative_source_rows": json.dumps(
                duplicate_rows.head(3).astype(int).tolist()
            ),
            "assessment": "count_and_representative_samples_only",
        }
    )
    return records


def _with_metadata(
    records: Iterable[dict[str, object]], metadata: dict[str, object]
) -> list[dict[str, object]]:
    shared = {
        "article_id": ARTICLE_ID,
        "dataset_version": DATASET_VERSION,
        "loader_schema_version": LOADER_SCHEMA_VERSION,
        "generated_at_utc": metadata["generated_at_utc"],
        "git_revision": metadata["git_revision"],
        "git_dirty": metadata["git_dirty"],
        "gap_threshold_s": metadata["gap_threshold_s"],
        "input_sha256": metadata["source_sha256"],
        "power_formula": POWER_FORMULA,
        "candidate_thresholds_json": json.dumps(
            CANDIDATE_THRESHOLDS, separators=(",", ":"), sort_keys=True
        ),
    }
    return [{**shared, **record} for record in records]


def run_evidence_audit(
    raw_root: Path,
    *,
    repo_root: Path = REPO_ROOT,
    audit_cells: Sequence[str] = AUDIT_CELLS,
    gap_cells: Sequence[str] = GAP_CONTEXT_CELLS,
) -> pd.DataFrame:
    """Audit the fixed representative cells sequentially."""
    cells = tuple(dict.fromkeys((*audit_cells, *gap_cells)))
    invalid = sorted(set(cells) - set(CMU_V3_CELL_IDS))
    if invalid:
        raise ValueError(f"Unsupported CMU audit cell(s): {invalid}")
    assets = pd.read_csv(
        repo_root / "data/manifests/cmu_evtol_vtc6_asset_inventory_v3.csv"
    ).set_index("cell_id")
    qc = pd.read_csv(
        repo_root / "data/manifests/cmu_evtol_vtc6_raw_qc_v3_120s.csv"
    ).set_index("cell_id")
    generated_at = datetime.now(timezone.utc).isoformat()
    evidence: list[dict[str, object]] = []

    for cell_id in cells:
        main_path = raw_root / f"{cell_id}.csv"
        result = load_cmu_v3_main_file(
            main_path, gap_threshold_s=GAP_THRESHOLD_S, repo_root=repo_root
        )
        metadata = dict(result.metadata)
        metadata["generated_at_utc"] = generated_at
        if metadata["source_sha256"] != assets.loc[cell_id, "sha256"]:
            raise ValueError(f"Asset SHA mismatch for {cell_id}: {main_path}")
        if metadata["block_count"] != int(qc.loc[cell_id, "block_count"]):
            raise ValueError(f"QC block-count mismatch for {cell_id}: {main_path}")

        cell_records: list[dict[str, object]] = []
        if cell_id in audit_cells:
            runs = contiguous_ns_runs(result.data)
            cell_records.extend(ns_evidence(result.data, runs))
            cell_records.append(mission_waveform_evidence(runs))
            cell_records.append(
                rpt_like_evidence(runs, raw_root / f"{cell_id}_impedance.csv")
            )
        cell_records.extend(discontinuity_evidence(result.data))
        evidence.extend(_with_metadata(cell_records, metadata))
        del result

    return pd.DataFrame(evidence, columns=EVIDENCE_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/manifests/cmu_evtol_mission_phase_evidence_v0_1.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    evidence = run_evidence_audit(args.raw_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(args.output, index=False)
    elapsed = time.perf_counter() - started
    print(
        f"Wrote {len(evidence)} descriptive evidence rows to {args.output} "
        f"in {elapsed:.2f} s"
    )


if __name__ == "__main__":
    main()
