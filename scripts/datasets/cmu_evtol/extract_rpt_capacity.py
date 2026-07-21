#!/usr/bin/env python3
"""Extract quality-controlled RPT capacity observations for CMU baseline cells."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.datasets.cmu_evtol.development_pipeline_v0 import load_diagnostic_protocol  # noqa: E402
from battery_benchmark.datasets.cmu_evtol.raw import (  # noqa: E402
    ARTICLE_ID,
    DATASET_NAME,
    DATASET_VERSION,
    LOADER_SCHEMA_VERSION,
    load_cmu_v3_main_file,
    sha256_file,
)
from battery_benchmark.datasets.cmu_evtol.rpt import (  # noqa: E402
    BASELINE_CELLS,
    detect_rpt_capacity_observations,
    load_rpt_capacity_protocol,
)
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty  # noqa: E402


def _plot_capacity(path: Path, cell_id: str, observations: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        observations["diagnostic_missions_before_rpt"],
        observations["capacity_net_ah"],
        marker="o",
    )
    ax.set(
        title=f"{cell_id} CMU RPT capacity v1",
        xlabel="Diagnostic mission matches before RPT",
        ylabel="Net removed capacity (Ah)",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_review(path: Path, observation: pd.Series, data: pd.DataFrame) -> None:
    rows = data.loc[data["source_row"].between(
        observation.start_source_row, observation.end_source_row
    )]
    elapsed = (rows["time_s"] - rows["time_s"].iloc[0]) / 3600
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(elapsed, rows["Ecell_V"]); axes[0].set_ylabel("Voltage (V)")
    axes[1].plot(elapsed, rows["I_mA"] / 1000); axes[1].set_ylabel("Current (A)")
    axes[1].set_xlabel("Elapsed time (h)")
    fig.suptitle(f"{observation.rpt_id}\n{observation.detector_status}: {observation.primary_reason}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def extract_outputs(
    raw_root: Path,
    output_root: Path,
    protocol_path: Path,
    asset_inventory_path: Path,
    mission_protocol_path: Path,
    *,
    cells: Sequence[str] = BASELINE_CELLS,
    repo_root: Path = REPO_ROOT,
) -> pd.DataFrame:
    cells = tuple(cells)
    if len(cells) != len(set(cells)):
        raise ValueError("RPT capacity v1 rejects duplicate cell IDs")
    invalid = sorted(set(cells) - set(BASELINE_CELLS))
    if invalid:
        raise ValueError(f"RPT capacity v1 supports baseline cells only: {invalid}")

    protocol, protocol_sha = load_rpt_capacity_protocol(protocol_path)
    mission_protocol, mission_protocol_sha = load_diagnostic_protocol(mission_protocol_path)
    inventory = pd.read_csv(asset_inventory_path).set_index("cell_id")
    inventory_sha = sha256_file(asset_inventory_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(exist_ok=True)
    observations, qc_rows, raw_inputs = [], [], {}

    for cell_id in cells:
        result = load_cmu_v3_main_file(raw_root / f"{cell_id}.csv", repo_root=repo_root)
        expected_sha = inventory.at[cell_id, "sha256"]
        actual_sha = result.metadata["source_sha256"]
        if actual_sha != expected_sha:
            raise ValueError(f"Asset inventory SHA mismatch for {cell_id}")
        cell = detect_rpt_capacity_observations(
            result.data,
            protocol,
            mission_protocol=mission_protocol,
            source_sha256=actual_sha,
            asset_inventory_sha256=inventory_sha,
            generated_at_utc=generated_at,
        )
        cell.to_csv(output_root / f"{cell_id}_capacity_observations.csv", index=False)
        _plot_capacity(output_root / "figures" / f"{cell_id}_capacity_vs_mission.png", cell_id, cell)
        for row in cell.loc[cell["detector_status"].ne("accepted")].itertuples(index=False):
            _plot_review(
                output_root / "figures" / f"{cell_id}_rpt_{row.cell_rpt_ordinal}_review.png",
                pd.Series(row._asdict()),
                result.data,
            )
        accepted = cell.loc[cell["detector_status"].eq("accepted")]
        ordered = cell.sort_values("start_source_row")
        overlap_count = int((
            ordered["start_source_row"].iloc[1:].to_numpy()
            <= ordered["end_source_row"].iloc[:-1].to_numpy()
        ).sum()) if len(ordered) > 1 else 0
        qc_rows.append({
            "cell_id": cell_id,
            "observation_count": len(cell),
            "accepted_count": len(accepted),
            "review_required_count": int(cell["detector_status"].eq("review_required").sum()),
            "rejected_count": int(cell["detector_status"].eq("rejected").sum()),
            "capacity_min_ah": float(cell["capacity_net_ah"].min()),
            "capacity_median_ah": float(cell["capacity_net_ah"].median()),
            "capacity_max_ah": float(cell["capacity_net_ah"].max()),
            "first_capacity_ah": float(cell["capacity_net_ah"].iloc[0]),
            "final_capacity_ah": float(cell["capacity_net_ah"].iloc[-1]),
            "final_capacity_retention_fraction": float(cell["capacity_retention_fraction"].iloc[-1]),
            "maximum_rebound_ah": float(max(0.0, cell["capacity_delta_ah"].max())),
            "maximum_abs_net_vs_discharge_only_ah": float(
                (cell["capacity_discharge_only_ah"] - cell["capacity_net_ah"]).abs().max()
            ),
            "maximum_abs_independent_vs_qdischarge_ah": float(
                cell["independent_vs_qdischarge_delta_ah"].abs().max()
            ),
            "non_finite_or_non_positive_capacity_count": int(
                (~cell["capacity_net_ah"].gt(0) | ~cell["capacity_net_ah"].notna()).sum()
            ),
            "initial_capacity_in_expected_range": bool(
                protocol["capacity_qc"]["initial_capacity_expected_range_Ah"][0]
                <= cell["capacity_net_ah"].iloc[0]
                <= protocol["capacity_qc"]["initial_capacity_expected_range_Ah"][1]
            ),
            "duplicate_rpt_id_count": int(cell["rpt_id"].duplicated().sum()),
            "overlapping_observation_count": overlap_count,
            "internal_mission_spacing_warning_count": int(
                cell["warning_codes"].str.contains("mission_spacing_not_50", regex=False).sum()
            ),
            "file_tail_observation_count": int(cell["candidate_ends_at_file_tail"].sum()),
            "cross_block_observation_count": int(
                cell["all_reason_codes"].str.contains("crosses_continuous_block", regex=False).sum()
            ),
        })
        raw_inputs[cell_id] = {
            "source_file": result.metadata["source_file"],
            "source_sha256": actual_sha,
            "impedance_file": result.metadata["impedance_file"],
            "impedance_sha256": result.metadata["impedance_sha256"],
        }
        observations.append(cell)
        del result

    combined = pd.concat(observations, ignore_index=True)
    combined.to_csv(output_root / "capacity_observations.csv", index=False)
    combined.loc[combined["detector_status"].eq("accepted")].to_csv(
        output_root / "accepted_capacity_observations.csv", index=False
    )
    combined.loc[:, [
        "rpt_id", "cell_id", "cell_rpt_ordinal", "detector_status",
        "accepted_for_capacity_lookup", "primary_reason", "all_reason_codes",
        "warning_codes", "continuous_block_id", "start_source_row", "end_source_row",
    ]].to_csv(output_root / "rpt_detection_manifest.csv", index=False)
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(output_root / "capacity_qc_summary.csv", index=False)
    counts = combined["detector_status"].value_counts().to_dict()
    summary = {
        "observation_count": len(combined),
        "status_counts": {key: int(value) for key, value in counts.items()},
        "cells": qc_rows,
    }
    (output_root / "capacity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    metadata = {
        "dataset_name": DATASET_NAME,
        "article_id": ARTICLE_ID,
        "dataset_version": DATASET_VERSION,
        "generator": "scripts/datasets/cmu_evtol/extract_rpt_capacity.py",
        "protocol_path": str(protocol_path),
        "mission_protocol_path": str(mission_protocol_path),
        "raw_loader_schema_version": LOADER_SCHEMA_VERSION,
        "raw_inputs": raw_inputs,
        "asset_inventory_sha256": inventory_sha,
        "protocol_file_sha256": protocol_sha,
        "mission_protocol_file_sha256": mission_protocol_sha,
        "git_revision": get_full_commit_hash(repo_root),
        "git_dirty": is_dirty(repo_root),
        "detector_version": protocol["detector_version"],
        "integration_definition": protocol["integration"],
        "current_sign_convention": "positive_charge_negative_discharge",
        "cycle_number_role": "auxiliary_observation_only_not_identity_or_join_key",
        "impedance_role": "ambiguous_auxiliary_only_no_direct_join",
        "status_counts": summary["status_counts"],
        "generated_at_utc": generated_at,
        "formal_soc_generated": False,
        "capacity_lookup_generated": False,
        "label_is_ground_truth": False,
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return combined


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/datasets/cmu_evtol/rpt_capacity_v1"),
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("configs/datasets/cmu_evtol/protocols/cmu_evtol_rpt_capacity_v1.yaml"),
    )
    parser.add_argument(
        "--asset-inventory", type=Path,
        default=Path("data/manifests/cmu_evtol/cmu_evtol_vtc6_asset_inventory_v3.csv"),
    )
    parser.add_argument(
        "--mission-protocol", type=Path,
        default=Path("configs/datasets/cmu_evtol/protocols/cmu_evtol_mission_segmentation_v0_1.yaml"),
    )
    parser.add_argument("--cells", nargs="+", default=list(BASELINE_CELLS))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    observations = extract_outputs(
        args.raw_root, args.output_root, args.protocol, args.asset_inventory,
        args.mission_protocol, cells=args.cells,
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        f"Wrote {len(observations)} RPT capacity observations in "
        f"{time.perf_counter() - started:.2f} s (peak_rss_native={peak})"
    )


if __name__ == "__main__":
    main()
