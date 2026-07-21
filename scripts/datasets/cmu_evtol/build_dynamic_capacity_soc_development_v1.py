#!/usr/bin/env python3
"""Build development-only CMU dynamic-capacity SOC labels and QC."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.datasets.cmu_evtol.development_pipeline_v0 import (  # noqa: E402
    contiguous_ns_runs,
    enumerate_diagnostic_candidates,
    load_diagnostic_protocol,
)
from battery_benchmark.datasets.cmu_evtol.dynamic_soc import (  # noqa: E402
    DYNAMIC_SOC_PROTOCOL_VERSION,
    build_dynamic_soc_rows,
    validate_dynamic_soc,
)
from battery_benchmark.datasets.cmu_evtol.raw import (  # noqa: E402
    ARTICLE_ID,
    DATASET_NAME,
    DATASET_VERSION,
    LOADER_SCHEMA_VERSION,
    load_cmu_v3_main_file,
    sha256_file,
)
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty  # noqa: E402


def _plot_lifetime(path: Path, cell_id: str, comparison: pd.DataFrame) -> None:
    grouped = comparison.groupby("mission_cycle_id", sort=False).agg(
        capacity_ah=("capacity_ah", "first"),
        max_abs_soc_difference=("abs_soc_difference", "max"),
    ).reset_index(drop=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].step(grouped.index, grouped["capacity_ah"], where="post")
    axes[0].set_ylabel("Capacity (Ah)"); axes[0].grid(alpha=0.25)
    axes[1].plot(grouped.index, grouped["max_abs_soc_difference"])
    axes[1].set(xlabel="Development candidate order", ylabel="Max |dynamic - fixed| SOC")
    axes[1].grid(alpha=0.25); fig.suptitle(f"{cell_id} dynamic-capacity SOC lifetime QC")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_representatives(path: Path, cell_id: str, comparison: pd.DataFrame) -> None:
    candidate_ids = comparison["mission_cycle_id"].drop_duplicates().tolist()
    positions = sorted({0, len(candidate_ids) // 2, len(candidate_ids) - 1})
    fig, axes = plt.subplots(len(positions), 1, figsize=(9, 3 * len(positions)), squeeze=False)
    for ax, position in zip(axes[:, 0], positions):
        candidate_id = candidate_ids[position]
        rows = comparison.loc[comparison["mission_cycle_id"].eq(candidate_id)]
        x = np.arange(len(rows))
        ax.plot(x, rows["soc_fixed_3ah"], label="fixed 3.0 Ah")
        ax.plot(x, rows["soc_dynamic"], label="dynamic")
        ax.set(title=f"candidate {position + 1}/{len(candidate_ids)}; capacity={rows['capacity_ah'].iloc[0]:.4f} Ah", ylabel="SOC")
        ax.grid(alpha=0.25); ax.legend(loc="best")
    axes[-1, 0].set_xlabel("Row within candidate")
    fig.suptitle(f"{cell_id} early/middle/late SOC comparison")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def build_outputs(
    raw_root: Path,
    output_root: Path,
    protocol_path: Path,
    mission_protocol_path: Path,
    lookup_path: Path,
    observations_path: Path,
    asset_inventory_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    mission_protocol, mission_protocol_sha = load_diagnostic_protocol(mission_protocol_path)
    fixed_path = repo_root / protocol["fixed_soc_baseline"]["implementation"]
    for path, expected, label in (
        (fixed_path, protocol["fixed_soc_baseline"]["implementation_sha256"], "fixed SOC implementation"),
        (lookup_path, protocol["capacity_lookup"]["sha256"], "capacity lookup"),
        (observations_path, protocol["rpt_observations"]["sha256"], "RPT observations"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen {label} SHA-256 mismatch")

    lookup = pd.read_csv(lookup_path)
    observations = pd.read_csv(observations_path)
    if len(observations) != protocol["rpt_observations"]["expected_count"]:
        raise ValueError("Frozen RPT observation count mismatch")
    inventory = pd.read_csv(asset_inventory_path).set_index("cell_id")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(exist_ok=True)

    summaries, all_comparisons, cell_qc, raw_inputs = [], [], {}, {}
    for cell_id in protocol["cells"]:
        result = load_cmu_v3_main_file(raw_root / f"{cell_id}.csv", repo_root=repo_root)
        if result.metadata["source_sha256"] != inventory.at[cell_id, "sha256"]:
            raise ValueError(f"Asset inventory SHA mismatch for {cell_id}")
        manifest = enumerate_diagnostic_candidates(contiguous_ns_runs(result.data), mission_protocol)
        cell_lookup = lookup.loc[lookup["cell_id"].eq(cell_id)].copy()
        dynamic, comparison, join_qc = build_dynamic_soc_rows(result.data, manifest, cell_lookup)
        qc = validate_dynamic_soc(dynamic, comparison, cell_lookup, manifest, observations)
        qc.update(join_qc)
        if any(qc.values()):
            raise ValueError(f"Dynamic SOC QC failed for {cell_id}: {qc}")

        dynamic.to_parquet(output_root / f"{cell_id}_dynamic_soc.parquet", index=False)
        comparison.to_parquet(output_root / f"{cell_id}_fixed_vs_dynamic.parquet", index=False)
        _plot_lifetime(output_root / "figures" / f"{cell_id}_lifetime_qc.png", cell_id, comparison)
        _plot_representatives(
            output_root / "figures" / f"{cell_id}_representative_soc.png", cell_id, comparison
        )
        endpoints = comparison.groupby("mission_cycle_id", sort=False).tail(1)
        summary = {
            "cell_id": cell_id,
            "candidate_count": int(comparison["mission_cycle_id"].nunique()),
            "row_count": len(comparison),
            "unique_capacity_count": int(comparison["capacity_ah"].nunique()),
            "capacity_min_ah": float(comparison["capacity_ah"].min()),
            "capacity_max_ah": float(comparison["capacity_ah"].max()),
            "mean_capacity_ah": float(comparison.groupby("mission_cycle_id")["capacity_ah"].first().mean()),
            "max_abs_soc_difference": float(comparison["abs_soc_difference"].max()),
            "mean_abs_soc_difference": float(comparison["abs_soc_difference"].mean()),
            "endpoint_soc_difference_min": float(endpoints["soc_difference"].min()),
            "endpoint_soc_difference_mean": float(endpoints["soc_difference"].mean()),
            "endpoint_soc_difference_max": float(endpoints["soc_difference"].max()),
            "fallback_candidate_count": int(cell_lookup["capacity_source"].eq("nominal_fallback").sum()),
            "observed_rpt_candidate_count": int(cell_lookup["capacity_source"].eq("observed_rpt").sum()),
        }
        summaries.append(summary); all_comparisons.append(comparison); cell_qc[cell_id] = qc
        raw_inputs[cell_id] = {
            "path": str(raw_root / f"{cell_id}.csv"),
            "sha256": result.metadata["source_sha256"],
        }
        del result, dynamic

    combined = pd.concat(all_comparisons, ignore_index=True)
    combined.to_parquet(output_root / "fixed_vs_dynamic_comparison.parquet", index=False)
    pd.DataFrame(summaries).to_csv(output_root / "cell_summary.csv", index=False)
    qc_summary = {
        "row_count": len(combined),
        "candidate_count": int(combined[["cell_id", "mission_cycle_id"]].drop_duplicates().shape[0]),
        "cells": cell_qc,
        "aggregate_counts": {
            key: sum(qc[key] for qc in cell_qc.values()) for key in next(iter(cell_qc.values()))
        },
    }
    (output_root / "qc_summary.json").write_text(json.dumps(qc_summary, indent=2), encoding="utf-8")
    metadata = {
        "dataset_name": DATASET_NAME,
        "article_id": ARTICLE_ID,
        "dataset_version": DATASET_VERSION,
        "raw_loader_schema_version": LOADER_SCHEMA_VERSION,
        "dynamic_soc_protocol_version": DYNAMIC_SOC_PROTOCOL_VERSION,
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "fixed_soc_baseline_version": protocol["fixed_soc_baseline"]["version"],
        "fixed_soc_baseline_path": str(fixed_path),
        "fixed_soc_baseline_sha256": sha256_file(fixed_path),
        "capacity_lookup_version": protocol["capacity_lookup"]["version"],
        "capacity_lookup_path": str(lookup_path),
        "capacity_lookup_sha256": sha256_file(lookup_path),
        "rpt_observations_path": str(observations_path),
        "rpt_observations_sha256": sha256_file(observations_path),
        "mission_protocol_path": str(mission_protocol_path),
        "mission_protocol_sha256": mission_protocol_sha,
        "raw_inputs": raw_inputs,
        "generator": "scripts/datasets/cmu_evtol/build_dynamic_capacity_soc_development_v1.py",
        "git_revision": get_full_commit_hash(repo_root),
        "git_dirty": is_dirty(repo_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_cells": protocol["cells"],
        "interpolation_used": False,
        "future_information_used": False,
        "within_candidate_capacity_change": False,
        "fixed_soc_outputs_modified": False,
        "formal_mission_labels": False,
        "mission_identity_source": "development_candidate_id",
        "soc_clipping_applied": False,
    }
    (output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"summary": summaries, "qc": qc_summary, "metadata": metadata}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/datasets/cmu_evtol/protocols/cmu_evtol_dynamic_capacity_soc_development_v1.yaml"))
    parser.add_argument("--mission-protocol", type=Path, default=Path("configs/datasets/cmu_evtol/protocols/cmu_evtol_mission_segmentation_v0_1.yaml"))
    parser.add_argument("--capacity-lookup", type=Path, default=Path("outputs/datasets/cmu_evtol/causal_capacity_lookup_v1/capacity_lookup.csv"))
    parser.add_argument("--rpt-observations", type=Path, default=Path("outputs/cmu_evtol_rpt_capacity_v1/capacity_observations.csv"))
    parser.add_argument("--asset-inventory", type=Path, default=Path("data/manifests/cmu_evtol/cmu_evtol_vtc6_asset_inventory_v3.csv"))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(); started = time.perf_counter()
    result = build_outputs(
        args.raw_root, args.output_root, args.protocol, args.mission_protocol,
        args.capacity_lookup, args.rpt_observations, args.asset_inventory,
    )
    print(
        f"Built {result['qc']['candidate_count']} dynamic SOC candidates and "
        f"{result['qc']['row_count']} rows in {time.perf_counter() - started:.2f} s"
    )


if __name__ == "__main__":
    main()
