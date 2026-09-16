#!/usr/bin/env python3
"""Regenerate CMU dynamic-capacity SOC outputs with Ns-exact mission phase labels.

This is a deliberate, documented variant of
``build_dynamic_capacity_soc_development_v1.py`` for the Ns-based phase
relabeling task. It differs from that frozen script in exactly one way:

The frozen script hash-checks ``capacity_lookup.csv`` against a value baked
into the protocol YAML and refuses to run if it doesn't match byte-for-byte.
That lookup CSV was built from the OLD power/duration-heuristic mission
manifest (``high_1``/``lower``/``high_2``). Switching mission-candidate
detection to exact Ns-sequence matching finds a strict superset of missions
per cell (verified: the old candidate set is always a subset of the new one,
by 11-47 additional missions per cell, 7/7 cells checked) -- so the frozen
lookup CSV is missing capacity entries for those newly-found missions and
can no longer be used byte-for-byte.

This script instead REGENERATES the capacity lookup by calling
``capacity_lookup.build_causal_capacity_lookup`` with the NEW Ns-based
mission manifest against the SAME frozen, hash-checked RPT observations
CSV (the actual RPT *detection* step -- the expensive/fragile one -- is not
re-run; only the RPT-to-mission join is redone, which is a small, pure,
already-tested function). The regenerated lookup is validated with
``capacity_lookup.validate_lookup`` before use, and every other frozen input
(the fixed SOC baseline implementation and the RPT observations file) is
still hash-checked exactly as in the original script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.datasets.cmu_evtol.capacity_lookup import (  # noqa: E402
    build_causal_capacity_lookup,
    validate_lookup,
)
from battery_benchmark.datasets.cmu_evtol.development_pipeline_v0 import (  # noqa: E402
    contiguous_ns_runs,
    enumerate_diagnostic_candidates,
    load_diagnostic_protocol,
)
from battery_benchmark.datasets.cmu_evtol.dynamic_soc import (  # noqa: E402
    DYNAMIC_SOC_PROTOCOL_VERSION,
    build_dynamic_soc_rows,
    combine_qc,
    qc_has_errors,
    validate_dynamic_soc,
)
from battery_benchmark.datasets.cmu_evtol.ns_phase_labeling_v1 import (  # noqa: E402
    label_full_timeline_phase_role,
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


def build_outputs(
    raw_root: Path,
    output_root: Path,
    protocol_path: Path,
    mission_protocol_path: Path,
    observations_path: Path,
    asset_inventory_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    mission_protocol, mission_protocol_sha = load_diagnostic_protocol(mission_protocol_path)

    fixed_path = repo_root / protocol["fixed_soc_baseline"]["implementation"]
    if sha256_file(fixed_path) != protocol["fixed_soc_baseline"]["implementation_sha256"]:
        raise ValueError("Frozen fixed SOC implementation SHA-256 mismatch")
    if sha256_file(observations_path) != protocol["rpt_observations"]["sha256"]:
        raise ValueError("Frozen RPT observations SHA-256 mismatch")

    observations = pd.read_csv(observations_path)
    if len(observations) != protocol["rpt_observations"]["expected_count"]:
        raise ValueError("Frozen RPT observation count mismatch")
    inventory = pd.read_csv(asset_inventory_path).set_index("cell_id")
    output_root.mkdir(parents=True, exist_ok=True)

    summaries, cell_qc, raw_inputs, manifest_deltas = [], {}, {}, {}
    for cell_id in protocol["cells"]:
        result = load_cmu_v3_main_file(raw_root / f"{cell_id}.csv", repo_root=repo_root)
        if result.metadata["source_sha256"] != inventory.at[cell_id, "sha256"]:
            raise ValueError(f"Asset inventory SHA mismatch for {cell_id}")
        canonical = result.data

        manifest = enumerate_diagnostic_candidates(contiguous_ns_runs(canonical), mission_protocol)

        # Regenerate the causal capacity lookup for THIS cell's new manifest
        # against the frozen, unchanged RPT observations. RPT *detection* is
        # not re-run; only the mission<->RPT join is redone.
        cell_observations = observations.loc[observations["cell_id"].eq(cell_id)]
        cell_lookup = build_causal_capacity_lookup(manifest, cell_observations)
        lookup_qc = validate_lookup(cell_lookup, manifest, cell_observations)
        if any(lookup_qc.values()):
            raise ValueError(f"Regenerated capacity lookup QC failed for {cell_id}: {lookup_qc}")

        fixed, dynamic, comparison, join_qc = build_dynamic_soc_rows(canonical, manifest, cell_lookup)
        validator_qc = validate_dynamic_soc(fixed, dynamic, comparison, cell_lookup, manifest, cell_observations)
        qc = combine_qc(join_qc, validator_qc)
        qc["lookup_qc"] = lookup_qc
        if qc_has_errors(qc):
            raise ValueError(f"Dynamic SOC QC failed for {cell_id}: {qc}")

        # Purely additive, diagnostic-only, full-raw-timeline Ns phase labels.
        # Never fed back into the manifest/lookup/dynamic-SOC steps above.
        full_timeline = canonical.copy()
        full_timeline["ns_mission_phase_role"] = label_full_timeline_phase_role(canonical)

        dynamic.to_parquet(output_root / f"{cell_id}_dynamic_soc.parquet", index=False)
        comparison.to_parquet(output_root / f"{cell_id}_fixed_vs_dynamic.parquet", index=False)
        cell_lookup.to_csv(output_root / f"{cell_id}_capacity_lookup.csv", index=False)
        manifest.to_csv(output_root / f"{cell_id}_candidate_manifest.csv", index=False)
        full_timeline.loc[:, [
            "cell_id", "source_row", "time_s", "Ns", "ns_mission_phase_role",
        ]].to_parquet(output_root / f"{cell_id}_full_timeline_ns_phase_labels.parquet", index=False)

        summary = {
            "cell_id": cell_id,
            "mission_candidate_count": int(len(cell_lookup)),
            "row_count": len(comparison),
            "unique_capacity_count": int(comparison["capacity_ah"].nunique()),
            "max_abs_soc_difference": float(comparison["abs_soc_difference"].max()),
            "mean_abs_soc_difference": float(comparison["abs_soc_difference"].mean()),
            "observed_rpt_candidate_count": int(cell_lookup["capacity_source"].eq("observed_rpt").sum()),
            "nominal_fallback_candidate_count": int(cell_lookup["capacity_source"].eq("nominal_fallback").sum()),
        }
        summaries.append(summary)
        cell_qc[cell_id] = qc
        raw_inputs[cell_id] = {"path": str(raw_root / f"{cell_id}.csv"), "sha256": result.metadata["source_sha256"]}
        del result, fixed, dynamic, full_timeline

    pd.DataFrame(summaries).to_csv(output_root / "cell_summary.csv", index=False)
    qc_summary = {"cells": cell_qc}
    (output_root / "qc_summary.json").write_text(json.dumps(qc_summary, indent=2), encoding="utf-8")
    metadata = {
        "dataset_name": DATASET_NAME,
        "article_id": ARTICLE_ID,
        "dataset_version": DATASET_VERSION,
        "raw_loader_schema_version": LOADER_SCHEMA_VERSION,
        "dynamic_soc_protocol_version": DYNAMIC_SOC_PROTOCOL_VERSION,
        "protocol_path": str(protocol_path),
        "mission_protocol_path": str(mission_protocol_path),
        "mission_protocol_sha256": mission_protocol_sha,
        "mission_candidate_detection": "ns_exact_v1 (take_off/cruise/landing via Ns 4/5/6)",
        "capacity_lookup_regenerated": True,
        "capacity_lookup_regeneration_reason": (
            "Ns-exact mission detection finds a strict superset of the old "
            "power/duration-heuristic missions (11-47 additional missions "
            "per cell, 0 missions lost, across all 7 cells); the previously "
            "frozen capacity_lookup.csv could not cover the new missions, "
            "so it was regenerated via build_causal_capacity_lookup against "
            "the same frozen, unchanged RPT observations."
        ),
        "rpt_observations_path": str(observations_path),
        "rpt_observations_sha256": sha256_file(observations_path),
        "rpt_detection_rerun": False,
        "raw_inputs": raw_inputs,
        "generator": "scripts/datasets/cmu_evtol/build_dynamic_capacity_soc_development_v1_ns_relabel.py",
        "git_revision": get_full_commit_hash(repo_root),
        "git_dirty": is_dirty(repo_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_cells": protocol["cells"],
        "formal_mission_labels": False,
        "formal_benchmark_eligible": False,
        "label_is_ground_truth": False,
    }
    (output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"summary": summaries, "qc": qc_summary, "metadata": metadata}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mission-protocol", type=Path, required=True)
    parser.add_argument("--rpt-observations", type=Path, required=True)
    parser.add_argument("--asset-inventory", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    result = build_outputs(
        args.raw_root, args.output_root, args.protocol, args.mission_protocol,
        args.rpt_observations, args.asset_inventory,
    )
    total_rows = sum(s["row_count"] for s in result["summary"])
    total_missions = sum(s["mission_candidate_count"] for s in result["summary"])
    print(f"Built {total_missions} missions, {total_rows} rows in {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
