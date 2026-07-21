#!/usr/bin/env python3
"""Build the independent CMU causal capacity lookup v1 artifact."""

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
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.datasets.cmu_evtol.capacity_lookup import (  # noqa: E402
    LOOKUP_PROTOCOL_VERSION,
    build_causal_capacity_lookup,
    validate_lookup,
)
from battery_benchmark.datasets.cmu_evtol.development_pipeline_v0 import (  # noqa: E402
    contiguous_ns_runs,
    enumerate_diagnostic_candidates,
    load_diagnostic_protocol,
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


def _plot_lookup(path: Path, lookup: pd.DataFrame, observations: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(lookup["cell_id"].unique()), 1, figsize=(10, 8), squeeze=False)
    for ax, (cell_id, rows) in zip(axes[:, 0], lookup.groupby("cell_id", sort=False)):
        rows = rows.sort_values("mission_start_source_row")
        ax.step(rows["mission_start_source_row"], rows["capacity_ah"], where="post", label="lookup")
        rpts = observations.loc[
            observations["cell_id"].eq(cell_id)
            & observations["detector_status"].eq("accepted")
            & observations["accepted_for_capacity_lookup"].eq(True)  # noqa: E712
        ]
        ax.scatter(rpts["end_source_row"], rpts["capacity_net_ah"], s=12, color="tab:red", label="RPT end")
        ax.set(title=cell_id, ylabel="Capacity (Ah)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1, 0].set_xlabel("Raw source row")
    fig.suptitle("CMU causal capacity lookup v1")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_outputs(
    raw_root: Path,
    output_root: Path,
    lookup_protocol_path: Path,
    mission_protocol_path: Path,
    observations_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> pd.DataFrame:
    protocol = yaml.safe_load(lookup_protocol_path.read_text(encoding="utf-8"))
    mission_protocol, mission_protocol_sha = load_diagnostic_protocol(mission_protocol_path)
    observations_sha = sha256_file(observations_path)
    if observations_sha != protocol["rpt_input"]["sha256"]:
        raise ValueError("Frozen RPT observation SHA-256 mismatch")
    observations = pd.read_csv(observations_path)
    if len(observations) != protocol["rpt_input"]["expected_observation_count"]:
        raise ValueError("Frozen RPT observation count mismatch")

    missions, raw_inputs = [], {}
    for cell_id in protocol["cells"]:
        result = load_cmu_v3_main_file(raw_root / f"{cell_id}.csv", repo_root=repo_root)
        candidates = enumerate_diagnostic_candidates(contiguous_ns_runs(result.data), mission_protocol)
        missions.append(candidates)
        raw_inputs[cell_id] = {
            "path": str(raw_root / f"{cell_id}.csv"),
            "sha256": result.metadata["source_sha256"],
        }
        del result
    mission_manifest = pd.concat(missions, ignore_index=True)
    lookup = build_causal_capacity_lookup(
        mission_manifest,
        observations,
        nominal_capacity_ah=float(protocol["causality"]["nominal_capacity_ah"]),
        protocol_version=protocol["protocol_version"],
    )
    anomaly_counts = validate_lookup(lookup, mission_manifest, observations)
    if any(anomaly_counts.values()):
        raise ValueError(f"Causal lookup QC failed: {anomaly_counts}")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(exist_ok=True)
    lookup.to_csv(output_root / "capacity_lookup.csv", index=False)

    usage = (
        lookup.dropna(subset=["source_rpt_id"])
        .groupby(["cell_id", "source_rpt_id"], sort=False)
        .size()
        .rename("mission_use_count")
        .reset_index()
    )
    all_rpts = observations.loc[:, ["cell_id", "rpt_id", "capacity_net_ah", "end_source_row"]].rename(
        columns={"rpt_id": "source_rpt_id", "capacity_net_ah": "source_rpt_capacity_ah"}
    )
    usage = all_rpts.merge(usage, how="left", on=["cell_id", "source_rpt_id"])
    usage["mission_use_count"] = usage["mission_use_count"].fillna(0).astype(int)
    usage.to_csv(output_root / "rpt_usage.csv", index=False)

    coverage_rows = []
    for cell_id, rows in lookup.groupby("cell_id", sort=False):
        fallback = int(rows["capacity_source"].eq("nominal_fallback").sum())
        observed = len(rows) - fallback
        lags = rows["causal_lag"].dropna()
        coverage_rows.append({
            "cell_id": cell_id,
            "mission_count": len(rows),
            "nominal_fallback_count": fallback,
            "nominal_fallback_fraction": fallback / len(rows) if len(rows) else 0.0,
            "observed_rpt_count": observed,
            "observed_rpt_fraction": observed / len(rows) if len(rows) else 0.0,
            "rpt_observations_used": int(rows["source_rpt_id"].nunique()),
            "minimum_causal_lag_source_rows": int(lags.min()) if len(lags) else None,
            "maximum_causal_lag_source_rows": int(lags.max()) if len(lags) else None,
        })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output_root / "coverage_summary.csv", index=False)
    _plot_lookup(output_root / "figures" / "causal_capacity_lookup_by_cell.png", lookup, observations)

    generated_at = datetime.now(timezone.utc).isoformat()
    qc = {
        "lookup_row_count": len(lookup),
        "capacity_source_counts": {
            key: int(value) for key, value in lookup["capacity_source"].value_counts().items()
        },
        "anomaly_counts": anomaly_counts,
        "coverage_by_cell": coverage_rows,
    }
    (output_root / "qc_summary.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    metadata = {
        "dataset_name": DATASET_NAME,
        "article_id": ARTICLE_ID,
        "dataset_version": DATASET_VERSION,
        "raw_loader_schema_version": LOADER_SCHEMA_VERSION,
        "lookup_protocol_version": LOOKUP_PROTOCOL_VERSION,
        "lookup_key": "cell_id + strict source_row order",
        "selection_rule": protocol["causality"]["selection"],
        "mission_identity_source": "development_candidate_id",
        "mission_input_status": "diagnostic_screening_match",
        "formal_mission_labels": False,
        "generator": "scripts/datasets/cmu_evtol/build_causal_capacity_lookup.py",
        "lookup_protocol_path": str(lookup_protocol_path),
        "lookup_protocol_sha256": sha256_file(lookup_protocol_path),
        "mission_protocol_path": str(mission_protocol_path),
        "mission_protocol_sha256": mission_protocol_sha,
        "rpt_observations_path": str(observations_path),
        "rpt_observations_sha256": observations_sha,
        "rpt_observation_count": len(observations),
        "raw_inputs": raw_inputs,
        "git_revision": get_full_commit_hash(repo_root),
        "git_dirty": is_dirty(repo_root),
        "generated_at_utc": generated_at,
        "nominal_capacity_ah": float(protocol["causality"]["nominal_capacity_ah"]),
        "interpolation_used": False,
        "future_information_used": False,
        "cycle_number_used_for_ordering": False,
        "soc_recomputed": False,
        "dynamic_capacity_soc_generated": False,
    }
    (output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return lookup


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/datasets/cmu_evtol/causal_capacity_lookup_v1"),
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("configs/datasets/cmu_evtol/protocols/cmu_evtol_causal_capacity_lookup_v1.yaml"),
    )
    parser.add_argument(
        "--mission-protocol", type=Path,
        default=Path("configs/datasets/cmu_evtol/protocols/cmu_evtol_mission_segmentation_v0_1.yaml"),
    )
    parser.add_argument(
        "--rpt-observations", type=Path,
        default=Path("outputs/cmu_evtol_rpt_capacity_v1/capacity_observations.csv"),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    lookup = build_outputs(
        args.raw_root, args.output_root, args.protocol, args.mission_protocol,
        args.rpt_observations,
    )
    print(f"Built {len(lookup)} causal lookup rows in {time.perf_counter() - started:.2f} s")


if __name__ == "__main__":
    main()
