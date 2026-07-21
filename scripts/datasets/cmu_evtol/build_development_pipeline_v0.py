#!/usr/bin/env python3
"""Build development-only CMU baseline candidate/SOC artifacts."""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.datasets.cmu_evtol.development_pipeline_v0 import (  # noqa: E402
    BASELINE_CELLS,
    write_pipeline_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/datasets/cmu_evtol/development_pipeline_v0"),
    )
    parser.add_argument("--cells", nargs="+", default=list(BASELINE_CELLS))
    parser.add_argument("--capacity-mode", choices=["constant"], default="constant")
    parser.add_argument("--capacity-ah", type=float, default=3.0)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/datasets/cmu_evtol/protocols/cmu_evtol_mission_segmentation_v0_1.yaml"),
    )
    parser.add_argument(
        "--asset-inventory",
        type=Path,
        default=Path("data/manifests/cmu_evtol/cmu_evtol_vtc6_asset_inventory_v3.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    metadata = write_pipeline_outputs(
        args.raw_root,
        args.output_root,
        args.protocol,
        asset_inventory_path=args.asset_inventory,
        cells=args.cells,
        capacity_ah=args.capacity_ah,
        repo_root=REPO_ROOT,
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        f"Built {metadata['pipeline_status']} CMU pipeline for {', '.join(args.cells)} "
        f"in {time.perf_counter() - started:.2f} s (peak_rss_native={peak})"
    )


if __name__ == "__main__":
    main()
