#!/usr/bin/env python3
"""Generate the sequential per-cell CMU Version 3 raw QC summary."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.data.cmu_evtol_raw import (  # noqa: E402
    DEFAULT_GAP_THRESHOLD_S,
    build_raw_qc_summary,
    write_raw_qc_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sequential CMU KiltHub Version 3 continuous-block QC"
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/cmu_evtol_vtc6_raw_qc_v3_120s.csv"),
    )
    parser.add_argument(
        "--gap-threshold-s",
        type=float,
        default=None,
        help=(
            f"Exploratory block-QC gap threshold in seconds; omit to use the "
            f"{DEFAULT_GAP_THRESHOLD_S:g} s exploratory loader default. This is "
            "not a formal downstream benchmark threshold"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    summary = build_raw_qc_summary(
        args.raw_root,
        gap_threshold_s=args.gap_threshold_s,
        repo_root=REPO_ROOT,
    )
    output = write_raw_qc_summary(summary, args.output)
    elapsed = time.perf_counter() - started
    threshold = float(summary.iloc[0]["gap_threshold_s"])
    threshold_source = summary.iloc[0]["gap_threshold_source"]
    print(
        f"Wrote {len(summary)} cell QC rows to {output} in {elapsed:.2f} s "
        f"(gap threshold={threshold:g} s, source={threshold_source})"
    )


if __name__ == "__main__":
    main()
