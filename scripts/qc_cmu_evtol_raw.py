#!/usr/bin/env python3
"""Generate the sequential per-cell CMU Version 3 raw QC summary."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.data.cmu_evtol_raw import (  # noqa: E402
    DEFAULT_GAP_THRESHOLD_S,
    build_raw_qc_summary,
    write_raw_qc_summary,
)

DEFAULT_QC_OUTPUT = Path("data/manifests/cmu_evtol_vtc6_raw_qc_v3_120s.csv")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sequential CMU KiltHub Version 3 continuous-block QC"
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            f"Output CSV. Defaults to {DEFAULT_QC_OUTPUT} only when the effective "
            f"threshold is {DEFAULT_GAP_THRESHOLD_S:g} s"
        ),
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
    return parser.parse_args(argv)


def resolve_output_path(
    gap_threshold_s: float | None,
    output: Path | None,
) -> Path:
    """Prevent a non-120 s scan from being labelled as a 120 s manifest."""
    if output is not None:
        return output
    effective_threshold = (
        DEFAULT_GAP_THRESHOLD_S if gap_threshold_s is None else gap_threshold_s
    )
    if effective_threshold != DEFAULT_GAP_THRESHOLD_S:
        raise ValueError(
            "--output is required when --gap-threshold-s is not "
            f"{DEFAULT_GAP_THRESHOLD_S:g}; refusing to write a non-120 s result "
            f"to {DEFAULT_QC_OUTPUT}"
        )
    return DEFAULT_QC_OUTPUT


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args.gap_threshold_s, args.output)
    started = time.perf_counter()
    summary = build_raw_qc_summary(
        args.raw_root,
        gap_threshold_s=args.gap_threshold_s,
        repo_root=REPO_ROOT,
    )
    output = write_raw_qc_summary(summary, output_path)
    elapsed = time.perf_counter() - started
    threshold = float(summary.iloc[0]["gap_threshold_s"])
    threshold_source = summary.iloc[0]["gap_threshold_source"]
    print(
        f"Wrote {len(summary)} cell QC rows to {output} in {elapsed:.2f} s "
        f"(gap threshold={threshold:g} s, source={threshold_source})"
    )


if __name__ == "__main__":
    main()
