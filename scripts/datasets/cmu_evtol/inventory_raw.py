#!/usr/bin/env python3
"""Generate the CMU KiltHub Version 3 raw asset inventory CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_benchmark.datasets.cmu_evtol.raw import (  # noqa: E402
    build_asset_inventory,
    write_asset_inventory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory CMU eVTOL KiltHub Version 3 raw assets without conversion"
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Directory containing VAH##.csv and optional VAH##_impedance.csv files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/cmu_evtol/cmu_evtol_vtc6_asset_inventory_v3.csv"),
        help="Output inventory CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = build_asset_inventory(args.raw_root)
    output = write_asset_inventory(inventory, args.output)
    present = int(inventory["main_file_present"].sum())
    impedance_present = int(inventory["impedance_file_present"].sum())
    print(
        f"Wrote {len(inventory)} expected cells to {output} "
        f"({present} main files, {impedance_present} impedance files present)"
    )


if __name__ == "__main__":
    main()
