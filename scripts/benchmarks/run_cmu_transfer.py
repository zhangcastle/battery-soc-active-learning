#!/usr/bin/env python
"""Run CMU eVTOL transfer experiment.

Requires CMU eVTOL dataset and source pre-trained checkpoint.
See docs/datasets/cmu_evtol/overview.md for the evaluation protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from battery_benchmark.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    print("CMU eVTOL transfer requires the dataset adapter (not yet implemented).")
    print("See notebooks/benchmarks/40_cmu_evtol_transfer.ipynb and docs/datasets/cmu_evtol/overview.md.")


if __name__ == "__main__":
    main()
