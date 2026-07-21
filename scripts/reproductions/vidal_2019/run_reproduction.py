#!/usr/bin/env python
"""Run all four Vidal et al. cases sequentially.

Requires Panasonic, LG HG2, Samsung 48G, and Samsung 30T datasets.
See docs/known_uncertainties.md before interpreting results.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from battery_benchmark.utils.logging import setup_logging

CASES = ["vidal_case1", "vidal_case2", "vidal_case3", "vidal_case4"]


def main() -> None:
    setup_logging()
    print("Vidal reproduction requires dataset adapters (not yet implemented).")
    print("Cases to run:", CASES)
    print("See docs/reproductions/vidal_2019/protocol.md for instructions.")


if __name__ == "__main__":
    main()
