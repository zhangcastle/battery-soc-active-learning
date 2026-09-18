"""Paper-faithful Shen et al. (2023) TATN reproduction.

This package restores the original TATN architecture and training/suite code
exactly as it existed before commit ``973525c`` ("Refactor Shen et al. (2023)
reproduction: Introduce MEDNet and DANNBaselineNet architectures"), which
replaced the TATN implementation in ``battery_benchmark.reproductions.shen_2023``
with the unrelated MEDNet/DANNBaselineNet architectures and left a
``TATN = DANNBaselineNet`` backwards-compatible alias behind.

TATN (here) and MED/DANN+MMD (in ``shen_2023``) are two independent
frameworks and must not be merged or substituted for one another. Only the
CMU-agnostic Panasonic data-loading utilities in ``shen_2023.data`` are
shared, because that module is unchanged from before the refactor.
"""

from battery_benchmark.reproductions.shen_2023.data import (
    PROFILES,
    TEMPERATURES,
    load_temperature_cycles,
)

from .model import TATN

__all__ = ["PROFILES", "TEMPERATURES", "TATN", "load_temperature_cycles"]
