"""Synthetic battery dataset generator.

Generates physically plausible but fully synthetic voltage / current /
temperature / SOC sequences.  Used for smoke tests and unit tests — no real
data files needed.

The synthetic model:
- Voltage ~ 3.0 + 1.2 * SOC with Gaussian noise
- Current ~ constant discharge with small noise
- Temperature ~ 25 + 5 * (1 - SOC) with noise
- SOC decreases linearly from soc_initial to soc_final over the cycle
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from battery_benchmark.data.base_dataset import BaseBatteryDataset
from battery_benchmark.data.registry import register_dataset
from battery_benchmark.data.schema import COLUMNS, LabelSource


@register_dataset("synthetic")
class SyntheticDataset(BaseBatteryDataset):
    """Generates synthetic battery cycles for benchmarking without real data.

    Parameters
    ----------
    config:
        Must contain ``domain_id``.  Accepts these optional keys:

        * ``num_cells`` (int, default 2)
        * ``num_cycles_per_cell`` (int, default 10)
        * ``samples_per_cycle`` (int, default 100)
        * ``noise_std`` (float, default 0.01)
        * ``soc_initial`` (float, default 1.0)
        * ``soc_final`` (float, default 0.0)
        * ``seed`` (int, default 0)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.num_cells: int = int(config.get("num_cells", 2))
        self.num_cycles_per_cell: int = int(config.get("num_cycles_per_cell", 10))
        self.samples_per_cycle: int = int(config.get("samples_per_cycle", 100))
        self.noise_std: float = float(config.get("noise_std", 0.01))
        self.soc_initial: float = float(config.get("soc_initial", 1.0))
        self.soc_final: float = float(config.get("soc_final", 0.0))
        self.seed: int = int(config.get("seed", 0))

    # ------------------------------------------------------------------
    # BaseBatteryDataset interface
    # ------------------------------------------------------------------

    def discover_files(self, raw_dir=None):
        """Synthetic data has no files; returns empty list."""
        return []

    def build_manifest(self, raw_dir=None) -> pd.DataFrame:
        """Return a manifest of (cell_id, cycle_id) pairs."""
        rows = []
        for cell_idx in range(self.num_cells):
            cell_id = f"{self.domain_id}_cell_{cell_idx:02d}"
            for cycle_idx in range(self.num_cycles_per_cell):
                rows.append({
                    "cell_id": cell_id,
                    "cycle_id": cycle_idx,
                    "domain_id": self.domain_id,
                    "profile_name": "synthetic_discharge",
                })
        return pd.DataFrame(rows)

    def load_cycle(self, file_path=None, **kwargs) -> pd.DataFrame:
        """Not applicable for synthetic data; use ``generate_cycles`` instead."""
        raise NotImplementedError(
            "SyntheticDataset does not load from files. "
            "Use SyntheticDataset.generate_cycles() instead."
        )

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df  # already in schema format

    def validate_cycle(self, df: pd.DataFrame) -> bool:
        required = [COLUMNS.VOLTAGE, COLUMNS.CURRENT, COLUMNS.TEMPERATURE, COLUMNS.SOC]
        return all(c in df.columns for c in required)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain_id": self.domain_id,
            "num_cells": self.num_cells,
            "num_cycles_per_cell": self.num_cycles_per_cell,
            "samples_per_cycle": self.samples_per_cycle,
            "label_source": LabelSource.COULOMB_COUNTED_FIXED_CAPACITY.value,
            "label_available": True,
        }

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def generate_cycles(self) -> list[pd.DataFrame]:
        """Generate all synthetic cycles and return as a list of DataFrames.

        Returns
        -------
        list[pd.DataFrame]
            One DataFrame per cycle.  Each contains the full benchmark schema.
        """
        rng = np.random.default_rng(self.seed)
        cycles: list[pd.DataFrame] = []
        for cell_idx in range(self.num_cells):
            cell_id = f"{self.domain_id}_cell_{cell_idx:02d}"
            for cycle_idx in range(self.num_cycles_per_cell):
                cycle_seed = int(rng.integers(0, 2**31))
                df = self._generate_single_cycle(
                    cell_id=cell_id,
                    cycle_id=cycle_idx,
                    seed=cycle_seed,
                )
                cycles.append(df)
        return cycles

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_single_cycle(
        self,
        cell_id: str,
        cycle_id: int,
        seed: int,
    ) -> pd.DataFrame:
        """Generate one discharge cycle as a DataFrame."""
        rng = np.random.default_rng(seed)
        n = self.samples_per_cycle

        t = np.linspace(0.0, 3600.0, n)
        soc = np.linspace(self.soc_initial, self.soc_final, n)

        # Physical-inspired relationships with additive noise
        voltage = (
            3.0 + 1.2 * soc
            + rng.normal(0, self.noise_std, n)
        )
        current = (
            -1.5 * np.ones(n)
            + rng.normal(0, self.noise_std * 0.5, n)
        )
        temperature = (
            25.0 + 5.0 * (1.0 - soc)
            + rng.normal(0, self.noise_std * 2.0, n)
        )

        return pd.DataFrame({
            COLUMNS.TIMESTAMP: t,
            COLUMNS.VOLTAGE: voltage,
            COLUMNS.CURRENT: current,
            COLUMNS.TEMPERATURE: temperature,
            COLUMNS.SOC: soc,
            COLUMNS.CELL_ID: cell_id,
            COLUMNS.CYCLE_ID: cycle_id,
            COLUMNS.DOMAIN_ID: self.domain_id,
            COLUMNS.PROFILE_NAME: "synthetic_discharge",
            COLUMNS.SPLIT: "unsplit",
            COLUMNS.LABEL_AVAILABLE: True,
            COLUMNS.LABEL_SOURCE: LabelSource.COULOMB_COUNTED_FIXED_CAPACITY.value,
            COLUMNS.LABEL_QUALITY: 1.0,
            COLUMNS.SOURCE_DATASET: self.name,
        })
