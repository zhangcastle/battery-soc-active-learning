"""Abstract base class for all battery dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseBatteryDataset(ABC):
    """Interface that every dataset adapter must implement.

    Each adapter is responsible for reading raw files, standardising column
    names to the benchmark schema, and returning a list of per-cycle
    DataFrames.  Adapters must never concatenate cycles before the benchmark
    code has assigned split labels — doing so risks cross-split leakage.

    Parameters
    ----------
    config:
        Dataset section of the experiment config dict.  Must contain at least
        ``name``, ``domain_id``, and either an explicit ``raw_dir`` or rely
        on the paths provided by the calling experiment runner.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.domain_id: str = config.get("domain_id", "unknown")
        self.name: str = config.get("name", "unknown")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def discover_files(self, raw_dir: str | Path) -> list[Path]:
        """Return a list of raw data file paths under *raw_dir*.

        Parameters
        ----------
        raw_dir:
            Root directory for this dataset's raw files.
        """

    @abstractmethod
    def build_manifest(self, raw_dir: str | Path) -> pd.DataFrame:
        """Scan *raw_dir* and return a manifest DataFrame.

        The manifest should contain one row per cycle with at minimum:
        ``file_path``, ``cell_id``, ``cycle_id``, ``profile_name``.
        """

    @abstractmethod
    def load_cycle(self, file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load a single raw cycle file and return a standardised DataFrame."""

    @abstractmethod
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename raw columns to benchmark schema column names."""

    @abstractmethod
    def validate_cycle(self, df: pd.DataFrame) -> bool:
        """Return True if *df* passes quality checks, False otherwise."""

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return a dict of dataset-level metadata (name, capacity, etc.)."""

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def get_split(self, df: pd.DataFrame) -> str:
        """Return the split label for a cycle DataFrame (if already assigned)."""
        from battery_benchmark.data.schema import COLUMNS
        if COLUMNS.SPLIT in df.columns and df[COLUMNS.SPLIT].nunique() == 1:
            return str(df[COLUMNS.SPLIT].iloc[0])
        return "unsplit"

    def save_processed_cycle(
        self, df: pd.DataFrame, output_path: str | Path
    ) -> None:
        """Save a processed cycle DataFrame to Parquet."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

    def load_processed_cycle(self, path: str | Path) -> pd.DataFrame:
        """Load a previously saved processed cycle from Parquet."""
        return pd.read_parquet(path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, domain={self.domain_id!r})"
