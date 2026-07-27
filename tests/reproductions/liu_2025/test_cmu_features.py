from __future__ import annotations

import numpy as np
import pandas as pd

from battery_benchmark.datasets.cmu_evtol.vidal_adapter import (
    UDA_COLUMNS,
    adapt_cmu_features_for_uda,
    load_cmu_features_for_uda,
)
from battery_benchmark.reproductions.liu_2025.windows import build_liu_windows


def _features() -> pd.DataFrame:
    elapsed = np.arange(45, dtype=float)
    mission = "cmu:VAH30:mission-1"
    return pd.DataFrame(
        {
            "cell_id": "VAH30",
            "mission_cycle_id": mission,
            "development_candidate_id": mission,
            "source_row": np.arange(45),
            "time_s": 1000.0 + elapsed,
            "elapsed_candidate_s": elapsed,
            "voltage_v": np.linspace(4.2, 3.2, 45),
            "current_a": np.linspace(-10.0, -5.0, 45),
            "Temperature__C": np.linspace(35.0, 40.0, 45),
        }
    )


def test_cmu_uda_path_has_no_labels_and_builds_liu_windows():
    result = adapt_cmu_features_for_uda(_features())
    record = result.records[0]
    assert "soc" not in record.data
    assert record.metadata["label_usage"] == "forbidden"
    x, y = build_liu_windows(result.records, require_labels=False)
    assert x.shape == (6, 40, 3)
    assert y is None


def test_cmu_uda_loader_reads_only_feature_columns(monkeypatch):
    requested = []

    def fake_read_parquet(_path, *, columns):
        requested.extend(columns)
        return _features()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    result = load_cmu_features_for_uda(["target.parquet"])
    assert len(result.records) == 1
    assert requested == list(UDA_COLUMNS)
    assert not {"soc_dynamic", "capacity_ah", "label_source"} & set(requested)
