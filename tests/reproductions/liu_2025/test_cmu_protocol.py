from pathlib import Path

import pandas as pd
import pytest

from battery_benchmark.reproductions.liu_2025.cmu_protocol import (
    select_duration_matched_missions,
)

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1.csv"


def test_duration_matched_core7_counts_and_split_isolation():
    manifest = pd.read_csv(MANIFEST)
    expected = {
        "VAH01": 9,
        "VAH17": 9,
        "VAH27": 9,
        "VAH30": 9,
        "VAH15": 8,
        "VAH12": 16,
        "VAH11": 9,
    }
    for cell_id, count in expected.items():
        adaptation, validation, test, metadata = select_duration_matched_missions(manifest, cell_id)
        assert len(adaptation) == count
        assert not adaptation & validation
        assert not adaptation & test
        assert not validation & test
        assert metadata["adaptation_seconds"] >= metadata["budget_seconds"]


def test_duration_budget_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        select_duration_matched_missions(pd.DataFrame(), "VAH30", budget_seconds=0)
