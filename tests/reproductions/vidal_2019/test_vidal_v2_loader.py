import pandas as pd
import pytest

from battery_benchmark.reproductions.vidal_2019.sequence_v2 import VidalSequenceV2Dataset
from tests.reproductions.vidal_2019.vidal_case1_test_utils import make_v2_split


def test_loader_maps_columns_and_uses_sequence_order(tmp_path):
    make_v2_split(tmp_path, "lg_hg2", "full_train")
    loaded = VidalSequenceV2Dataset(tmp_path, "lg_hg2", "full_train")
    assert [r.metadata["sequence_order"] for r in loaded.records] == [1, 2]
    assert {"timestamp", "voltage", "current", "temperature", "soc"} <= set(loaded.records[0].data)
    assert list(loaded.inventory["label_qc_status"]) == ["WARNING", "PASS"]


def test_loader_rejects_fail(tmp_path):
    make_v2_split(tmp_path, "lg_hg2", "test", [(1, 5, "FAIL", "drive_cycle", "UDDS")])
    with pytest.raises(ValueError, match="disallowed"):
        VidalSequenceV2Dataset(tmp_path, "lg_hg2", "test")


def test_loader_detects_unregistered_record(tmp_path):
    split = make_v2_split(tmp_path, "lg_hg2", "test", [(1, 5, "PASS", "drive_cycle", "UDDS")])
    pd.DataFrame({"x": [1]}).to_parquet(split / "records" / "extra.parquet")
    with pytest.raises(ValueError, match="unregistered"):
        VidalSequenceV2Dataset(tmp_path, "lg_hg2", "test")
