from battery_benchmark.data.vidal_case1_windows import build_case1_windows
from battery_benchmark.data.vidal_sequence_v2_dataset import VidalSequenceV2Dataset
from vidal_case1_test_utils import make_v2_split


def test_each_window_has_one_record_and_order(tmp_path):
    make_v2_split(tmp_path, "lg_hg2", "full_train")
    records = VidalSequenceV2Dataset(tmp_path, "lg_hg2", "full_train").records
    windows = build_case1_windows(records, seq_len=4, stride=4)
    assert all(m["source_filename"] in {"record_1.mat", "record_2.mat"} for m in windows.metadata)
    assert all(m["window_end"] <= len(next(r.data for r in records if r.metadata["source_filename"] == m["source_filename"])) for m in windows.metadata)
