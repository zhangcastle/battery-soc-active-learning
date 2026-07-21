from battery_benchmark.reproductions.vidal_2019.windows import build_case1_windows
from battery_benchmark.reproductions.vidal_2019.sequence_v2 import VidalSequenceV2Dataset
from tests.reproductions.vidal_2019.vidal_case1_test_utils import make_v2_split


def test_window_segment_matches_source_record(tmp_path):
    make_v2_split(tmp_path, "lg_hg2", "full_train")
    records = VidalSequenceV2Dataset(tmp_path, "lg_hg2", "full_train").records
    expected = {r.metadata["source_filename"]: r.metadata["sequence_segment_id"] for r in records}
    windows = build_case1_windows(records, 4, 4)
    assert all(m["sequence_segment_id"] == expected[m["source_filename"]] for m in windows.metadata)
