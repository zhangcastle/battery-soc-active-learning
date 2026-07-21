import numpy as np
import torch

from battery_benchmark.reproductions.vidal_2019.windows import build_case1_windows
from battery_benchmark.reproductions.vidal_2019.sequence_v2 import VidalSequenceV2Dataset
from battery_benchmark.reproductions.vidal_2019.case1 import masked_mse
from tests.reproductions.vidal_2019.vidal_case1_test_utils import make_v2_split


def test_incomplete_test_window_kept_and_masked(tmp_path):
    make_v2_split(tmp_path, "lg_hg2", "test", [(1, 7, "PASS", "drive_cycle", "UDDS")])
    records = VidalSequenceV2Dataset(tmp_path, "lg_hg2", "test").records
    windows = build_case1_windows(records, seq_len=5, stride=5, keep_incomplete=True)
    assert len(windows.X) == 2
    assert windows.valid_mask.sum() == 7
    pred = torch.zeros((2, 5)); target = torch.zeros((2, 5)); target[1, 2:] = 100
    assert masked_mse(pred, target, torch.from_numpy(windows.valid_mask)).item() == 0
