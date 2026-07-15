import torch

from battery_benchmark.data.vidal_case1_windows import build_case1_windows
from battery_benchmark.data.vidal_sequence_v2_dataset import VidalSequenceV2Dataset
from battery_benchmark.experiments.vidal_case1 import Case1Config, evaluate_model
from vidal_case1_test_utils import make_v2_split


class ZeroModel(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[:2], device=x.device)


def test_prediction_rows_restore_sequence_and_sample_order(tmp_path):
    make_v2_split(tmp_path, "lg_hg2", "test")
    loaded = VidalSequenceV2Dataset(tmp_path, "lg_hg2", "test")
    windows = build_case1_windows(loaded.records, seq_len=5, stride=5, keep_incomplete=True)
    _, predictions = evaluate_model(
        ZeroModel(), windows, loaded.records, Case1Config("lg_hg2"), torch.device("cpu")
    )
    pairs = list(zip(predictions["sequence_order"], predictions["sample_index_in_record"]))
    assert pairs == sorted(pairs)
    assert len(predictions) == sum(len(record.data) for record in loaded.records)
