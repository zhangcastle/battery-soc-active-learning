import numpy as np
import pandas as pd

from battery_benchmark.data.vidal_sequence_v2_dataset import VidalV2Record
from battery_benchmark.experiments.vidal_case1 import fit_train_normalizer


def record(values):
    frame = pd.DataFrame({"voltage": values, "current": values, "temperature": values})
    return VidalV2Record(frame, {})


def test_test_extremes_cannot_change_train_scaler(tmp_path):
    train = [record(np.array([1.0, 2.0]))]
    a, meta_a = fit_train_normalizer(train, "lg_hg2", tmp_path)
    _test_a = record(np.array([-999.0, 999.0]))
    b, meta_b = fit_train_normalizer(train, "lg_hg2", tmp_path)
    assert a.state_dict() == b.state_dict()
    assert meta_a["fit_split"] == meta_b["fit_split"] == "full_train"
