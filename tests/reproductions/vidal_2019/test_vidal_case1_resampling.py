from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_benchmark.reproductions.vidal_2019.resampling import resample_vidal_record
from battery_benchmark.reproductions.vidal_2019.sequence_v2 import VidalV2Record


def _record(time, soc=None):
    time = np.asarray(time, dtype=float)
    if soc is None:
        soc = np.linspace(1.0, 0.5, len(time))
    frame = pd.DataFrame({
        "timestamp": time,
        "time_s": time,
        "voltage": np.linspace(4.2, 3.2, len(time)),
        "current": np.linspace(-2.0, -1.0, len(time)),
        "temperature": np.linspace(25.0, 26.0, len(time)),
        "soc": soc,
    })
    return VidalV2Record(frame, {
        "dataset": "lg_hg2", "protocol_split": "full_train", "sequence_order": 1,
        "sequence_segment_id": 1, "source_filename": "synthetic.mat",
        "file_type": "drive_cycle", "paper_profile": "Mixed1",
        "label_qc_status": "PASS", "warning_reasons": "",
    })


def test_10hz_record_resamples_to_1hz_reasonable_grid():
    out = resample_vidal_record(_record(np.arange(0.0, 5.0, 0.1)), target_hz=1.0)
    diffs = np.diff(out.data["timestamp"])
    assert np.allclose(diffs[:-1], 1.0)
    assert 0 < diffs[-1] <= 1.0
    assert out.data["timestamp"].iloc[0] == pytest.approx(0.0)
    assert out.data["timestamp"].iloc[-1] <= 4.9
    assert len(out.data) == 6
    assert out.metadata["original_n_samples"] == 50
    assert out.metadata["resampled_n_samples"] == 6
    assert out.data["soc"].between(0, 1).all()


def test_sparse_charge_resamples_without_nan_and_preserves_endpoints():
    out = resample_vidal_record(_record([0.0, 2.5, 7.0], soc=[0.2, 0.5, 1.0]), target_hz=1.0)
    assert np.all(np.diff(out.data["timestamp"]) > 0)
    assert np.isfinite(out.data[["voltage", "current", "temperature", "soc"]].to_numpy()).all()
    assert out.data["soc"].iloc[0] == pytest.approx(0.2)
    assert out.data["soc"].iloc[-1] == pytest.approx(1.0)


def test_duplicate_timestamps_are_averaged_before_resampling():
    record = _record([0.0, 0.0, 1.0, 2.0], soc=[1.0, 0.8, 0.6, 0.4])
    out = resample_vidal_record(record, target_hz=1.0)
    assert out.metadata["duplicate_timestamp_count"] == 2
    assert out.metadata["deduplicated_n_samples"] == 3
    assert out.data["timestamp"].is_unique
    assert out.data["soc"].iloc[0] == pytest.approx(0.9)


def test_soc_clipping_can_be_disabled_without_changing_default():
    record = _record([0.0, 1.0], soc=[1.1, -0.1])
    default = resample_vidal_record(record, target_hz=1.0)
    unchanged = resample_vidal_record(record, target_hz=1.0, clip_soc=False)
    assert default.data["soc"].tolist() == [1.0, 0.0]
    assert default.metadata["soc_clipping_count"] == 2
    assert unchanged.data["soc"].tolist() == [1.1, -0.1]
    assert unchanged.metadata["soc_clipping_count"] == 0
