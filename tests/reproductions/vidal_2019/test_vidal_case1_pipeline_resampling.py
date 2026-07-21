from __future__ import annotations

import pandas as pd
import pytest

from battery_benchmark.reproductions.vidal_2019.sequence_v2 import validate_case1_roots
from battery_benchmark.reproductions.vidal_2019 import case1
from battery_benchmark.reproductions.vidal_2019.case1 import (
    Case1Config,
    cache_case1_inputs,
    run_case1,
)
from tests.reproductions.vidal_2019.vidal_case1_test_utils import make_v2_split


def test_validate_case1_roots_can_limit_dataset_and_splits(tmp_path):
    make_v2_split(tmp_path, "lg_hg2", "full_train")
    make_v2_split(tmp_path, "lg_hg2", "test")
    limited = validate_case1_roots(tmp_path, datasets=["lg_hg2"], splits=["full_train", "test"])
    assert list(limited["status"]) == ["PASS", "PASS"]
    full = validate_case1_roots(tmp_path)
    assert (full["status"] == "FAIL").any()


def test_cache_root_with_current_dataset_only_runs_smoke(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    cache = tmp_path / "cache"
    for split in ("full_train", "test"):
        make_v2_split(source, "lg_hg2", split)
    cache_case1_inputs(source, cache, "lg_hg2")
    cfg = Case1Config(
        "lg_hg2",
        epochs=1,
        batch_size=2,
        seq_len=4,
        stride=4,
        max_train_windows=2,
        resample_hz=1.0,
    )
    result = run_case1(cfg, cache, output, tmp_path, device_name="cpu")
    assert result["metadata"]["sampling_protocol"] == "paper_aligned_1hz_reconstructed"
    assert result["metadata"]["train_windows"] > 0
    assert (pd.read_csv(result["run_dir"] + "/predictions/test_predictions.csv")["soc_predicted"].notna()).all()


def test_scaler_fits_resampled_full_train_not_raw_count(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    make_v2_split(
        source, "lg_hg2", "full_train",
        [(1, 51, "PASS", "drive_cycle", "Mixed1")],
        timestamp_step=0.1,
    )
    make_v2_split(source, "lg_hg2", "test", [(1, 11, "PASS", "drive_cycle", "UDDS")])
    cfg = Case1Config(
        "lg_hg2",
        epochs=1,
        batch_size=2,
        seq_len=4,
        stride=4,
        max_train_windows=1,
        resample_hz=1.0,
    )
    result = run_case1(cfg, source, output, tmp_path, device_name="cpu")
    assert result["scaler"]["n_fit_samples"] == result["metadata"]["train_resampled_samples"]
    assert result["metadata"]["train_original_samples"] == 51
    assert result["metadata"]["train_resampled_samples"] < 51


def test_test_records_do_not_change_train_scaler_after_resampling(tmp_path):
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    for source, test_n in ((source_a, 11), (source_b, 31)):
        make_v2_split(
            source, "lg_hg2", "full_train",
            [(1, 51, "PASS", "drive_cycle", "Mixed1")],
            timestamp_step=0.1,
        )
        make_v2_split(source, "lg_hg2", "test", [(1, test_n, "PASS", "drive_cycle", "UDDS")])
    cfg = Case1Config("lg_hg2", epochs=1, batch_size=2, seq_len=4, stride=4, max_train_windows=1)
    a = run_case1(cfg, source_a, tmp_path / "out_a", tmp_path, device_name="cpu")
    b = run_case1(cfg, source_b, tmp_path / "out_b", tmp_path, device_name="cpu")
    assert a["scaler"]["feature_min"] == b["scaler"]["feature_min"]
    assert a["scaler"]["feature_max"] == b["scaler"]["feature_max"]


def test_formal_mode_rejects_unknown_commit(monkeypatch, tmp_path):
    for split in ("full_train", "test"):
        make_v2_split(tmp_path / "source", "lg_hg2", split)
    monkeypatch.setattr(case1, "get_full_commit_hash", lambda _repo: "unknown")
    monkeypatch.setattr(case1, "is_dirty", lambda _repo: False)
    cfg = Case1Config(
        "lg_hg2",
        run_mode="formal",
        epochs=515,
        max_train_windows=None,
        development_only=False,
        resample_hz=1.0,
    )
    with pytest.raises(RuntimeError, match="known Git commit"):
        run_case1(cfg, tmp_path / "source", tmp_path / "out", tmp_path, device_name="cpu")
