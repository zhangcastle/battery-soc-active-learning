from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from battery_benchmark.datasets.cmu_evtol.vidal_adapter import (
    EXPECTED_LABEL_SOURCE,
    EXPECTED_SOC_PROTOCOL,
    adapt_cmu_dynamic_soc,
    build_cmu_vidal_windows,
    load_cmu_dynamic_soc_for_vidal,
    transform_cmu_records,
)


REAL_ROOT = Path("outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1")


def _frame(n: int = 101, *, cell: str = "VAH01", mission: str | None = None) -> pd.DataFrame:
    mission = mission or f"cmu_v3_dev_candidate_v0:{cell}:b0:s10:e110"
    elapsed = np.arange(n, dtype=float) * 10.0
    return pd.DataFrame({
        "cell_id": cell,
        "mission_cycle_id": mission,
        "development_candidate_id": mission,
        "source_row": np.arange(10, 10 + n),
        "time_s": 1000.0 + elapsed,
        "elapsed_candidate_s": elapsed,
        "voltage_v": np.linspace(4.1, 3.1, n),
        "current_a": np.linspace(-20.0, -4.0, n),
        "Temperature__C": np.linspace(24.0, 40.0, n),
        "soc_dynamic": np.linspace(1.0, 0.4, n),
        "capacity_ah": 2.9,
        "capacity_source": "observed_rpt",
        "source_rpt_id": f"rpt:{cell}:1",
        "dynamic_soc_protocol_version": EXPECTED_SOC_PROTOCOL,
        "label_source": EXPECTED_LABEL_SOURCE,
        "label_is_ground_truth": False,
        "formal_benchmark_eligible": False,
    })


def test_mapping_resampling_windows_and_provenance():
    frame = _frame()
    result = adapt_cmu_dynamic_soc(frame, target_hz=1.0)
    assert len(result.records) == 1
    record = result.records[0]
    assert len(record.data) == 1001
    assert np.array_equal(record.data["voltage"], np.interp(record.data["timestamp"], frame["elapsed_candidate_s"], frame["voltage_v"]))
    assert np.array_equal(record.data["current"], np.interp(record.data["timestamp"], frame["elapsed_candidate_s"], frame["current_a"]))
    assert np.array_equal(record.data["temperature"], np.interp(record.data["timestamp"], frame["elapsed_candidate_s"], frame["Temperature__C"]))
    assert np.array_equal(record.data["soc"], np.interp(record.data["timestamp"], frame["elapsed_candidate_s"], frame["soc_dynamic"]))
    assert record.data["current"].max() < 0
    assert record.metadata["label_is_ground_truth"] is False
    assert record.metadata["formal_benchmark_eligible"] is False
    assert record.metadata["soc_clipping_enabled"] is False
    assert record.metadata["soc_clipping_count"] == 0
    windows = build_cmu_vidal_windows(result.records)
    assert windows.X.shape == (20, 50, 3)
    assert windows.y.shape == (20, 50)
    assert windows.valid_mask.all()
    assert windows.dropped_samples == 1
    assert windows.metadata[0]["cell_id"] == "VAH01"
    assert windows.metadata[0]["source_row_left"] == 10


def test_candidates_and_cells_remain_independent():
    a = _frame(6, cell="VAH01")
    b = _frame(6, cell="VAH17")
    result = adapt_cmu_dynamic_soc(pd.concat([a, b], ignore_index=True), target_hz=None)
    assert [(r.metadata["cell_id"], r.metadata["mission_cycle_id"]) for r in result.records] == [
        ("VAH01", a["mission_cycle_id"].iloc[0]),
        ("VAH17", b["mission_cycle_id"].iloc[0]),
    ]
    windows = build_cmu_vidal_windows(result.records, seq_len=5, stride=5, keep_incomplete=True)
    assert windows.X.shape == (4, 5, 3)
    assert [m["cell_id"] for m in windows.metadata] == ["VAH01", "VAH01", "VAH17", "VAH17"]
    assert windows.valid_mask.sum() == 12


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.drop(columns="voltage_v"), "missing columns"),
        (lambda d: d.assign(current_a=np.nan), "non-finite"),
        (lambda d: d.assign(source_row=[10] * len(d)), "duplicate source_row"),
        (lambda d: d.iloc[[0, 2, 1, *range(3, len(d))]], "source_row must be strictly increasing"),
        (lambda d: d.assign(time_s=d["time_s"].where(d.index != 2, d["time_s"].iloc[1])), "time axes"),
        (lambda d: d.assign(time_s=d["time_s"].where(d.index != 2, d["time_s"].iloc[0])), "time axes"),
        (lambda d: d.assign(cell_id=None), "identity fields"),
        (lambda d: d.assign(development_candidate_id="other"), "must agree"),
        (lambda d: d.assign(capacity_ah=np.where(d.index == 0, 3.0, 2.9)), "must be constant"),
        (lambda d: d.assign(capacity_source=np.where(d.index == 0, "other", "observed_rpt")), "must be constant"),
        (lambda d: d.assign(label_source="wrong"), "label_source"),
        (lambda d: d.assign(dynamic_soc_protocol_version="wrong"), "dynamic SOC protocol"),
        (lambda d: d.assign(label_is_ground_truth=True), "ground truth"),
        (lambda d: d.assign(formal_benchmark_eligible=True), "formal-benchmark eligible"),
        (lambda d: d.assign(soc_dynamic=1.1), "outside"),
    ],
)
def test_invalid_inputs_fail_closed(mutate, message):
    with pytest.raises(ValueError, match=message):
        adapt_cmu_dynamic_soc(mutate(_frame(6)).reset_index(drop=True), target_hz=None)


def test_reappearing_candidate_fails_instead_of_sorting():
    a = _frame(3)
    b = _frame(3, mission="cmu_v3_dev_candidate_v0:VAH01:b0:s20:e22")
    b["source_row"] += 20
    interleaved = pd.concat([a.iloc[:2], b, a.iloc[2:]], ignore_index=True)
    with pytest.raises(ValueError, match="contiguous"):
        adapt_cmu_dynamic_soc(interleaved, target_hz=None)


def test_same_mission_identity_cannot_span_cells():
    a = _frame(3)
    b = _frame(3, cell="VAH17", mission=a["mission_cycle_id"].iloc[0])
    with pytest.raises(ValueError, match="cannot span multiple cells"):
        adapt_cmu_dynamic_soc(pd.concat([a, b], ignore_index=True), target_hz=None)


def test_adapter_only_calls_transform_on_existing_normalizer():
    class Spy:
        fit_calls = 0
        transform_calls = 0

        def fit(self, _):
            self.fit_calls += 1
            raise AssertionError("adapter must not fit")

        def transform(self, values):
            self.transform_calls += 1
            return values + 1.0

    result = adapt_cmu_dynamic_soc(_frame(6), target_hz=None)
    spy = Spy()
    transformed = transform_cmu_records(result.records, spy)
    assert spy.fit_calls == 0
    assert spy.transform_calls == 1
    assert np.allclose(transformed[0].data["voltage"], result.records[0].data["voltage"] + 1.0)


def test_protocol_locks_feature_order_and_no_adapter_fit():
    path = Path("configs/datasets/cmu_evtol/protocols/cmu_evtol_vidal_adapter_v1.yaml")
    protocol = yaml.safe_load(path.read_text())
    assert protocol["features"]["order"] == ["voltage", "current", "temperature"]
    assert protocol["normalization"]["adapter_fits_scaler"] is False
    assert protocol["sampling"]["soc_clipping"] is False


@pytest.mark.integration
def test_real_dynamic_soc_adapter_inventory():
    paths = [REAL_ROOT / f"{cell}_dynamic_soc.parquet" for cell in ("VAH01", "VAH17", "VAH27")]
    if not all(path.is_file() for path in paths):
        pytest.skip("frozen CMU dynamic SOC artifacts are not available")
    expected = {
        "VAH01": (828, 381353, 812268, 15732, 25668),
        "VAH17": (980, 99960, 961380, 18620, 30380),
        "VAH27": (572, 58344, 561130, 10868, 17730),
    }
    for path in paths:
        cell = path.name[:5]
        result = load_cmu_dynamic_soc_for_vidal([path], target_hz=1.0)
        windows = build_cmu_vidal_windows(result.records)
        records, before, after, n_windows, dropped = expected[cell]
        assert (len(result.records), int(result.inventory["original_rows"].sum())) == (records, before)
        assert (int(result.inventory["adapted_rows"].sum()), len(windows.X), windows.dropped_samples) == (after, n_windows, dropped)
        assert int(result.inventory["duplicate_timestamp_count"].sum()) == 0
        assert int(result.inventory["soc_clipping_count"].sum()) == 0
        assert all(m["cell_id"] == cell for m in windows.metadata)
        assert all(m["mission_cycle_id"] == m["development_candidate_id"] for m in windows.metadata)
        for record in result.records:
            assert record.data["soc"].between(0, 1).all()
            assert np.all(np.diff(record.data["timestamp"]) > 0)
