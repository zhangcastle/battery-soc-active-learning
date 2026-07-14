"""Tests for shared manifest+mat adapter utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_benchmark.data.adapter_common import (
    AdapterLoadStats,
    build_adapter_summary,
    detect_adapter_qc,
    measurements_to_dataframe,
    parse_manifest_soc_init,
)
from battery_benchmark.data.adapter_qc import QCResult
from battery_benchmark.data.manifest_loader import ManifestEntry
from battery_benchmark.data.sequence_record import SequenceRecord


def _entry(**kwargs) -> ManifestEntry:
    defaults = dict(
        battery="lg_hg2",
        temperature_c=25.0,
        filename="f.mat",
        relative_path="data/f.mat",
        profile="Mixed1",
        file_type="drive_cycle",
        full_train=1,
        partial_train=0,
        test=0,
        exclude=0,
        qc_status="PASS",
        paper_profile="Mix1",
        soc_init_str="1.0",
    )
    defaults.update(kwargs)
    return ManifestEntry(**defaults)


def _valid_df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "time_s": np.arange(n, dtype=float),
        "timestamp": np.array([None] * n, dtype=object),
        "voltage_v": np.linspace(4.0, 3.5, n),
        "current_a": np.full(n, -1.0),
        "temperature_c": np.full(n, 25.0),
        "cycler_ah": np.linspace(0.0, -1.0, n),
    })


class TestParseManifestSocInit:
    def test_parses_float_string(self):
        assert parse_manifest_soc_init(_entry(soc_init_str="1.0")) == 1.0

    def test_empty_returns_none(self):
        assert parse_manifest_soc_init(_entry(soc_init_str="")) is None


class TestDetectAdapterQC:
    def test_duplicate_timestamp_warn(self):
        df = _valid_df()
        df.loc[df.index[-1], "time_s"] = df["time_s"].iloc[-2]
        qc = QCResult()
        detect_adapter_qc(df, _entry(), qc)
        assert qc.duplicate_timestamp_count == 1
        assert qc.status == "WARN"

    def test_negative_dt_fail(self):
        df = _valid_df()
        df.loc[df.index[5], "time_s"] = df["time_s"].iloc[4] - 1.0
        qc = QCResult()
        detect_adapter_qc(df, _entry(), qc)
        assert qc.status == "FAIL"


class TestMeasurementsToDataframe:
    def test_missing_field_records_fail(self):
        meas = {"Time": np.arange(5.0), "Voltage": np.ones(5)}
        qc = QCResult()
        df = measurements_to_dataframe(meas, qc)
        assert len(df) == 5
        assert qc.status == "FAIL"


class TestBuildAdapterSummary:
    def _record(self, status: str = "PASS") -> SequenceRecord:
        qc = QCResult()
        if status == "WARN":
            qc.add_warning("warn")
        elif status == "FAIL":
            qc.add_failure("fail")
        return SequenceRecord(
            data=_valid_df(),
            metadata={
                "dataset": "lg_hg2",
                "battery": "lg_hg2",
                "source_file": "data/a.mat",
                "source_sha256": "abc",
                "sequence_id": "lg_hg2_25c_a",
                "profile": "Mixed1",
                "paper_profile": "Mix1",
                "file_type": "drive_cycle",
                "split": "full_train",
                "temperature_nominal_c": 25.0,
                "soc_init": 1.0,
                "soc_anchor_source": "preceding_full_charge",
                "capacity_hint_ah": 3.0,
            },
            qc=qc,
        )

    def test_expected_equals_loaded_plus_skipped(self):
        records = [self._record()]
        stats = AdapterLoadStats(
            split="full_train",
            expected_manifest_entries=2,
            loaded_records=1,
            skipped_manifest_failures=1,
        )
        summary = build_adapter_summary(records, split="full_train", stats=stats)
        assert summary.expected_manifest_entries == 2
        assert summary.loaded_records == 1
        assert summary.skipped_manifest_failures == 1
        assert summary.fail_count == 0
