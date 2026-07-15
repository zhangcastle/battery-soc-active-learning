"""Tests for shared manifest+mat adapter utilities."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from battery_benchmark.data.adapter_common import (
    AdapterLoadStats,
    build_adapter_summary,
    detect_adapter_qc,
    measurements_to_dataframe,
    parse_manifest_soc_init,
    validate_soc_metadata,
)
from battery_benchmark.data.adapter_qc import AdapterSchemaError, QCResult
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
        record_role="model_input",
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
    @pytest.mark.parametrize("text,expected", [
        ("", None),
        ("0", 0.0),
        ("0.0", 0.0),
        ("0.5", 0.5),
        ("1", 1.0),
        ("1.0", 1.0),
    ])
    def test_valid_values(self, text, expected):
        assert parse_manifest_soc_init(_entry(soc_init_str=text)) == expected

    @pytest.mark.parametrize("text", ["1.O", "full", "nan", "inf", "-0.1", "1.1"])
    def test_invalid_values_raise(self, text):
        with pytest.raises(AdapterSchemaError, match="soc_init"):
            parse_manifest_soc_init(_entry(soc_init_str=text, filename=f"{text}.mat"))

    def test_nan_float_string_fails(self):
        with pytest.raises(AdapterSchemaError, match="soc_init"):
            parse_manifest_soc_init(_entry(soc_init_str="NaN"))

    def test_inf_float_string_fails(self):
        with pytest.raises(AdapterSchemaError, match="soc_init"):
            parse_manifest_soc_init(_entry(soc_init_str="+Inf"))


class TestValidateSocMetadata:
    def test_model_input_complete_succeeds(self):
        validate_soc_metadata(
            _entry(record_role="model_input"),
            "full_train",
            1.0,
            "preceding_full_charge",
        )

    def test_model_input_missing_soc_init_fails(self):
        with pytest.raises(AdapterSchemaError, match="soc_init"):
            validate_soc_metadata(
                _entry(record_role="model_input"),
                "full_train",
                None,
                "preceding_full_charge",
            )

    def test_model_input_missing_anchor_fails(self):
        with pytest.raises(AdapterSchemaError, match="soc_anchor_source"):
            validate_soc_metadata(
                _entry(record_role="model_input"),
                "full_train",
                1.0,
                "",
            )

    def test_model_input_not_applicable_anchor_fails(self):
        with pytest.raises(AdapterSchemaError, match="not_applicable"):
            validate_soc_metadata(
                _entry(record_role="model_input"),
                "full_train",
                1.0,
                "not_applicable",
            )

    @pytest.mark.parametrize("soc_init", [float("nan"), float("inf"), -0.1, 1.1])
    def test_model_input_invalid_soc_init_range_fails(self, soc_init):
        with pytest.raises(AdapterSchemaError, match="soc_init"):
            validate_soc_metadata(
                _entry(record_role="model_input"),
                "full_train",
                soc_init,
                "preceding_full_charge",
            )

    @pytest.mark.parametrize("soc_init", [0.0, 0.5, 1.0])
    def test_model_input_valid_soc_init_succeeds(self, soc_init):
        validate_soc_metadata(
            _entry(record_role="model_input"),
            "full_train",
            soc_init,
            "preceding_full_charge",
        )

    def test_model_input_charge_skips_soc_metadata_requirements(self):
        validate_soc_metadata(
            _entry(record_role="model_input", file_type="charge"),
            "full_train",
            None,
            "",
        )

    def test_soc_anchor_allows_none_soc_init(self):
        validate_soc_metadata(
            _entry(record_role="soc_anchor", file_type="charge"),
            "full_train",
            None,
            "not_applicable",
        )

    def test_excluded_allows_empty_anchor(self):
        validate_soc_metadata(
            _entry(record_role="excluded", exclude=1, full_train=0),
            "exclude",
            None,
            "",
        )


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
    def _record(self, status: str = "PASS", split: str = "full_train") -> SequenceRecord:
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
                "record_role": "model_input",
                "split": split,
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

    def test_consistent_split_succeeds_full_train(self):
        records = [self._record(split="full_train")]
        summary = build_adapter_summary(records, split="full_train", stats=None)
        assert summary.split == "full_train"

    def test_consistent_split_succeeds_test(self):
        records = [self._record(split="test")]
        summary = build_adapter_summary(records, split="test", stats=None)
        assert summary.split == "test"

    def test_mismatched_split_fails(self):
        records = [self._record(split="test")]
        with pytest.raises(AdapterSchemaError, match="full_train"):
            build_adapter_summary(records, split="full_train", stats=None)

    def test_mixed_splits_fails(self):
        records = [
            self._record(split="full_train"),
            self._record(split="test"),
        ]
        with pytest.raises(AdapterSchemaError, match="split"):
            build_adapter_summary(records, split="full_train", stats=None)

    def test_empty_records_succeeds(self):
        summary = build_adapter_summary([], split="full_train", stats=None)
        assert summary.loaded_records == 0
        assert summary.split == "full_train"
