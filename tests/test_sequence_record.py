"""Tests for SequenceRecord — the standard adapter output container."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from battery_benchmark.data.adapter_qc import (
    AdapterSchemaError,
    InvalidAdapterSplitError,
    InvalidFileTypeError,
    MissingDataColumnError,
    MissingMetadataFieldError,
    QCResult,
)
from battery_benchmark.data.schema import REQUIRED_DATA_COLUMNS, REQUIRED_METADATA_FIELDS
from battery_benchmark.data.sequence_record import SequenceRecord


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

def _make_data(n: int = 20) -> pd.DataFrame:
    """Return a minimal valid data DataFrame."""
    return pd.DataFrame({
        "time_s": np.arange(n, dtype=float),
        "timestamp": [f"2019-03-18 16:{i:02d}:00" for i in range(n)],
        "voltage_v": np.linspace(4.185, 3.354, n),
        "current_a": np.full(n, -2.798),
        "temperature_c": np.full(n, 25.0),
        "cycler_ah": np.linspace(0.0, -2.798, n),
    })


def _make_metadata(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid metadata dict, with optional overrides."""
    base: dict[str, Any] = {
        "dataset": "samsung_30t",
        "battery": "Samsung_INR21700-30T",
        "source_file": "data/raw/Samsung_30T/25degC/03-18-19_16.26 734_UDDS_25degC_IN21700_30T.mat",
        "source_sha256": "abc123",
        "sequence_id": "samsung_30t_25c_UDDS_734",
        "profile": "UDDS",
        "paper_profile": "UDDS",
        "file_type": "drive_cycle",
        "record_role": "model_input",
        "split": "test",
        "temperature_nominal_c": 25.0,
        "soc_init": 1.0,
        "soc_anchor_source": "preceding_full_charge",
        "capacity_hint_ah": 3.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Valid construction
# ---------------------------------------------------------------------------

class TestSequenceRecordValidConstruction:
    def test_creates_successfully_with_valid_inputs(self):
        rec = SequenceRecord(data=_make_data(), metadata=_make_metadata())
        assert isinstance(rec, SequenceRecord)

    def test_n_rows_correct(self):
        rec = SequenceRecord(data=_make_data(n=15), metadata=_make_metadata())
        assert rec.n_rows == 15

    def test_columns_property(self):
        rec = SequenceRecord(data=_make_data(), metadata=_make_metadata())
        assert set(REQUIRED_DATA_COLUMNS).issubset(set(rec.columns))

    def test_has_required_columns_true(self):
        rec = SequenceRecord(data=_make_data(), metadata=_make_metadata())
        assert rec.has_required_columns is True

    def test_qc_defaults_to_pass(self):
        rec = SequenceRecord(data=_make_data(), metadata=_make_metadata())
        assert rec.qc.status == "PASS"

    def test_custom_qc_accepted(self):
        qc = QCResult()
        qc.add_warning("duplicate_timestamp(dt=0 count:1)")
        rec = SequenceRecord(data=_make_data(), metadata=_make_metadata(), qc=qc)
        assert rec.qc.status == "WARN"

    def test_soc_init_none_allowed(self):
        rec = SequenceRecord(data=_make_data(), metadata=_make_metadata(soc_init=None))
        assert rec.metadata["soc_init"] is None

    def test_capacity_hint_none_allowed(self):
        rec = SequenceRecord(data=_make_data(), metadata=_make_metadata(capacity_hint_ah=None))
        assert rec.metadata["capacity_hint_ah"] is None

    def test_paper_profile_empty_string_allowed(self):
        rec = SequenceRecord(data=_make_data(), metadata=_make_metadata(paper_profile=""))
        assert rec.metadata["paper_profile"] == ""

    def test_all_valid_splits_accepted(self):
        from battery_benchmark.data.schema import ALLOWED_ADAPTER_SPLITS
        for split in ALLOWED_ADAPTER_SPLITS:
            rec = SequenceRecord(data=_make_data(), metadata=_make_metadata(split=split))
            assert rec.metadata["split"] == split

    def test_all_valid_file_types_accepted(self):
        from battery_benchmark.data.schema import ALLOWED_FILE_TYPES
        for ft in ALLOWED_FILE_TYPES:
            rec = SequenceRecord(data=_make_data(), metadata=_make_metadata(file_type=ft))
            assert rec.metadata["file_type"] == ft


# ---------------------------------------------------------------------------
# Data type validation
# ---------------------------------------------------------------------------

class TestSequenceRecordDataType:
    def test_non_dataframe_raises_type_error(self):
        with pytest.raises(TypeError, match="DataFrame"):
            SequenceRecord(data=[[1, 2], [3, 4]], metadata=_make_metadata())  # type: ignore[arg-type]

    def test_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="DataFrame"):
            SequenceRecord(data={"a": [1, 2]}, metadata=_make_metadata())  # type: ignore[arg-type]

    def test_numpy_array_raises_type_error(self):
        with pytest.raises(TypeError, match="DataFrame"):
            SequenceRecord(data=np.zeros((5, 6)), metadata=_make_metadata())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Missing data columns
# ---------------------------------------------------------------------------

class TestSequenceRecordMissingColumns:
    @pytest.mark.parametrize("col", REQUIRED_DATA_COLUMNS)
    def test_missing_single_required_column_raises(self, col: str):
        df = _make_data().drop(columns=[col])
        with pytest.raises(MissingDataColumnError) as exc_info:
            SequenceRecord(data=df, metadata=_make_metadata())
        assert col in str(exc_info.value)

    def test_error_message_lists_all_missing_columns(self):
        df = _make_data().drop(columns=["voltage_v", "current_a"])
        with pytest.raises(MissingDataColumnError) as exc_info:
            SequenceRecord(data=df, metadata=_make_metadata())
        msg = str(exc_info.value)
        assert "voltage_v" in msg
        assert "current_a" in msg


# ---------------------------------------------------------------------------
# Missing metadata fields
# ---------------------------------------------------------------------------

class TestSequenceRecordMissingMetadata:
    @pytest.mark.parametrize("field_name", sorted(REQUIRED_METADATA_FIELDS))
    def test_missing_single_required_field_raises(self, field_name: str):
        meta = _make_metadata()
        del meta[field_name]
        with pytest.raises(MissingMetadataFieldError) as exc_info:
            SequenceRecord(data=_make_data(), metadata=meta)
        assert field_name in str(exc_info.value)


# ---------------------------------------------------------------------------
# Invalid metadata values
# ---------------------------------------------------------------------------

class TestSequenceRecordInvalidMetadataValues:
    def test_invalid_split_raises(self):
        with pytest.raises(InvalidAdapterSplitError):
            SequenceRecord(data=_make_data(), metadata=_make_metadata(split="train"))

    def test_invalid_file_type_raises(self):
        with pytest.raises(InvalidFileTypeError):
            SequenceRecord(data=_make_data(), metadata=_make_metadata(file_type="battery_test"))

    def test_empty_source_file_raises(self):
        with pytest.raises(AdapterSchemaError, match="source_file"):
            SequenceRecord(data=_make_data(), metadata=_make_metadata(source_file=""))

    def test_whitespace_source_file_raises(self):
        with pytest.raises(AdapterSchemaError, match="source_file"):
            SequenceRecord(data=_make_data(), metadata=_make_metadata(source_file="   "))

    def test_source_file_as_list_raises(self):
        with pytest.raises(AdapterSchemaError, match="source_file"):
            SequenceRecord(data=_make_data(), metadata=_make_metadata(source_file=["file1", "file2"]))

    def test_empty_sequence_id_raises(self):
        with pytest.raises(AdapterSchemaError, match="sequence_id"):
            SequenceRecord(data=_make_data(), metadata=_make_metadata(sequence_id=""))

    def test_whitespace_sequence_id_raises(self):
        with pytest.raises(AdapterSchemaError, match="sequence_id"):
            SequenceRecord(data=_make_data(), metadata=_make_metadata(sequence_id="  "))

    def test_sequence_id_as_int_raises(self):
        with pytest.raises(AdapterSchemaError, match="sequence_id"):
            SequenceRecord(data=_make_data(), metadata=_make_metadata(sequence_id=42))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Immutability of input DataFrame
# ---------------------------------------------------------------------------

class TestSequenceRecordDataImmutability:
    def test_does_not_modify_input_dataframe(self):
        original = _make_data()
        original_cols = list(original.columns)
        original_values = original["voltage_v"].copy()

        rec = SequenceRecord(data=original, metadata=_make_metadata())
        # Mutate the internal copy
        rec.data["voltage_v"] = 0.0

        # Caller's DataFrame is untouched
        assert list(original.columns) == original_cols
        pd.testing.assert_series_equal(original["voltage_v"], original_values)

    def test_internal_data_is_a_copy(self):
        original = _make_data()
        rec = SequenceRecord(data=original, metadata=_make_metadata())
        assert rec.data is not original


# ---------------------------------------------------------------------------
# Mutable default isolation between instances
# ---------------------------------------------------------------------------

class TestSequenceRecordMutableDefaults:
    def test_qc_issues_not_shared_between_instances(self):
        r1 = SequenceRecord(data=_make_data(), metadata=_make_metadata())
        r2 = SequenceRecord(data=_make_data(), metadata=_make_metadata())
        r1.qc.add_warning("only in r1")
        assert r2.qc.issues == []

    def test_metadata_dicts_independent(self):
        m1 = _make_metadata()
        m2 = _make_metadata()
        r1 = SequenceRecord(data=_make_data(), metadata=m1)
        r2 = SequenceRecord(data=_make_data(), metadata=m2)
        r1.metadata["extra_key"] = "r1_only"
        assert "extra_key" not in r2.metadata


# ---------------------------------------------------------------------------
# to_summary_dict
# ---------------------------------------------------------------------------

class TestSequenceRecordSummary:
    def test_summary_dict_keys(self):
        rec = SequenceRecord(data=_make_data(n=30), metadata=_make_metadata())
        s = rec.to_summary_dict()
        for key in ("dataset", "battery", "source_file", "sequence_id",
                    "profile", "paper_profile", "file_type", "split",
                    "n_rows", "columns", "qc_status"):
            assert key in s, f"Missing key {key!r} in to_summary_dict()"

    def test_summary_n_rows_matches(self):
        rec = SequenceRecord(data=_make_data(n=42), metadata=_make_metadata())
        assert rec.to_summary_dict()["n_rows"] == 42

    def test_summary_qc_status_matches(self):
        rec = SequenceRecord(data=_make_data(), metadata=_make_metadata())
        assert rec.to_summary_dict()["qc_status"] == "PASS"

    def test_summary_columns_list(self):
        rec = SequenceRecord(data=_make_data(), metadata=_make_metadata())
        cols = rec.to_summary_dict()["columns"]
        assert isinstance(cols, list)
        assert "voltage_v" in cols


# ---------------------------------------------------------------------------
# validate() method can be called explicitly
# ---------------------------------------------------------------------------

def test_validate_called_explicitly_passes_on_valid_record():
    rec = SequenceRecord(data=_make_data(), metadata=_make_metadata())
    rec.validate()  # should not raise
