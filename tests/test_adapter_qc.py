"""Tests for QCResult and custom adapter exceptions."""

from __future__ import annotations

import pytest

from battery_benchmark.data.adapter_qc import (
    ALLOWED_QC_STATUS,
    QC_FAIL,
    QC_PASS,
    QC_WARN,
    AdapterSchemaError,
    InvalidAdapterSplitError,
    InvalidFileTypeError,
    InvalidQCStatusError,
    MissingDataColumnError,
    MissingMetadataFieldError,
    QCResult,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestQCResultConstruction:
    def test_default_status_is_pass(self):
        qc = QCResult()
        assert qc.status == QC_PASS

    def test_empty_lists_by_default(self):
        qc = QCResult()
        assert qc.issues == []
        assert qc.notes == []

    def test_zero_counts_by_default(self):
        qc = QCResult()
        assert qc.duplicate_timestamp_count == 0
        assert qc.negative_time_step_count == 0
        assert qc.nan_count == 0
        assert qc.inf_count == 0

    def test_ah_reset_false_by_default(self):
        qc = QCResult()
        assert qc.ah_reset_detected is False

    def test_explicit_warn_status(self):
        qc = QCResult(status=QC_WARN)
        assert qc.status == QC_WARN

    def test_explicit_fail_status(self):
        qc = QCResult(status=QC_FAIL)
        assert qc.status == QC_FAIL

    def test_nonzero_counts_accepted(self):
        qc = QCResult(nan_count=5, duplicate_timestamp_count=3)
        assert qc.nan_count == 5
        assert qc.duplicate_timestamp_count == 3

    def test_invalid_status_raises(self):
        with pytest.raises(InvalidQCStatusError):
            QCResult(status="MAYBE")

    def test_negative_nan_count_raises(self):
        with pytest.raises(ValueError, match="nan_count"):
            QCResult(nan_count=-1)

    def test_negative_duplicate_count_raises(self):
        with pytest.raises(ValueError, match="duplicate_timestamp_count"):
            QCResult(duplicate_timestamp_count=-1)

    def test_negative_negative_time_step_count_raises(self):
        with pytest.raises(ValueError, match="negative_time_step_count"):
            QCResult(negative_time_step_count=-5)

    def test_negative_inf_count_raises(self):
        with pytest.raises(ValueError, match="inf_count"):
            QCResult(inf_count=-2)


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

class TestQCResultTransitions:
    def test_add_warning_sets_warn(self):
        qc = QCResult()
        qc.add_warning("some warning")
        assert qc.status == QC_WARN

    def test_add_warning_appends_to_issues(self):
        qc = QCResult()
        qc.add_warning("w1")
        qc.add_warning("w2")
        assert qc.issues == ["w1", "w2"]

    def test_add_failure_sets_fail(self):
        qc = QCResult()
        qc.add_failure("bad thing")
        assert qc.status == QC_FAIL

    def test_add_failure_appends_to_issues(self):
        qc = QCResult()
        qc.add_failure("f1")
        assert "f1" in qc.issues

    def test_fail_not_downgraded_by_subsequent_warning(self):
        qc = QCResult()
        qc.add_failure("hard error")
        qc.add_warning("soft warning added after failure")
        assert qc.status == QC_FAIL

    def test_multiple_warnings_stay_warn(self):
        qc = QCResult()
        qc.add_warning("w1")
        qc.add_warning("w2")
        assert qc.status == QC_WARN

    def test_warn_then_fail_becomes_fail(self):
        qc = QCResult()
        qc.add_warning("first a warning")
        qc.add_failure("then a failure")
        assert qc.status == QC_FAIL

    def test_add_note_does_not_change_status(self):
        qc = QCResult()
        qc.add_note("just informational")
        assert qc.status == QC_PASS
        assert qc.notes == ["just informational"]

    def test_has_failures_false_on_pass(self):
        qc = QCResult()
        assert qc.has_failures is False

    def test_has_failures_false_on_warn(self):
        qc = QCResult()
        qc.add_warning("w")
        assert qc.has_failures is False

    def test_has_failures_true_on_fail(self):
        qc = QCResult()
        qc.add_failure("f")
        assert qc.has_failures is True


# ---------------------------------------------------------------------------
# Mutable default isolation
# ---------------------------------------------------------------------------

class TestQCResultMutableDefaults:
    def test_issues_not_shared_between_instances(self):
        a = QCResult()
        b = QCResult()
        a.add_warning("only in a")
        assert b.issues == []

    def test_notes_not_shared_between_instances(self):
        a = QCResult()
        b = QCResult()
        a.add_note("note in a")
        assert b.notes == []


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestQCResultToDict:
    def test_to_dict_keys(self):
        qc = QCResult()
        d = qc.to_dict()
        assert set(d) == {
            "status", "issues", "notes",
            "duplicate_timestamp_count", "negative_time_step_count",
            "nan_count", "inf_count", "ah_reset_detected",
        }

    def test_to_dict_values_are_copies(self):
        qc = QCResult()
        qc.add_warning("w")
        d = qc.to_dict()
        d["issues"].append("mutated")
        assert len(qc.issues) == 1  # original unchanged

    def test_to_dict_status_matches(self):
        qc = QCResult()
        qc.add_failure("f")
        assert qc.to_dict()["status"] == QC_FAIL


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------

class TestExceptions:
    def test_missing_data_column_error_message(self):
        exc = MissingDataColumnError(["voltage_v", "current_a"])
        assert "voltage_v" in str(exc)
        assert "current_a" in str(exc)

    def test_missing_metadata_field_error_message(self):
        exc = MissingMetadataFieldError(["dataset", "battery"])
        assert "dataset" in str(exc)
        assert "battery" in str(exc)

    def test_invalid_qc_status_error_message(self):
        exc = InvalidQCStatusError("UNKNOWN")
        assert "UNKNOWN" in str(exc)

    def test_adapter_schema_error_is_value_error(self):
        assert issubclass(AdapterSchemaError, ValueError)

    def test_missing_data_column_is_adapter_schema_error(self):
        assert issubclass(MissingDataColumnError, AdapterSchemaError)

    def test_missing_metadata_is_adapter_schema_error(self):
        assert issubclass(MissingMetadataFieldError, AdapterSchemaError)

    def test_invalid_adapter_split_error(self):
        from battery_benchmark.data.schema import ALLOWED_ADAPTER_SPLITS
        exc = InvalidAdapterSplitError("bad_split", ALLOWED_ADAPTER_SPLITS)
        assert "bad_split" in str(exc)

    def test_invalid_file_type_error(self):
        from battery_benchmark.data.schema import ALLOWED_FILE_TYPES
        exc = InvalidFileTypeError("bad_type", ALLOWED_FILE_TYPES)
        assert "bad_type" in str(exc)


# ---------------------------------------------------------------------------
# ALLOWED_QC_STATUS constant
# ---------------------------------------------------------------------------

def test_allowed_qc_status_contains_all_three():
    assert QC_PASS in ALLOWED_QC_STATUS
    assert QC_WARN in ALLOWED_QC_STATUS
    assert QC_FAIL in ALLOWED_QC_STATUS
