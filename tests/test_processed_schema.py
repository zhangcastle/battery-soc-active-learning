"""Tests for adapter-level schema constants and enums in schema.py."""

from __future__ import annotations

import pytest

from battery_benchmark.data.schema import (
    ALLOWED_ADAPTER_SPLITS,
    ALLOWED_FILE_TYPES,
    ALLOWED_RECORD_ROLES,
    AdapterSplit,
    FileType,
    RecordRole,
    REQUIRED_DATA_COLUMNS,
    REQUIRED_METADATA_FIELDS,
    get_schema_info,
)


# ---------------------------------------------------------------------------
# REQUIRED_DATA_COLUMNS
# ---------------------------------------------------------------------------

class TestRequiredDataColumns:
    def test_is_list(self):
        assert isinstance(REQUIRED_DATA_COLUMNS, list)

    def test_contains_expected_columns(self):
        expected = {"time_s", "timestamp", "voltage_v", "current_a", "temperature_c", "cycler_ah"}
        assert expected.issubset(set(REQUIRED_DATA_COLUMNS))

    def test_no_duplicates(self):
        assert len(REQUIRED_DATA_COLUMNS) == len(set(REQUIRED_DATA_COLUMNS))

    def test_all_strings(self):
        assert all(isinstance(c, str) for c in REQUIRED_DATA_COLUMNS)


# ---------------------------------------------------------------------------
# REQUIRED_METADATA_FIELDS
# ---------------------------------------------------------------------------

class TestRequiredMetadataFields:
    def test_is_frozenset(self):
        assert isinstance(REQUIRED_METADATA_FIELDS, frozenset)

    def test_contains_expected_fields(self):
        expected = {
            "dataset", "battery", "source_file", "source_sha256",
            "sequence_id", "profile", "paper_profile", "file_type",
            "record_role", "split", "temperature_nominal_c", "soc_init",
            "soc_anchor_source", "capacity_hint_ah",
        }
        assert expected.issubset(REQUIRED_METADATA_FIELDS)

    def test_all_strings(self):
        assert all(isinstance(f, str) for f in REQUIRED_METADATA_FIELDS)

    def test_not_empty(self):
        assert len(REQUIRED_METADATA_FIELDS) > 0


# ---------------------------------------------------------------------------
# AdapterSplit enum
# ---------------------------------------------------------------------------

class TestAdapterSplitEnum:
    def test_full_train_value(self):
        assert AdapterSplit.FULL_TRAIN.value == "full_train"

    def test_partial_train_value(self):
        assert AdapterSplit.PARTIAL_TRAIN.value == "partial_train"

    def test_test_value(self):
        assert AdapterSplit.TEST.value == "test"

    def test_exclude_value(self):
        assert AdapterSplit.EXCLUDE.value == "exclude"

    def test_unspecified_value(self):
        assert AdapterSplit.UNSPECIFIED.value == "unspecified"

    def test_all_values_in_allowed_set(self):
        for member in AdapterSplit:
            assert member.value in ALLOWED_ADAPTER_SPLITS

    def test_allowed_adapter_splits_is_frozenset(self):
        assert isinstance(ALLOWED_ADAPTER_SPLITS, frozenset)

    def test_invalid_string_not_in_allowed(self):
        assert "train" not in ALLOWED_ADAPTER_SPLITS  # benchmark-level value
        assert "val" not in ALLOWED_ADAPTER_SPLITS

    def test_adapter_split_is_str_subclass(self):
        # AdapterSplit(str, Enum) means values are str-comparable
        assert AdapterSplit.TEST == "test"


# ---------------------------------------------------------------------------
# FileType enum
# ---------------------------------------------------------------------------

class TestFileTypeEnum:
    def test_drive_cycle_value(self):
        assert FileType.DRIVE_CYCLE.value == "drive_cycle"

    def test_charge_value(self):
        assert FileType.CHARGE.value == "charge"

    def test_characterization_value(self):
        assert FileType.CHARACTERIZATION.value == "characterization"

    def test_other_value(self):
        assert FileType.OTHER.value == "other"

    def test_unknown_value(self):
        assert FileType.UNKNOWN.value == "unknown"

    def test_all_values_in_allowed_set(self):
        for member in FileType:
            assert member.value in ALLOWED_FILE_TYPES

    def test_allowed_file_types_is_frozenset(self):
        assert isinstance(ALLOWED_FILE_TYPES, frozenset)

    def test_file_type_is_str_subclass(self):
        assert FileType.CHARGE == "charge"


# ---------------------------------------------------------------------------
# Backward compatibility — existing schema symbols still present
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_columns_class_unchanged(self):
        from battery_benchmark.data.schema import COLUMNS
        assert COLUMNS.VOLTAGE == "voltage"
        assert COLUMNS.CURRENT == "current"
        assert COLUMNS.TEMPERATURE == "temperature"
        assert COLUMNS.SPLIT == "split"

    def test_required_cols_unchanged(self):
        from battery_benchmark.data.schema import REQUIRED_COLS, COLUMNS
        assert COLUMNS.VOLTAGE in REQUIRED_COLS
        assert COLUMNS.CURRENT in REQUIRED_COLS

    def test_split_enum_unchanged(self):
        from battery_benchmark.data.schema import Split
        assert Split.TRAIN.value == "train"
        assert Split.TEST.value == "test"
        assert Split.VAL.value == "val"
        assert Split.UNSPLIT.value == "unsplit"

    def test_label_source_enum_unchanged(self):
        from battery_benchmark.data.schema import LabelSource
        assert LabelSource.UNAVAILABLE.value == "unavailable"

    def test_validate_cycle_still_works(self):
        import numpy as np
        import pandas as pd
        from battery_benchmark.data.schema import COLUMNS, validate_cycle
        df = pd.DataFrame({
            COLUMNS.TIMESTAMP: np.arange(5, dtype=float),
            COLUMNS.VOLTAGE: np.ones(5) * 3.7,
            COLUMNS.CURRENT: np.ones(5) * -1.0,
            COLUMNS.TEMPERATURE: np.ones(5) * 25.0,
            COLUMNS.CELL_ID: "c1",
            COLUMNS.CYCLE_ID: 0,
            COLUMNS.DOMAIN_ID: "src",
            COLUMNS.PROFILE_NAME: "UDDS",
            COLUMNS.SPLIT: "train",
        })
        result = validate_cycle(df)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# get_schema_info includes adapter fields
# ---------------------------------------------------------------------------

def test_get_schema_info_includes_adapter_fields():
    info = get_schema_info()
    assert "adapter_splits" in info
    assert "file_types" in info
    assert "required_data_columns" in info
    assert "full_train" in info["adapter_splits"]
    assert "drive_cycle" in info["file_types"]
    assert "model_input" in info["record_roles"]
    assert "voltage_v" in info["required_data_columns"]


class TestRecordRoleEnum:
    def test_model_input_value(self):
        assert RecordRole.MODEL_INPUT.value == "model_input"

    def test_soc_anchor_value(self):
        assert RecordRole.SOC_ANCHOR.value == "soc_anchor"

    def test_excluded_value(self):
        assert RecordRole.EXCLUDED.value == "excluded"

    def test_all_values_in_allowed_set(self):
        for member in RecordRole:
            assert member.value in ALLOWED_RECORD_ROLES

    def test_allowed_record_roles_is_frozenset(self):
        assert isinstance(ALLOWED_RECORD_ROLES, frozenset)
