"""Tests for the generic manifest CSV loader."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from battery_benchmark.data.manifest_loader import (
    REQUIRED_MANIFEST_COLUMNS,
    SUPPORTED_SPLITS,
    ALLOWED_QC_STATUSES,
    ManifestEntry,
    ManifestError,
    MissingManifestColumnError,
    InvalidManifestRowError,
    UnsupportedManifestSplitError,
    entry_manifest_splits,
    load_manifest,
    select_manifest_entries,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_row(**overrides) -> dict:
    """Return a minimal valid manifest row dict."""
    row = {
        "battery": "panasonic_18650pf",
        "temperature_c": "25",
        "filename": "file_A.mat",
        "relative_path": "data/raw/Pan/file_A.mat",
        "profile": "Cycle_1",
        "paper_profile": "Mix1",
        "file_type": "drive_cycle",
        "full_train": "1",
        "partial_train": "1",
        "test": "0",
        "exclude": "0",
        "qc_status": "PASS",
        "qc_notes": "",
    }
    row.update(overrides)
    return row


def _write_csv(tmp_path: Path, rows: list[dict], name: str = "manifest.csv") -> Path:
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Test 1 — Normal manifest reads successfully
# ---------------------------------------------------------------------------

class TestLoadManifestSuccess:
    def test_returns_list_of_manifest_entries(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row()])
        entries = load_manifest(p)
        assert isinstance(entries, list)
        assert len(entries) == 1
        assert isinstance(entries[0], ManifestEntry)

    def test_battery_field_parsed(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(battery="my_battery")])
        assert load_manifest(p)[0].battery == "my_battery"

    def test_temperature_c_float(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(temperature_c="25")])
        assert load_manifest(p)[0].temperature_c == 25.0

    def test_multiple_rows_preserved(self, tmp_path):
        rows = [
            _base_row(filename="a.mat", relative_path="data/a.mat"),
            _base_row(filename="b.mat", relative_path="data/b.mat",
                      full_train="0", partial_train="0", test="1"),
        ]
        entries = load_manifest(_write_csv(tmp_path, rows))
        assert len(entries) == 2
        assert entries[0].filename == "a.mat"
        assert entries[1].filename == "b.mat"

    def test_is_selected_property(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(full_train="1")])
        assert load_manifest(p)[0].is_selected is True

    def test_is_excluded_property(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(
            full_train="0", partial_train="0", test="0", exclude="1"
        )])
        assert load_manifest(p)[0].is_excluded is True

    def test_optional_sha256_empty_by_default(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row()])
        assert load_manifest(p)[0].sha256 == ""

    def test_optional_sha256_read_when_present(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(sha256="abcdef1234")])
        assert load_manifest(p)[0].sha256 == "abcdef1234"

    def test_raw_row_contains_original_columns(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row()])
        entry = load_manifest(p)[0]
        assert "battery" in entry.raw_row
        assert "relative_path" in entry.raw_row


# ---------------------------------------------------------------------------
# Test 2 — full_train selection
# ---------------------------------------------------------------------------

class TestSelectFullTrain:
    def test_selects_full_train_rows(self, tmp_path):
        rows = [
            _base_row(filename="train.mat", relative_path="data/train.mat",
                      full_train="1", partial_train="0"),
            _base_row(filename="test.mat", relative_path="data/test.mat",
                      full_train="0", partial_train="0", test="1"),
        ]
        entries = load_manifest(_write_csv(tmp_path, rows))
        selected = select_manifest_entries(entries, "full_train")
        assert len(selected) == 1
        assert selected[0].filename == "train.mat"

    def test_excludes_test_rows(self, tmp_path):
        rows = [_base_row(filename="t.mat", relative_path="d/t.mat",
                          full_train="0", partial_train="0", test="1")]
        entries = load_manifest(_write_csv(tmp_path, rows))
        assert select_manifest_entries(entries, "full_train") == []


# ---------------------------------------------------------------------------
# Test 3 — partial_train selection
# ---------------------------------------------------------------------------

class TestSelectPartialTrain:
    def test_selects_partial_train_rows(self, tmp_path):
        rows = [
            _base_row(filename="mix1.mat", relative_path="d/mix1.mat",
                      full_train="1", partial_train="1"),
            _base_row(filename="mix4.mat", relative_path="d/mix4.mat",
                      full_train="1", partial_train="0"),
        ]
        entries = load_manifest(_write_csv(tmp_path, rows))
        selected = select_manifest_entries(entries, "partial_train")
        assert len(selected) == 1
        assert selected[0].filename == "mix1.mat"


# ---------------------------------------------------------------------------
# Test 4 — test selection
# ---------------------------------------------------------------------------

class TestSelectTest:
    def test_selects_test_rows_only(self, tmp_path):
        rows = [
            _base_row(filename="tr.mat", relative_path="d/tr.mat",
                      full_train="1", partial_train="0"),
            _base_row(filename="ts.mat", relative_path="d/ts.mat",
                      full_train="0", partial_train="0", test="1"),
        ]
        entries = load_manifest(_write_csv(tmp_path, rows))
        selected = select_manifest_entries(entries, "test")
        assert len(selected) == 1
        assert selected[0].filename == "ts.mat"


# ---------------------------------------------------------------------------
# Test 5 — exclude selection
# ---------------------------------------------------------------------------

class TestSelectExclude:
    def test_selects_excluded_rows(self, tmp_path):
        rows = [
            _base_row(filename="used.mat", relative_path="d/used.mat",
                      full_train="1"),
            _base_row(filename="excl.mat", relative_path="d/excl.mat",
                      full_train="0", partial_train="0", test="0", exclude="1"),
        ]
        entries = load_manifest(_write_csv(tmp_path, rows))
        selected = select_manifest_entries(entries, "exclude")
        assert len(selected) == 1
        assert selected[0].filename == "excl.mat"


# ---------------------------------------------------------------------------
# Test 6 — missing required column fails
# ---------------------------------------------------------------------------

class TestMissingColumn:
    @pytest.mark.parametrize("col", sorted(REQUIRED_MANIFEST_COLUMNS))
    def test_missing_required_column_raises(self, tmp_path, col):
        row = _base_row()
        del row[col]
        p = _write_csv(tmp_path, [row])
        with pytest.raises(MissingManifestColumnError) as exc_info:
            load_manifest(p)
        assert col in str(exc_info.value)

    def test_missing_column_error_is_manifest_error(self):
        assert issubclass(MissingManifestColumnError, ManifestError)


# ---------------------------------------------------------------------------
# Test 7 — invalid split raises UnsupportedManifestSplitError
# ---------------------------------------------------------------------------

class TestUnsupportedSplit:
    def test_unknown_split_raises(self, tmp_path):
        entries = load_manifest(_write_csv(tmp_path, [_base_row()]))
        with pytest.raises(UnsupportedManifestSplitError):
            select_manifest_entries(entries, "val")

    def test_error_message_contains_split_name(self, tmp_path):
        entries = load_manifest(_write_csv(tmp_path, [_base_row()]))
        with pytest.raises(UnsupportedManifestSplitError, match="val"):
            select_manifest_entries(entries, "val")


# ---------------------------------------------------------------------------
# Test 8 — exclude=1 and selected split=1 fails
# ---------------------------------------------------------------------------

class TestExcludeConflict:
    def test_exclude_and_full_train_raises(self, tmp_path):
        row = _base_row(full_train="1", exclude="1")
        p = _write_csv(tmp_path, [row])
        with pytest.raises(InvalidManifestRowError, match="exclude"):
            load_manifest(p)

    def test_exclude_and_test_raises(self, tmp_path):
        row = _base_row(full_train="0", partial_train="0", test="1", exclude="1")
        p = _write_csv(tmp_path, [row])
        with pytest.raises(InvalidManifestRowError):
            load_manifest(p)


# ---------------------------------------------------------------------------
# Test 9 — full_train=1 and test=1 fails
# ---------------------------------------------------------------------------

class TestTrainTestOverlap:
    def test_full_train_and_test_raises(self, tmp_path):
        row = _base_row(full_train="1", partial_train="0", test="1")
        p = _write_csv(tmp_path, [row])
        with pytest.raises(InvalidManifestRowError, match="full_train"):
            load_manifest(p)

    def test_partial_train_and_test_raises(self, tmp_path):
        # partial_train=1, test=1 is never allowed regardless of full_train
        row = _base_row(full_train="0", partial_train="1", test="1")
        p = _write_csv(tmp_path, [row])
        with pytest.raises(InvalidManifestRowError):
            load_manifest(p)


# ---------------------------------------------------------------------------
# Test 10 — partial_train without full_train fails
# ---------------------------------------------------------------------------

class TestPartialTrainSubset:
    def test_partial_train_requires_full_train(self, tmp_path):
        row = _base_row(full_train="0", partial_train="1", test="0")
        p = _write_csv(tmp_path, [row])
        with pytest.raises(InvalidManifestRowError, match="partial_train"):
            load_manifest(p)


# ---------------------------------------------------------------------------
# Test 11 — empty relative_path fails
# ---------------------------------------------------------------------------

class TestEmptyPaths:
    def test_empty_relative_path_raises(self, tmp_path):
        row = _base_row(relative_path="")
        p = _write_csv(tmp_path, [row])
        with pytest.raises(InvalidManifestRowError, match="relative_path"):
            load_manifest(p)

    def test_empty_filename_raises(self, tmp_path):
        row = _base_row(filename="")
        p = _write_csv(tmp_path, [row])
        with pytest.raises(InvalidManifestRowError, match="filename"):
            load_manifest(p)


# ---------------------------------------------------------------------------
# Test 12 — row order preserved
# ---------------------------------------------------------------------------

class TestRowOrderPreserved:
    def test_order_matches_csv(self, tmp_path):
        rows = [
            _base_row(filename="first.mat", relative_path="d/first.mat"),
            _base_row(filename="second.mat", relative_path="d/second.mat",
                      full_train="0", partial_train="0", test="1"),
            _base_row(filename="third.mat", relative_path="d/third.mat",
                      full_train="0", partial_train="0", test="0", exclude="1"),
        ]
        entries = load_manifest(_write_csv(tmp_path, rows))
        assert [e.filename for e in entries] == ["first.mat", "second.mat", "third.mat"]

    def test_selection_order_preserved(self, tmp_path):
        rows = [
            _base_row(filename=f"f{i}.mat", relative_path=f"d/f{i}.mat",
                      full_train="1", partial_train="0")
            for i in range(5)
        ]
        entries = load_manifest(_write_csv(tmp_path, rows))
        selected = select_manifest_entries(entries, "full_train")
        assert [e.filename for e in selected] == [f"f{i}.mat" for i in range(5)]


# ---------------------------------------------------------------------------
# Test 13 — "0"/"1" strings correctly parsed
# ---------------------------------------------------------------------------

class TestStringFlags:
    def test_string_zero_parsed_as_0(self, tmp_path):
        row = _base_row(full_train="0", partial_train="0", test="1", exclude="0")
        p = _write_csv(tmp_path, [row])
        e = load_manifest(p)[0]
        assert e.full_train == 0
        assert e.test == 1

    def test_string_one_parsed_as_1(self, tmp_path):
        row = _base_row(full_train="1", partial_train="1")
        p = _write_csv(tmp_path, [row])
        e = load_manifest(p)[0]
        assert e.full_train == 1
        assert e.partial_train == 1

    def test_integer_zero_accepted(self, tmp_path):
        """Pandas may write 0/1 as integers in some versions."""
        rows = [_base_row()]
        df = pd.DataFrame(rows)
        df["full_train"] = 1
        df["partial_train"] = 1
        df["test"] = 0
        df["exclude"] = 0
        p = tmp_path / "int_flags.csv"
        df.to_csv(p, index=False)
        e = load_manifest(p)[0]
        assert e.full_train == 1
        assert e.test == 0


# ---------------------------------------------------------------------------
# Test 14 — manifest file not modified after load
# ---------------------------------------------------------------------------

class TestFileImmutability:
    def test_sha256_unchanged_after_load(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row()])
        sha_before = _sha256_of(p)
        load_manifest(p)
        sha_after = _sha256_of(p)
        assert sha_before == sha_after


# ---------------------------------------------------------------------------
# Error path: non-.csv file
# ---------------------------------------------------------------------------

def test_non_csv_file_raises(tmp_path):
    p = tmp_path / "manifest.txt"
    p.write_text("not a csv")
    with pytest.raises(ManifestError, match=r"\.txt"):
        load_manifest(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "no_such_file.csv")


# ---------------------------------------------------------------------------
# SUPPORTED_SPLITS constant
# ---------------------------------------------------------------------------

def test_supported_splits_contains_expected():
    assert "full_train" in SUPPORTED_SPLITS
    assert "partial_train" in SUPPORTED_SPLITS
    assert "test" in SUPPORTED_SPLITS
    assert "exclude" in SUPPORTED_SPLITS
    assert "val" not in SUPPORTED_SPLITS


# ---------------------------------------------------------------------------
# Strict temperature_c validation
# ---------------------------------------------------------------------------

class TestTemperatureCValidation:
    def test_string_25_parses(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(temperature_c="25")])
        assert load_manifest(p)[0].temperature_c == 25.0

    def test_string_25_0_parses(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(temperature_c="25.0")])
        assert load_manifest(p)[0].temperature_c == 25.0

    def test_negative_20_allowed(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(temperature_c="-20")])
        assert load_manifest(p)[0].temperature_c == -20.0

    def test_empty_string_fails(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(temperature_c="")])
        with pytest.raises(InvalidManifestRowError, match="temperature_c"):
            load_manifest(p)

    def test_non_numeric_fails(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(temperature_c="twenty-five")])
        with pytest.raises(InvalidManifestRowError, match="temperature_c"):
            load_manifest(p)

    def test_nan_string_fails(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(temperature_c="nan")])
        with pytest.raises(InvalidManifestRowError, match="temperature_c"):
            load_manifest(p)

    def test_inf_string_fails(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(temperature_c="inf")])
        with pytest.raises(InvalidManifestRowError, match="temperature_c"):
            load_manifest(p)


# ---------------------------------------------------------------------------
# Strict qc_status validation
# ---------------------------------------------------------------------------

class TestQCStatusValidation:
    def test_pass_uppercase(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(qc_status="PASS")])
        assert load_manifest(p)[0].qc_status == "PASS"

    def test_pass_lowercase_normalised(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(qc_status="pass")])
        assert load_manifest(p)[0].qc_status == "PASS"

    def test_warn_normalised(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(qc_status="warn")])
        assert load_manifest(p)[0].qc_status == "WARN"

    def test_fail_normalised(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(qc_status="fail")])
        assert load_manifest(p)[0].qc_status == "FAIL"

    @pytest.mark.parametrize("bad", ["PAS", "OK", ""])
    def test_invalid_qc_status_fails(self, tmp_path, bad):
        p = _write_csv(tmp_path, [_base_row(qc_status=bad)])
        with pytest.raises(InvalidManifestRowError, match="qc_status"):
            load_manifest(p)


# ---------------------------------------------------------------------------
# Strict file_type validation
# ---------------------------------------------------------------------------

class TestFileTypeValidation:
    @pytest.mark.parametrize("ft", ["drive_cycle", "charge", "characterization", "other", "unknown"])
    def test_allowed_file_types(self, tmp_path, ft):
        p = _write_csv(tmp_path, [_base_row(file_type=ft)])
        assert load_manifest(p)[0].file_type == ft

    def test_empty_file_type_fails(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(file_type="")])
        with pytest.raises(InvalidManifestRowError, match="file_type"):
            load_manifest(p)

    def test_invalid_file_type_fails(self, tmp_path):
        p = _write_csv(tmp_path, [_base_row(file_type="rest_cycle")])
        with pytest.raises(InvalidManifestRowError, match="file_type"):
            load_manifest(p)


# ---------------------------------------------------------------------------
# entry_manifest_splits helper
# ---------------------------------------------------------------------------

class TestEntryManifestSplits:
    def test_full_and_partial_membership(self, tmp_path):
        entry = load_manifest(_write_csv(tmp_path, [_base_row()]))[0]
        assert entry_manifest_splits(entry) == ["full_train", "partial_train"]

    def test_test_only(self, tmp_path):
        row = _base_row(
            full_train="0", partial_train="0", test="1", exclude="0",
            paper_profile="US06", profile="US06",
        )
        entry = load_manifest(_write_csv(tmp_path, [row]))[0]
        assert entry_manifest_splits(entry) == ["test"]

    def test_exclude_only(self, tmp_path):
        row = _base_row(
            full_train="0", partial_train="0", test="0", exclude="1",
            file_type="characterization", paper_profile="",
        )
        entry = load_manifest(_write_csv(tmp_path, [row]))[0]
        assert entry_manifest_splits(entry) == ["exclude"]
