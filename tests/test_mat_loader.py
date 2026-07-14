"""Tests for the generic MATLAB .mat file loader.

All tests create temporary .mat files using ``scipy.io.savemat`` so that no
real battery data is required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import scipy.io

from battery_benchmark.data.mat_loader import (
    MatFileError,
    MatLoadResult,
    MissingMeasurementStructError,
    UnsupportedMatFormatError,
    load_mat_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_mat(tmp_path: Path, **meas_fields) -> Path:
    """Write a minimal .mat file with a ``meas`` struct and return its path."""
    p = tmp_path / "test_data.mat"
    scipy.io.savemat(str(p), {"meas": meas_fields})
    return p


def _default_meas(n: int = 30) -> dict:
    """Return a dict of standard measurement arrays."""
    return {
        "Time": np.arange(n, dtype=float),
        "TimeStamp": np.array(
            [f"2019-03-18 16:{i:02d}:00" for i in range(n)], dtype=object
        ),
        "Voltage": np.linspace(4.185, 3.354, n),
        "Current": np.full(n, -2.798),
        "Ah": np.linspace(0.0, -2.5, n),
        "Battery_Temp_degC": np.full(n, 25.0),
    }


# ---------------------------------------------------------------------------
# Test 1 — Successful load of a normal meas struct
# ---------------------------------------------------------------------------

class TestSuccessfulLoad:
    def test_returns_mat_load_result(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        result = load_mat_file(path)
        assert isinstance(result, MatLoadResult)

    def test_measurements_contains_voltage(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        result = load_mat_file(path)
        assert "Voltage" in result.measurements

    def test_measurements_contains_current(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        result = load_mat_file(path)
        assert "Current" in result.measurements

    def test_all_expected_fields_present(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        result = load_mat_file(path)
        for field in ("Time", "Voltage", "Current", "Ah", "Battery_Temp_degC"):
            assert field in result.measurements, f"Missing field {field!r}"

    def test_backend_is_scipy(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        result = load_mat_file(path)
        assert result.loader_backend == "scipy"

    def test_path_is_resolved_absolute(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        result = load_mat_file(path)
        assert result.path.is_absolute()

    def test_accepts_string_path(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        result = load_mat_file(str(path))
        assert isinstance(result, MatLoadResult)

    def test_root_keys_contains_meas(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        result = load_mat_file(path)
        assert "meas" in result.root_keys

    def test_root_keys_excludes_dunder_keys(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        result = load_mat_file(path)
        assert all(not k.startswith("__") for k in result.root_keys)

    def test_warnings_is_a_list(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        result = load_mat_file(path)
        assert isinstance(result.warnings, list)


# ---------------------------------------------------------------------------
# Test 2 — Shape normalisation: (n,1) and (1,n) both become (n,)
# ---------------------------------------------------------------------------

class TestShapeNormalisation:
    def test_column_vector_flattened(self, tmp_path):
        n = 20
        col_vec = np.arange(n, dtype=float).reshape(n, 1)  # (n, 1)
        path = _make_mat(tmp_path, Time=col_vec, Voltage=col_vec,
                         Current=col_vec, Ah=col_vec, Battery_Temp_degC=col_vec)
        result = load_mat_file(path)
        assert result.measurements["Voltage"].ndim == 1
        assert result.measurements["Voltage"].shape == (n,)

    def test_row_vector_flattened(self, tmp_path):
        n = 20
        row_vec = np.arange(n, dtype=float).reshape(1, n)  # (1, n)
        path = _make_mat(tmp_path, Time=row_vec, Voltage=row_vec,
                         Current=row_vec, Ah=row_vec, Battery_Temp_degC=row_vec)
        result = load_mat_file(path)
        assert result.measurements["Voltage"].ndim == 1
        assert result.measurements["Voltage"].shape == (n,)

    def test_1d_array_unchanged(self, tmp_path):
        n = 25
        arr = np.linspace(4.0, 3.0, n)
        path = _make_mat(tmp_path, Time=np.arange(n, dtype=float),
                         Voltage=arr, Current=arr,
                         Ah=arr, Battery_Temp_degC=arr)
        result = load_mat_file(path)
        assert result.measurements["Voltage"].shape == (n,)

    def test_scalar_field_flattened_to_1d(self, tmp_path):
        n = 5
        arr = np.arange(n, dtype=float)
        path = _make_mat(tmp_path, Time=arr, Voltage=arr,
                         Current=arr, Ah=arr, Battery_Temp_degC=arr,
                         ExtraScalar=np.array([[42.0]]))
        result = load_mat_file(path)
        assert result.measurements["ExtraScalar"].ndim == 1


# ---------------------------------------------------------------------------
# Test 3 — SHA-256
# ---------------------------------------------------------------------------

class TestSHA256:
    def test_sha256_is_nonempty_string(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        result = load_mat_file(path)
        assert isinstance(result.sha256, str)
        assert len(result.sha256) == 64

    def test_sha256_matches_independent_computation(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        result = load_mat_file(path)
        assert result.sha256 == _sha256(path)

    def test_sha256_stable_across_two_loads(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        r1 = load_mat_file(path)
        r2 = load_mat_file(path)
        assert r1.sha256 == r2.sha256


# ---------------------------------------------------------------------------
# Test 4 — Error on missing file
# ---------------------------------------------------------------------------

class TestMissingFile:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist.mat"
        with pytest.raises(FileNotFoundError, match="does_not_exist.mat"):
            load_mat_file(nonexistent)


# ---------------------------------------------------------------------------
# Test 5 — Error on wrong extension
# ---------------------------------------------------------------------------

class TestWrongExtension:
    def test_txt_extension_raises_mat_file_error(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("not a mat file")
        with pytest.raises(MatFileError, match=r"\.txt"):
            load_mat_file(txt_file)

    def test_no_extension_raises_mat_file_error(self, tmp_path):
        no_ext = tmp_path / "data"
        no_ext.write_bytes(b"\x00\x01\x02")
        with pytest.raises(MatFileError):
            load_mat_file(no_ext)

    def test_csv_extension_raises_mat_file_error(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c")
        with pytest.raises(MatFileError):
            load_mat_file(csv_file)


# ---------------------------------------------------------------------------
# Test 6 — Error when meas struct is absent
# ---------------------------------------------------------------------------

class TestMissingMeasStruct:
    def test_missing_meas_raises_missing_struct_error(self, tmp_path):
        path = tmp_path / "no_meas.mat"
        scipy.io.savemat(str(path), {"other_var": np.array([1, 2, 3])})
        with pytest.raises(MissingMeasurementStructError):
            load_mat_file(path)

    def test_error_message_lists_root_keys(self, tmp_path):
        path = tmp_path / "no_meas.mat"
        scipy.io.savemat(str(path), {"temperature": np.array([25.0])})
        with pytest.raises(MissingMeasurementStructError) as exc_info:
            load_mat_file(path)
        assert "temperature" in str(exc_info.value)

    def test_missing_struct_error_is_mat_file_error(self):
        assert issubclass(MissingMeasurementStructError, MatFileError)


# ---------------------------------------------------------------------------
# Test 7 — Partial fields: loader returns what is present
# ---------------------------------------------------------------------------

class TestPartialFields:
    def test_only_voltage_and_current_present(self, tmp_path):
        """Loader should not raise if some standard fields are absent."""
        path = _make_mat(tmp_path, Voltage=np.ones(10), Current=np.zeros(10))
        result = load_mat_file(path)
        assert "Voltage" in result.measurements
        assert "Current" in result.measurements

    def test_missing_fields_emit_warnings_not_errors(self, tmp_path):
        path = _make_mat(tmp_path, Voltage=np.ones(10))
        result = load_mat_file(path)
        # Missing fields like Time, Ah etc. cause warnings, not exceptions
        assert any("Time" in w or "Current" in w or "Ah" in w for w in result.warnings)

    def test_extra_unexpected_fields_preserved(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas(),
                         ExtraField=np.array([1.0, 2.0, 3.0]))
        result = load_mat_file(path)
        assert "ExtraField" in result.measurements


# ---------------------------------------------------------------------------
# Test 8 — TimeStamp as object/string array
# ---------------------------------------------------------------------------

class TestTimestampField:
    def test_timestamp_object_array_loaded(self, tmp_path):
        n = 10
        ts = np.array([f"2019-03-18 16:{i:02d}:00" for i in range(n)], dtype=object)
        meas = {**_default_meas(n), "TimeStamp": ts}
        path = _make_mat(tmp_path, **meas)
        result = load_mat_file(path)
        assert "TimeStamp" in result.measurements

    def test_timestamp_is_1d(self, tmp_path):
        n = 10
        ts = np.array([f"2019-03-18 16:{i:02d}:00" for i in range(n)], dtype=object)
        meas = {**_default_meas(n), "TimeStamp": ts}
        path = _make_mat(tmp_path, **meas)
        result = load_mat_file(path)
        assert result.measurements["TimeStamp"].ndim == 1


# ---------------------------------------------------------------------------
# Test 9 — Input file not modified (SHA-256 before == after load)
# ---------------------------------------------------------------------------

class TestFileImmutability:
    def test_sha256_unchanged_after_load(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        sha_before = _sha256(path)
        load_mat_file(path)
        sha_after = _sha256(path)
        assert sha_before == sha_after, "File was modified during loading!"


# ---------------------------------------------------------------------------
# Test 10 — Two consecutive reads give identical results
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_two_loads_same_sha256(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        r1 = load_mat_file(path)
        r2 = load_mat_file(path)
        assert r1.sha256 == r2.sha256

    def test_two_loads_same_root_keys(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        r1 = load_mat_file(path)
        r2 = load_mat_file(path)
        assert r1.root_keys == r2.root_keys

    def test_two_loads_same_voltage_values(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        r1 = load_mat_file(path)
        r2 = load_mat_file(path)
        np.testing.assert_array_equal(
            r1.measurements["Voltage"], r2.measurements["Voltage"]
        )

    def test_two_loads_same_measurement_keys(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        r1 = load_mat_file(path)
        r2 = load_mat_file(path)
        assert set(r1.measurements) == set(r2.measurements)


# ---------------------------------------------------------------------------
# Test 11 — Mutable defaults not shared between MatLoadResult instances
# ---------------------------------------------------------------------------

class TestMutableDefaults:
    def test_warnings_not_shared_between_results(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        r1 = load_mat_file(path)
        r2 = load_mat_file(path)
        r1.warnings.append("mutated in r1")
        assert "mutated in r1" not in r2.warnings

    def test_measurements_not_shared_between_results(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())
        r1 = load_mat_file(path)
        r2 = load_mat_file(path)
        r1.measurements["Voltage"][0] = 999.0
        assert r2.measurements["Voltage"][0] != 999.0


# ---------------------------------------------------------------------------
# Test 12 — UnsupportedMatFormatError (v7.3 / HDF5)
# ---------------------------------------------------------------------------

class TestUnsupportedFormat:
    def test_unsupported_format_error_is_mat_file_error(self):
        assert issubclass(UnsupportedMatFormatError, MatFileError)

    def test_v73_without_h5py_raises_unsupported(self, tmp_path):
        """Simulate scipy raising NotImplementedError for a v7.3 file."""
        path = _make_mat(tmp_path, **_default_meas())

        def mock_loadmat(*args, **kwargs):
            raise NotImplementedError("Please use HDF reader for matlab v7.3 files")

        # Patch out both scipy and any h5py import
        with patch("battery_benchmark.data.mat_loader.scipy.io.loadmat", mock_loadmat), \
             patch.dict("sys.modules", {"h5py": None}):
            with pytest.raises(UnsupportedMatFormatError):
                load_mat_file(path)

    def test_unsupported_error_message_mentions_h5py(self, tmp_path):
        path = _make_mat(tmp_path, **_default_meas())

        def mock_loadmat(*args, **kwargs):
            raise NotImplementedError("Please use HDF reader for matlab v7.3 files")

        with patch("battery_benchmark.data.mat_loader.scipy.io.loadmat", mock_loadmat), \
             patch.dict("sys.modules", {"h5py": None}):
            with pytest.raises(UnsupportedMatFormatError, match="h5py"):
                load_mat_file(path)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_meas_struct_returns_empty_measurements(self, tmp_path):
        """An empty meas struct should load without raising, with no measurements."""
        path = tmp_path / "empty_meas.mat"
        scipy.io.savemat(str(path), {"meas": {}})
        result = load_mat_file(path)
        # No measurement fields, but also no crash
        assert isinstance(result.measurements, dict)

    def test_mat_file_error_is_os_error(self):
        assert issubclass(MatFileError, OSError)

    def test_wh_and_power_preserved_if_present(self, tmp_path):
        n = 20
        meas = {**_default_meas(n),
                "Wh": np.linspace(0, 10, n),
                "Power": np.linspace(0, 5, n)}
        path = _make_mat(tmp_path, **meas)
        result = load_mat_file(path)
        assert "Wh" in result.measurements
        assert "Power" in result.measurements

    def test_voltage_values_preserved_correctly(self, tmp_path):
        voltage = np.array([4.185, 4.100, 4.050, 3.990, 3.800])
        path = _make_mat(tmp_path, Voltage=voltage, Current=np.zeros(5),
                         Time=np.arange(5, dtype=float), Ah=np.zeros(5),
                         Battery_Temp_degC=np.full(5, 25.0))
        result = load_mat_file(path)
        np.testing.assert_allclose(result.measurements["Voltage"], voltage, rtol=1e-6)
