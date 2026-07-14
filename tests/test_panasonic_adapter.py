"""Tests for PanasonicAdapter — unit tests + integration smoke test.

Unit tests (class ``TestPanasonicAdapterUnit``) create temporary .mat files
and manifest CSVs via ``scipy.io.savemat``; they require NO real battery data.

Integration tests (``@pytest.mark.integration``) load real Panasonic files
from ``data/raw/Panasonic_18650PF/25degC/`` and are skipped when that
directory is absent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io

from battery_benchmark.data.adapter_qc import AdapterSchemaError
from battery_benchmark.data.manifest_loader import SHA256MismatchError
from battery_benchmark.data.panasonic import PanasonicAdapter
from battery_benchmark.data.sequence_record import SequenceRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_meas(n: int = 50, *, neg_time: bool = False, dup_time: bool = False,
               nan_voltage: bool = False, inf_current: bool = False,
               ah_reset: bool = False) -> dict:
    """Build a minimal meas dict for scipy.io.savemat."""
    time = np.arange(n, dtype=float)
    if neg_time:
        time[n // 2] = time[n // 2 - 1] - 1.0   # strictly decreasing
    if dup_time:
        time[-1] = time[-2]                        # duplicate last timestamp

    voltage = np.linspace(4.185, 3.354, n)
    if nan_voltage:
        voltage[5] = np.nan

    current = np.full(n, -2.798)
    if inf_current:
        current[3] = np.inf

    ah = np.linspace(0.0, -2.5, n)
    if ah_reset:
        # Ah goes to -2.5 then drops back to ~0 (cycler reset)
        ah[-1] = -0.001

    return {
        "Time": time,
        "TimeStamp": np.array([f"2019-03-18 16:{i:02d}:00" for i in range(n)], dtype=object),
        "Voltage": voltage,
        "Current": current,
        "Battery_Temp_degC": np.full(n, 25.0),
        "Ah": ah,
    }


def _write_mat(tmp_path: Path, name: str, **meas_override) -> Path:
    p = tmp_path / name
    meas = _make_meas()
    meas.update(meas_override)
    scipy.io.savemat(str(p), {"meas": meas})
    return p


def _write_manifest(tmp_path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    p = tmp_path / "manifest.csv"
    df.to_csv(p, index=False)
    return p


def _manifest_row(mat_path: Path, tmp_path: Path, *,
                  full_train: int = 1,
                  partial_train: int = 0,
                  test: int = 0,
                  exclude: int = 0,
                  sha256: str | None = None,
                  qc_status: str = "PASS",
                  qc_notes: str = "",
                  profile: str = "Cycle_1",
                  paper_profile: str = "Mix1",
                  file_type: str = "drive_cycle",
                  record_role: str | None = None,
                  **overrides) -> dict:
    """Build a minimal manifest row pointing at *mat_path* (absolute → relative)."""
    rel = str(mat_path.relative_to(tmp_path.parent))
    row = {
        "battery": "panasonic_18650pf",
        "temperature_c": "25",
        "filename": mat_path.name,
        "relative_path": rel,
        "profile": profile,
        "paper_profile": paper_profile,
        "file_type": file_type,
        "full_train": str(full_train),
        "partial_train": str(partial_train),
        "test": str(test),
        "exclude": str(exclude),
        "qc_status": qc_status,
        "qc_notes": qc_notes,
        "sha256": sha256 if sha256 is not None else "",
        "pairing_status": "confirmed",
        "soc_anchor_evidence": "SOC_init=1.0",
        "confidence": "high",
        "evidence": "test evidence",
        "notes": "",
    }
    row.update(overrides)
    if record_role is not None:
        row["record_role"] = record_role
    elif "record_role" not in overrides:
        ex = int(row["exclude"])
        ft = row["file_type"]
        selected = int(row["full_train"]) or int(row["partial_train"]) or int(row["test"])
        if ex:
            row["record_role"] = "excluded"
        elif ft == "charge" and selected:
            row["record_role"] = "soc_anchor"
        else:
            row["record_role"] = "model_input"
    return row


def _make_adapter(tmp_path: Path, rows: list[dict]) -> PanasonicAdapter:
    """Create a PanasonicAdapter with raw_root=tmp_path.parent."""
    manifest_path = _write_manifest(tmp_path, rows)
    return PanasonicAdapter(
        raw_root=tmp_path.parent,
        manifest_path=manifest_path,
    )


# ---------------------------------------------------------------------------
# Test 15 — Normal .mat → SequenceRecord
# ---------------------------------------------------------------------------

class TestNormalLoad:
    def test_creates_sequence_record(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        adapter = _make_adapter(tmp_path, [_manifest_row(mat, tmp_path)])
        records = adapter.load_split("full_train")
        assert len(records) == 1
        assert isinstance(records[0], SequenceRecord)

    def test_n_rows_matches_mat(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        adapter = _make_adapter(tmp_path, [_manifest_row(mat, tmp_path)])
        rec = adapter.load_split("full_train")[0]
        assert rec.n_rows == 50

    def test_has_required_columns(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        adapter = _make_adapter(tmp_path, [_manifest_row(mat, tmp_path)])
        rec = adapter.load_split("full_train")[0]
        assert rec.has_required_columns


# ---------------------------------------------------------------------------
# Test 16 — Column mapping correct
# ---------------------------------------------------------------------------

class TestColumnMapping:
    def test_voltage_v_present(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        adapter = _make_adapter(tmp_path, [_manifest_row(mat, tmp_path)])
        rec = adapter.load_split("full_train")[0]
        assert "voltage_v" in rec.data.columns

    def test_current_a_present(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        adapter = _make_adapter(tmp_path, [_manifest_row(mat, tmp_path)])
        rec = adapter.load_split("full_train")[0]
        assert "current_a" in rec.data.columns

    def test_temperature_c_present(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        adapter = _make_adapter(tmp_path, [_manifest_row(mat, tmp_path)])
        rec = adapter.load_split("full_train")[0]
        assert "temperature_c" in rec.data.columns

    def test_time_s_present(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        adapter = _make_adapter(tmp_path, [_manifest_row(mat, tmp_path)])
        rec = adapter.load_split("full_train")[0]
        assert "time_s" in rec.data.columns

    def test_cycler_ah_present(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        adapter = _make_adapter(tmp_path, [_manifest_row(mat, tmp_path)])
        rec = adapter.load_split("full_train")[0]
        assert "cycler_ah" in rec.data.columns

    def test_voltage_values_correct(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        adapter = _make_adapter(tmp_path, [_manifest_row(mat, tmp_path)])
        rec = adapter.load_split("full_train")[0]
        # First value should be ~4.185, last ~3.354
        assert abs(rec.data["voltage_v"].iloc[0] - 4.185) < 0.01
        assert abs(rec.data["voltage_v"].iloc[-1] - 3.354) < 0.01


# ---------------------------------------------------------------------------
# Test 17 — Core column lengths consistent
# ---------------------------------------------------------------------------

class TestColumnLengths:
    def test_all_core_columns_same_length(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        adapter = _make_adapter(tmp_path, [_manifest_row(mat, tmp_path)])
        rec = adapter.load_split("full_train")[0]
        n = rec.n_rows
        for col in ("time_s", "voltage_v", "current_a", "temperature_c", "cycler_ah"):
            assert len(rec.data[col]) == n, f"Column {col!r} length mismatch"


# ---------------------------------------------------------------------------
# Test 18 — Metadata filled correctly
# ---------------------------------------------------------------------------

class TestMetadataFilled:
    def _get_rec(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        adapter = _make_adapter(tmp_path, [_manifest_row(mat, tmp_path)])
        return adapter.load_split("full_train")[0]

    def test_dataset_field(self, tmp_path):
        assert self._get_rec(tmp_path).metadata["dataset"] == "panasonic_18650pf"

    def test_battery_field(self, tmp_path):
        assert self._get_rec(tmp_path).metadata["battery"] == "panasonic_18650pf"

    def test_source_file_non_empty(self, tmp_path):
        sf = self._get_rec(tmp_path).metadata["source_file"]
        assert isinstance(sf, str) and len(sf) > 0

    def test_split_label(self, tmp_path):
        assert self._get_rec(tmp_path).metadata["split"] == "full_train"

    def test_manifest_splits_dual_membership(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        adapter = _make_adapter(
            tmp_path,
            [_manifest_row(mat, tmp_path, full_train=1, partial_train=1)],
        )
        rec = adapter.load_split("full_train")[0]
        assert rec.metadata["manifest_splits"] == ["full_train", "partial_train"]

    def test_profile_field(self, tmp_path):
        assert self._get_rec(tmp_path).metadata["profile"] == "Cycle_1"

    def test_temperature_nominal(self, tmp_path):
        assert self._get_rec(tmp_path).metadata["temperature_nominal_c"] == 25.0

    def test_capacity_hint(self, tmp_path):
        assert self._get_rec(tmp_path).metadata["capacity_hint_ah"] == 2.9

    def test_soc_init_set(self, tmp_path):
        # pairing_status=confirmed → soc_init = 1.0
        assert self._get_rec(tmp_path).metadata["soc_init"] == 1.0

    def test_current_sign_convention(self, tmp_path):
        assert self._get_rec(tmp_path).metadata["current_sign_convention"] == "as_recorded"

    def test_sequence_id_non_empty(self, tmp_path):
        sid = self._get_rec(tmp_path).metadata["sequence_id"]
        assert isinstance(sid, str) and len(sid) > 0


# ---------------------------------------------------------------------------
# Test 19 — Source SHA-256 written to metadata (even without manifest SHA)
# ---------------------------------------------------------------------------

class TestSourceSHA:
    def test_source_sha256_in_metadata(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        adapter = _make_adapter(tmp_path, [_manifest_row(mat, tmp_path)])
        rec = adapter.load_split("full_train")[0]
        actual_sha = _sha256(mat)
        assert rec.metadata["source_sha256"] == actual_sha

    def test_no_manifest_sha_produces_qc_warn(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        adapter = _make_adapter(tmp_path, [_manifest_row(mat, tmp_path, sha256="")])
        rec = adapter.load_split("full_train")[0]
        # WARN because SHA was not in manifest
        assert rec.qc.status in ("WARN", "PASS")
        assert any("SHA" in issue for issue in rec.qc.issues)


# ---------------------------------------------------------------------------
# Test 20 — SHA mismatch raises SHA256MismatchError
# ---------------------------------------------------------------------------

class TestSHAMismatch:
    def test_sha_mismatch_raises(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        # Deliberately wrong SHA
        wrong_sha = "a" * 64
        row = _manifest_row(mat, tmp_path, sha256=wrong_sha)
        adapter = _make_adapter(tmp_path, [row])
        with pytest.raises(SHA256MismatchError):
            adapter.load_split("full_train")

    def test_sha_mismatch_error_message(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        wrong_sha = "b" * 64
        row = _manifest_row(mat, tmp_path, sha256=wrong_sha)
        adapter = _make_adapter(tmp_path, [row])
        with pytest.raises(SHA256MismatchError, match="mismatch"):
            adapter.load_split("full_train")

    def test_correct_sha_succeeds(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        correct_sha = _sha256(mat)
        row = _manifest_row(mat, tmp_path, sha256=correct_sha)
        adapter = _make_adapter(tmp_path, [row])
        records = adapter.load_split("full_train")
        assert len(records) == 1


# ---------------------------------------------------------------------------
# Test 21 — Missing required meas field → FAIL
# ---------------------------------------------------------------------------

class TestMissingField:
    def test_missing_voltage_gives_fail_qc(self, tmp_path):
        meas = _make_meas()
        del meas["Voltage"]
        p = tmp_path / "no_v.mat"
        scipy.io.savemat(str(p), {"meas": meas})
        row = _manifest_row(p, tmp_path)
        adapter = _make_adapter(tmp_path, [row])
        rec = adapter.load_split("full_train")[0]
        assert rec.qc.status == "FAIL"
        assert any("Voltage" in issue for issue in rec.qc.issues)

    def test_missing_time_gives_fail_qc(self, tmp_path):
        meas = _make_meas()
        del meas["Time"]
        p = tmp_path / "no_t.mat"
        scipy.io.savemat(str(p), {"meas": meas})
        row = _manifest_row(p, tmp_path)
        adapter = _make_adapter(tmp_path, [row])
        rec = adapter.load_split("full_train")[0]
        assert rec.qc.status == "FAIL"


# ---------------------------------------------------------------------------
# Test 22 — Mismatched array lengths → FAIL
# ---------------------------------------------------------------------------

class TestArrayLengthMismatch:
    def test_mismatched_lengths_fail(self, tmp_path):
        meas = _make_meas(n=50)
        # Make Voltage shorter than Time
        meas["Voltage"] = np.ones(20)
        p = tmp_path / "mismatch.mat"
        scipy.io.savemat(str(p), {"meas": meas})
        row = _manifest_row(p, tmp_path)
        adapter = _make_adapter(tmp_path, [row])
        rec = adapter.load_split("full_train")[0]
        assert rec.qc.status == "FAIL"
        assert any("length" in issue.lower() for issue in rec.qc.issues)


# ---------------------------------------------------------------------------
# Test 23 — Duplicate timestamp → WARN
# ---------------------------------------------------------------------------

class TestDuplicateTimestamp:
    def test_dup_timestamp_produces_warn(self, tmp_path):
        meas = _make_meas(dup_time=True)
        p = tmp_path / "dup_ts.mat"
        scipy.io.savemat(str(p), {"meas": meas})
        row = _manifest_row(p, tmp_path)
        adapter = _make_adapter(tmp_path, [row])
        rec = adapter.load_split("full_train")[0]
        assert rec.qc.duplicate_timestamp_count >= 1
        assert rec.qc.status in ("WARN", "FAIL")


# ---------------------------------------------------------------------------
# Test 24 — Negative time step → FAIL
# ---------------------------------------------------------------------------

class TestNegativeTimeStep:
    def test_neg_time_produces_fail(self, tmp_path):
        meas = _make_meas(neg_time=True)
        p = tmp_path / "neg_t.mat"
        scipy.io.savemat(str(p), {"meas": meas})
        row = _manifest_row(p, tmp_path)
        adapter = _make_adapter(tmp_path, [row])
        rec = adapter.load_split("full_train")[0]
        assert rec.qc.status == "FAIL"
        assert rec.qc.negative_time_step_count >= 1


# ---------------------------------------------------------------------------
# Test 25 — NaN recorded in QC
# ---------------------------------------------------------------------------

class TestNaNRecorded:
    def test_nan_in_voltage_sets_warn(self, tmp_path):
        meas = _make_meas(nan_voltage=True)
        p = tmp_path / "nan_v.mat"
        scipy.io.savemat(str(p), {"meas": meas})
        row = _manifest_row(p, tmp_path)
        adapter = _make_adapter(tmp_path, [row])
        rec = adapter.load_split("full_train")[0]
        assert rec.qc.status in ("WARN",)
        assert rec.qc.nan_count > 0

    def test_nan_count_nonzero(self, tmp_path):
        meas = _make_meas(nan_voltage=True)
        p = tmp_path / "nan_c.mat"
        scipy.io.savemat(str(p), {"meas": meas})
        row = _manifest_row(p, tmp_path)
        adapter = _make_adapter(tmp_path, [row])
        rec = adapter.load_split("full_train")[0]
        assert rec.qc.nan_count >= 1


# ---------------------------------------------------------------------------
# Test 26 — Inf → FAIL
# ---------------------------------------------------------------------------

class TestInfFail:
    def test_inf_current_produces_fail(self, tmp_path):
        meas = _make_meas(inf_current=True)
        p = tmp_path / "inf_c.mat"
        scipy.io.savemat(str(p), {"meas": meas})
        row = _manifest_row(p, tmp_path)
        adapter = _make_adapter(tmp_path, [row])
        rec = adapter.load_split("full_train")[0]
        assert rec.qc.status == "FAIL"
        assert rec.qc.inf_count >= 1


# ---------------------------------------------------------------------------
# Test 27 — Ah reset → WARN
# ---------------------------------------------------------------------------

class TestAhReset:
    def test_ah_reset_detected(self, tmp_path):
        meas = _make_meas(ah_reset=True)
        p = tmp_path / "ah_reset.mat"
        scipy.io.savemat(str(p), {"meas": meas})
        row = _manifest_row(p, tmp_path)
        adapter = _make_adapter(tmp_path, [row])
        rec = adapter.load_split("full_train")[0]
        assert rec.qc.ah_reset_detected is True
        assert rec.qc.status in ("WARN", "FAIL")

    def test_normal_ah_no_reset(self, tmp_path):
        mat = _write_mat(tmp_path, "ok_ah.mat")
        row = _manifest_row(mat, tmp_path)
        adapter = _make_adapter(tmp_path, [row])
        rec = adapter.load_split("full_train")[0]
        assert rec.qc.ah_reset_detected is False


# ---------------------------------------------------------------------------
# Test 28 — Original .mat not modified
# ---------------------------------------------------------------------------

class TestFileImmutability:
    def test_mat_sha256_unchanged_after_load(self, tmp_path):
        mat = _write_mat(tmp_path, "cycle.mat")
        sha_before = _sha256(mat)
        adapter = _make_adapter(tmp_path, [_manifest_row(mat, tmp_path)])
        adapter.load_split("full_train")
        sha_after = _sha256(mat)
        assert sha_before == sha_after


# ---------------------------------------------------------------------------
# Test 29 — Duplicate rows not deleted
# ---------------------------------------------------------------------------

class TestDuplicateRowsPreserved:
    def test_dup_timestamp_rows_not_dropped(self, tmp_path):
        meas = _make_meas(n=50, dup_time=True)
        p = tmp_path / "dup_rows.mat"
        scipy.io.savemat(str(p), {"meas": meas})
        row = _manifest_row(p, tmp_path)
        adapter = _make_adapter(tmp_path, [row])
        rec = adapter.load_split("full_train")[0]
        # Row count should still be 50 (no deletion)
        assert rec.n_rows == 50


# ---------------------------------------------------------------------------
# Test 30 — One SequenceRecord per file
# ---------------------------------------------------------------------------

class TestOneRecordPerFile:
    def test_two_files_two_records(self, tmp_path):
        mat1 = _write_mat(tmp_path, "f1.mat")
        mat2 = _write_mat(tmp_path, "f2.mat")
        rows = [
            _manifest_row(mat1, tmp_path, full_train=1, partial_train=1),
            _manifest_row(mat2, tmp_path, full_train=1, partial_train=0),
        ]
        rows[1]["filename"] = mat2.name
        rows[1]["profile"] = "Cycle_2"
        adapter = _make_adapter(tmp_path, rows)
        records = adapter.load_split("full_train")
        assert len(records) == 2

    def test_each_record_has_unique_source_file(self, tmp_path):
        mat1 = _write_mat(tmp_path, "fa.mat")
        mat2 = _write_mat(tmp_path, "fb.mat")
        rows = [
            _manifest_row(mat1, tmp_path),
            {**_manifest_row(mat2, tmp_path),
             "filename": mat2.name,
             "relative_path": str(mat2.relative_to(tmp_path.parent)),
             "profile": "Cycle_2", "paper_profile": "Mix2"},
        ]
        adapter = _make_adapter(tmp_path, rows)
        records = adapter.load_split("full_train")
        source_files = [r.metadata["source_file"] for r in records]
        assert len(source_files) == len(set(source_files))


# ---------------------------------------------------------------------------
# Test 31 — Return order matches manifest
# ---------------------------------------------------------------------------

class TestManifestOrderPreserved:
    def test_order_matches_manifest(self, tmp_path):
        mat1 = _write_mat(tmp_path, "first.mat")
        mat2 = _write_mat(tmp_path, "second.mat")
        rows = [
            _manifest_row(mat1, tmp_path),
            {**_manifest_row(mat2, tmp_path),
             "filename": mat2.name,
             "relative_path": str(mat2.relative_to(tmp_path.parent)),
             "profile": "Cycle_2", "paper_profile": "Mix2"},
        ]
        adapter = _make_adapter(tmp_path, rows)
        records = adapter.load_split("full_train")
        assert records[0].metadata["source_file"].endswith("first.mat")
        assert records[1].metadata["source_file"].endswith("second.mat")


# ---------------------------------------------------------------------------
# Test 32 — Split leakage: full_train doesn't include test files
# ---------------------------------------------------------------------------

class TestSplitLeakage:
    def test_full_train_excludes_test_files(self, tmp_path):
        mat_tr = _write_mat(tmp_path, "train.mat")
        mat_ts = _write_mat(tmp_path, "test_file.mat")
        rows = [
            _manifest_row(mat_tr, tmp_path, full_train=1, test=0),
            {**_manifest_row(mat_ts, tmp_path),
             "filename": mat_ts.name,
             "relative_path": str(mat_ts.relative_to(tmp_path.parent)),
             "full_train": "0", "partial_train": "0", "test": "1",
             "profile": "UDDS", "paper_profile": "UDDS"},
        ]
        adapter = _make_adapter(tmp_path, rows)
        train_records = adapter.load_split("full_train")
        train_sources = [r.metadata["source_file"] for r in train_records]
        assert all("train.mat" in s for s in train_sources)
        assert not any("test_file.mat" in s for s in train_sources)

    def test_test_excludes_train_files(self, tmp_path):
        mat_tr = _write_mat(tmp_path, "train2.mat")
        mat_ts = _write_mat(tmp_path, "test2.mat")
        rows = [
            _manifest_row(mat_tr, tmp_path, full_train=1, test=0),
            {**_manifest_row(mat_ts, tmp_path),
             "filename": mat_ts.name,
             "relative_path": str(mat_ts.relative_to(tmp_path.parent)),
             "full_train": "0", "partial_train": "0", "test": "1",
             "profile": "UDDS", "paper_profile": "UDDS"},
        ]
        adapter = _make_adapter(tmp_path, rows)
        test_records = adapter.load_split("test")
        test_sources = [r.metadata["source_file"] for r in test_records]
        assert all("test2.mat" in s for s in test_sources)
        assert not any("train2.mat" in s for s in test_sources)


# ---------------------------------------------------------------------------
# Test 33 — exclude not in full_train / test
# ---------------------------------------------------------------------------

class TestExcludeNotSelected:
    def test_excluded_file_not_in_full_train(self, tmp_path):
        mat_sel = _write_mat(tmp_path, "sel.mat")
        mat_exc = _write_mat(tmp_path, "excl.mat")
        rows = [
            _manifest_row(mat_sel, tmp_path, full_train=1),
            _manifest_row(
                mat_exc, tmp_path,
                full_train=0, partial_train=0, test=0, exclude=1,
                profile="HWFET", paper_profile="",
            ),
        ]
        adapter = _make_adapter(tmp_path, rows)
        records = adapter.load_split("full_train")
        sources = [r.metadata["source_file"] for r in records]
        assert not any("excl.mat" in s for s in sources)


# ---------------------------------------------------------------------------
# Test 34 — Determinism: two runs give identical results
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_two_runs_same_records(self, tmp_path):
        mat = _write_mat(tmp_path, "det.mat")
        rows = [_manifest_row(mat, tmp_path)]
        adapter = _make_adapter(tmp_path, rows)
        r1 = adapter.load_split("full_train")
        r2 = adapter.load_split("full_train")
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(
                a.data["voltage_v"].values, b.data["voltage_v"].values
            )

    def test_two_runs_same_sha256(self, tmp_path):
        mat = _write_mat(tmp_path, "det2.mat")
        rows = [_manifest_row(mat, tmp_path)]
        adapter = _make_adapter(tmp_path, rows)
        r1 = adapter.load_split("full_train")[0]
        r2 = adapter.load_split("full_train")[0]
        assert r1.metadata["source_sha256"] == r2.metadata["source_sha256"]


# ---------------------------------------------------------------------------
# Test 35 — sequence_id stability
# ---------------------------------------------------------------------------

class TestSequenceIDStability:
    def test_sequence_id_same_on_two_runs(self, tmp_path):
        mat = _write_mat(tmp_path, "sid.mat")
        rows = [_manifest_row(mat, tmp_path)]
        adapter = _make_adapter(tmp_path, rows)
        r1 = adapter.load_split("full_train")[0]
        r2 = adapter.load_split("full_train")[0]
        assert r1.metadata["sequence_id"] == r2.metadata["sequence_id"]

    def test_sequence_id_contains_filename_stem(self, tmp_path):
        mat = _write_mat(tmp_path, "my_file.mat")
        rows = [_manifest_row(mat, tmp_path)]
        adapter = _make_adapter(tmp_path, rows)
        rec = adapter.load_split("full_train")[0]
        assert "my_file" in rec.metadata["sequence_id"]

    def test_sequence_ids_are_unique_per_file(self, tmp_path):
        mat1 = _write_mat(tmp_path, "file1.mat")
        mat2 = _write_mat(tmp_path, "file2.mat")
        rows = [
            _manifest_row(mat1, tmp_path),
            {**_manifest_row(mat2, tmp_path),
             "filename": mat2.name,
             "relative_path": str(mat2.relative_to(tmp_path.parent)),
             "profile": "Cycle_2", "paper_profile": "Mix2"},
        ]
        adapter = _make_adapter(tmp_path, rows)
        records = adapter.load_split("full_train")
        ids = [r.metadata["sequence_id"] for r in records]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Manifest qc_status FAIL entries are skipped
# ---------------------------------------------------------------------------

class TestManifestQCFail:
    def test_manifest_fail_entry_skipped(self, tmp_path):
        mat = _write_mat(tmp_path, "bad.mat")
        row = _manifest_row(mat, tmp_path, qc_status="FAIL",
                            qc_notes="zero current rest file")
        adapter = _make_adapter(tmp_path, [row])
        records = adapter.load_split("full_train")
        assert records == []

    def test_manifest_warn_entry_loaded_with_note(self, tmp_path):
        mat = _write_mat(tmp_path, "warn.mat")
        row = _manifest_row(mat, tmp_path, qc_status="WARN",
                            qc_notes="temp slightly off")
        adapter = _make_adapter(tmp_path, [row])
        records = adapter.load_split("full_train")
        assert len(records) == 1
        assert any("temp slightly off" in n for n in records[0].qc.notes)


# ---------------------------------------------------------------------------
# summarize()
# ---------------------------------------------------------------------------

class TestAdapterSummarize:
    def test_summarize_returns_correct_counts(self, tmp_path):
        mat = _write_mat(tmp_path, "sum.mat")
        row = _manifest_row(mat, tmp_path)
        adapter = _make_adapter(tmp_path, [row])
        records = adapter.load_split("full_train")
        summary = adapter.summarize(records, split="full_train")
        assert summary.loaded_records == 1
        assert summary.expected_manifest_entries == 1
        assert summary.skipped_manifest_failures == 0
        assert summary.total_rows == 50
        assert summary.split == "full_train"

    def test_summary_accounts_for_skipped_manifest_fail(self, tmp_path):
        mat_ok = _write_mat(tmp_path, "ok.mat")
        mat_bad = _write_mat(tmp_path, "bad.mat")
        rows = [
            _manifest_row(mat_ok, tmp_path, full_train=1, partial_train=0),
            {**_manifest_row(mat_bad, tmp_path, full_train=1, partial_train=0),
             "filename": mat_bad.name,
             "relative_path": str(mat_bad.relative_to(tmp_path.parent)),
             "qc_status": "FAIL",
             "qc_notes": "rest file"},
        ]
        adapter = _make_adapter(tmp_path, rows)
        records = adapter.load_split("full_train")
        summary = adapter.summarize(records, split="full_train")
        assert summary.expected_manifest_entries == 2
        assert summary.loaded_records == 1
        assert summary.skipped_manifest_failures == 1
        assert summary.fail_count == 0  # skipped entries are not QC FAIL records


class TestMultiSplitSummarizeStats:
    """Regression: summarize must use stats for the requested split, not the last load."""

    def test_summarize_full_train_after_loading_test(self, tmp_path):
        mat_tr = _write_mat(tmp_path, "train.mat")
        mat_ts = _write_mat(tmp_path, "test_file.mat")
        rows = [
            _manifest_row(mat_tr, tmp_path, full_train=1, partial_train=0, test=0),
            {**_manifest_row(mat_ts, tmp_path),
             "filename": mat_ts.name,
             "relative_path": str(mat_ts.relative_to(tmp_path.parent)),
             "full_train": "0", "partial_train": "0", "test": "1",
             "profile": "UDDS", "paper_profile": "UDDS"},
        ]
        adapter = _make_adapter(tmp_path, rows)
        train_records = adapter.load_split("full_train")
        adapter.load_split("test")
        summary = adapter.summarize(train_records, split="full_train")
        assert summary.expected_manifest_entries == 1
        assert summary.loaded_records == 1
        assert summary.skipped_manifest_failures == 0
        assert summary.split == "full_train"


class TestRequestedSplitSemantics:
    def test_full_train_request_keeps_split_label(self, tmp_path):
        mat = _write_mat(tmp_path, "mix1.mat")
        row = _manifest_row(mat, tmp_path, full_train=1, partial_train=1)
        adapter = _make_adapter(tmp_path, [row])
        rec = adapter.load_split("full_train")[0]
        assert rec.metadata["split"] == "full_train"
        assert rec.metadata["manifest_splits"] == ["full_train", "partial_train"]

    def test_partial_train_request_uses_requested_split(self, tmp_path):
        mat = _write_mat(tmp_path, "mix1b.mat")
        row = _manifest_row(mat, tmp_path, full_train=1, partial_train=1)
        adapter = _make_adapter(tmp_path, [row])
        rec = adapter.load_split("partial_train")[0]
        assert rec.metadata["split"] == "partial_train"
        assert rec.metadata["manifest_splits"] == ["full_train", "partial_train"]

    def test_test_request_uses_requested_split(self, tmp_path):
        mat = _write_mat(tmp_path, "us06.mat")
        row = _manifest_row(
            mat, tmp_path,
            full_train=0, partial_train=0, test=1,
            profile="US06", paper_profile="US06",
        )
        adapter = _make_adapter(tmp_path, [row])
        rec = adapter.load_split("test")[0]
        assert rec.metadata["split"] == "test"
        assert rec.metadata["manifest_splits"] == ["test"]

    def test_exclude_request_uses_requested_split(self, tmp_path):
        mat = _write_mat(tmp_path, "hwfet.mat")
        row = _manifest_row(
            mat, tmp_path,
            full_train=0, partial_train=0, test=0, exclude=1,
            profile="HWFET", paper_profile="",
            file_type="drive_cycle",
        )
        adapter = _make_adapter(tmp_path, [row])
        rec = adapter.load_split("exclude")[0]
        assert rec.metadata["split"] == "exclude"
        assert rec.metadata["manifest_splits"] == ["exclude"]

    def test_paper_profile_not_inferred_from_profile(self, tmp_path):
        mat = _write_mat(tmp_path, "no_pp.mat")
        row = _manifest_row(mat, tmp_path, paper_profile="")
        adapter = _make_adapter(tmp_path, [row])
        with pytest.raises(AdapterSchemaError, match="paper_profile"):
            adapter.load_split("full_train")

    def test_charge_selected_uses_cc_paper_profile(self, tmp_path):
        mat = _write_mat(tmp_path, "charge.mat")
        row = _manifest_row(
            mat, tmp_path,
            profile="Charge", paper_profile="CC", file_type="charge",
        )
        adapter = _make_adapter(tmp_path, [row])
        rec = adapter.load_split("full_train")[0]
        assert rec.metadata["paper_profile"] == "CC"
        assert rec.metadata["profile"] == "Charge"
        assert rec.metadata["record_role"] == "soc_anchor"


class TestRecordRoleMetadata:
    def test_drive_cycle_record_role_model_input(self, tmp_path):
        mat = _write_mat(tmp_path, "mix1.mat")
        rec = _make_adapter(tmp_path, [_manifest_row(mat, tmp_path)]).load_split("full_train")[0]
        assert rec.metadata["record_role"] == "model_input"

    def test_charge_record_role_soc_anchor(self, tmp_path):
        mat = _write_mat(tmp_path, "charge2.mat")
        row = _manifest_row(
            mat, tmp_path,
            profile="Charge", paper_profile="CC", file_type="charge",
        )
        rec = _make_adapter(tmp_path, [row]).load_split("full_train")[0]
        assert rec.metadata["record_role"] == "soc_anchor"

    def test_excluded_record_role(self, tmp_path):
        mat = _write_mat(tmp_path, "excl2.mat")
        row = _manifest_row(
            mat, tmp_path,
            full_train=0, partial_train=0, test=0, exclude=1,
            profile="HWFET", paper_profile="",
        )
        rec = _make_adapter(tmp_path, [row]).load_split("exclude")[0]
        assert rec.metadata["record_role"] == "excluded"


# ===========================================================================
# Integration smoke test (requires real Panasonic data)
# ===========================================================================

_MANIFEST_PATH = _REPO_ROOT / "data/manifests/panasonic_vidal2019_25c_manifest.csv"
_RAW_EXISTS = (_REPO_ROOT / "data/raw/Panasonic_18650PF/25degC").exists()


@pytest.mark.integration
@pytest.mark.skipif(
    not _RAW_EXISTS or not _MANIFEST_PATH.exists(),
    reason="Real Panasonic data or manifest not available",
)
class TestIntegrationPanasonic:
    """Smoke tests against real Panasonic 25 °C files."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return PanasonicAdapter(
            raw_root=_REPO_ROOT,
            manifest_path=_MANIFEST_PATH,
        )

    def test_full_train_loads_nonzero_records(self, adapter):
        records = adapter.load_split("full_train")
        assert len(records) > 0

    def test_all_records_are_sequence_records(self, adapter):
        for rec in adapter.load_split("full_train"):
            assert isinstance(rec, SequenceRecord)

    def test_all_records_have_required_columns(self, adapter):
        for rec in adapter.load_split("full_train"):
            assert rec.has_required_columns, (
                f"{rec.metadata['source_file']} missing required columns"
            )

    def test_all_records_nonzero_rows(self, adapter):
        for rec in adapter.load_split("full_train"):
            assert rec.n_rows > 0

    def test_split_labels_correct_full_train(self, adapter):
        for rec in adapter.load_split("full_train"):
            assert rec.metadata["split"] == "full_train"

    def test_manifest_splits_for_partial_subset(self, adapter):
        records = adapter.load_split("full_train")
        partial = [r for r in records if "partial_train" in r.metadata["manifest_splits"]]
        assert len(partial) >= 3
        for rec in partial:
            assert rec.metadata["split"] == "full_train"
            assert "full_train" in rec.metadata["manifest_splits"]
            assert "partial_train" in rec.metadata["manifest_splits"]

    def test_split_labels_correct_test(self, adapter):
        for rec in adapter.load_split("test"):
            assert rec.metadata["split"] == "test"

    def test_no_train_test_overlap(self, adapter):
        train_src = {r.metadata["source_file"] for r in adapter.load_split("full_train")}
        test_src = {r.metadata["source_file"] for r in adapter.load_split("test")}
        overlap = train_src & test_src
        assert overlap == set(), f"Train/test overlap: {overlap}"

    def test_sequence_ids_unique(self, adapter):
        records = adapter.load_split("full_train") + adapter.load_split("test")
        ids = [r.metadata["sequence_id"] for r in records]
        assert len(ids) == len(set(ids))

    def test_source_sha256_matches_actual(self, adapter):
        records = adapter.load_split("full_train")
        for rec in records:
            src = _REPO_ROOT / rec.metadata["source_file"]
            actual = _sha256(src)
            assert rec.metadata["source_sha256"] == actual, (
                f"SHA mismatch for {src.name}"
            )

    def test_profile_matches_manifest(self, adapter):
        from battery_benchmark.data.manifest_loader import load_manifest, select_manifest_entries
        entries = load_manifest(_MANIFEST_PATH)
        selected = select_manifest_entries(entries, "full_train")
        records = adapter.load_split("full_train")
        # Manifest entries that are not skipped (qc_status != FAIL)
        expected_profiles = [e.profile for e in selected if e.qc_status != "FAIL"]
        actual_profiles = [r.metadata["profile"] for r in records]
        assert expected_profiles == actual_profiles

    def test_summarize_pass_count(self, adapter):
        records = adapter.load_split("full_train")
        summary = adapter.summarize(records, split="full_train")
        assert summary.loaded_records == len(records)
        assert summary.expected_manifest_entries == len(records)
        assert summary.skipped_manifest_failures == 0
        assert summary.pass_count + summary.warn_count + summary.fail_count == len(records)

    def test_paper_profile_from_manifest_not_profile(self, adapter):
        records = adapter.load_split("full_train")
        cycle1 = next(r for r in records if r.metadata["profile"] == "Cycle_1")
        assert cycle1.metadata["paper_profile"] == "Mix1"
        assert cycle1.metadata["paper_profile"] != cycle1.metadata["profile"]

    def test_record_roles_full_train(self, adapter):
        records = adapter.load_split("full_train")
        for rec in records:
            if rec.metadata["file_type"] == "drive_cycle":
                assert rec.metadata["record_role"] == "model_input"
            elif rec.metadata["file_type"] == "charge":
                assert rec.metadata["record_role"] == "soc_anchor"

    def test_excluded_records_have_excluded_role(self, adapter):
        for rec in adapter.load_split("exclude"):
            assert rec.metadata["record_role"] == "excluded"

    def test_model_input_drive_cycles_have_soc_metadata(self, adapter):
        records = adapter.load_split("full_train") + adapter.load_split("test")
        for rec in records:
            if rec.metadata["record_role"] == "model_input":
                assert rec.metadata["soc_init"] is not None
                assert rec.metadata["soc_anchor_source"]
                assert rec.metadata["soc_anchor_source"] != "not_applicable"
