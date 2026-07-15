"""Tests for LGHG2Adapter — unit tests + integration smoke test."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io

from battery_benchmark.data.adapter_qc import AdapterSchemaError
from battery_benchmark.data.lg_hg2 import LGHG2Adapter
from battery_benchmark.data.manifest_loader import SHA256MismatchError, load_manifest, select_manifest_entries
from battery_benchmark.data.sequence_record import SequenceRecord

_REPO_ROOT = Path(__file__).parent.parent
_MANIFEST = _REPO_ROOT / "data/manifests/lg_hg2_vidal2019_25c_manifest.csv"
_RAW_EXISTS = (_REPO_ROOT / "data/raw/LG_HG2/25degC").exists()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_meas(n: int = 50, **overrides) -> dict:
    meas = {
        "Time": np.arange(n, dtype=float),
        "TimeStamp": np.array([f"2018-10-29 14:{i:02d}:00" for i in range(n)], dtype=object),
        "Voltage": np.linspace(4.2, 3.0, n),
        "Current": np.full(n, -2.5),
        "Battery_Temp_degC": np.full(n, 25.0),
        "Ah": np.linspace(0.0, -2.5, n),
    }
    meas.update(overrides)
    return meas


def _write_mat(tmp_path: Path, name: str, **overrides) -> Path:
    p = tmp_path / name
    scipy.io.savemat(str(p), {"meas": _make_meas(**overrides)})
    return p


def _manifest_row(mat_path: Path, tmp_path: Path, **kwargs) -> dict:
    row = {
        "battery": "lg_hg2",
        "temperature_c": "25",
        "filename": mat_path.name,
        "relative_path": str(mat_path.relative_to(tmp_path.parent)),
        "profile": "Mixed1",
        "paper_profile": "Mix1",
        "file_type": "drive_cycle",
        "full_train": "1",
        "partial_train": "1",
        "test": "0",
        "exclude": "0",
        "qc_status": "PASS",
        "qc_notes": "",
        "sha256": "",
        "pairing_status": "confirmed",
        "soc_init": "1.0",
        "soc_anchor_source": "full_charge_protocol",
        "confidence": "high",
        "evidence": "test",
        "notes": "",
    }
    row.update(kwargs)
    if "record_role" not in kwargs:
        exclude = int(row["exclude"])
        ft = row["file_type"]
        selected = int(row["full_train"]) or int(row["partial_train"]) or int(row["test"])
        if exclude:
            row["record_role"] = "excluded"
        elif ft == "charge" and selected:
            row["record_role"] = "model_input"
        else:
            row["record_role"] = "model_input"
    return row


def _adapter(tmp_path: Path, rows: list[dict]) -> LGHG2Adapter:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return LGHG2Adapter(raw_root=tmp_path.parent, manifest_path=manifest)


class TestLGBasicLoad:
    def test_creates_sequence_record(self, tmp_path):
        mat = _write_mat(tmp_path, "mixed1.mat")
        rec = _adapter(tmp_path, [_manifest_row(mat, tmp_path)]).load_split("full_train")[0]
        assert isinstance(rec, SequenceRecord)
        assert rec.n_rows == 50

    def test_column_mapping(self, tmp_path):
        mat = _write_mat(tmp_path, "mixed1.mat")
        rec = _adapter(tmp_path, [_manifest_row(mat, tmp_path)]).load_split("full_train")[0]
        for col in ("time_s", "voltage_v", "current_a", "temperature_c", "cycler_ah"):
            assert col in rec.data.columns

    def test_metadata_fields(self, tmp_path):
        mat = _write_mat(tmp_path, "mixed1.mat")
        meta = _adapter(tmp_path, [_manifest_row(mat, tmp_path)]).load_split("full_train")[0].metadata
        assert meta["dataset"] == "lg_hg2"
        assert meta["paper_profile"] == "Mix1"
        assert meta["capacity_hint_ah"] == 3.0
        assert meta["current_sign_convention"] == "as_recorded"


class TestLGRequestedSplit:
    def test_full_train_keeps_requested_split(self, tmp_path):
        mat = _write_mat(tmp_path, "m1.mat")
        rec = _adapter(tmp_path, [_manifest_row(mat, tmp_path, full_train=1, partial_train=1)]).load_split("full_train")[0]
        assert rec.metadata["split"] == "full_train"
        assert rec.metadata["manifest_splits"] == ["full_train", "partial_train"]

    def test_partial_train_request(self, tmp_path):
        mat = _write_mat(tmp_path, "m1b.mat")
        rec = _adapter(tmp_path, [_manifest_row(mat, tmp_path, full_train=1, partial_train=1)]).load_split("partial_train")[0]
        assert rec.metadata["split"] == "partial_train"

    def test_test_split_request(self, tmp_path):
        mat = _write_mat(tmp_path, "us06.mat")
        row = _manifest_row(mat, tmp_path, full_train=0, partial_train=0, test=1,
                            profile="US06", paper_profile="US06")
        rec = _adapter(tmp_path, [row]).load_split("test")[0]
        assert rec.metadata["split"] == "test"
        assert rec.metadata["manifest_splits"] == ["test"]


class TestLGPaperProfile:
    def test_no_profile_fallback(self, tmp_path):
        mat = _write_mat(tmp_path, "bad_pp.mat")
        row = _manifest_row(mat, tmp_path, paper_profile="")
        with pytest.raises(AdapterSchemaError, match="paper_profile"):
            _adapter(tmp_path, [row]).load_split("full_train")

    def test_charge_uses_cc(self, tmp_path):
        mat = _write_mat(tmp_path, "charge.mat")
        row = _manifest_row(mat, tmp_path, profile="Charge", paper_profile="CC",
                            file_type="charge", soc_init="", soc_anchor_source="")
        rec = _adapter(tmp_path, [row]).load_split("full_train")[0]
        assert rec.metadata["paper_profile"] == "CC"
        assert rec.metadata["record_role"] == "model_input"


class TestLGRecordRoleMetadata:
    def test_drive_cycle_record_role_model_input(self, tmp_path):
        mat = _write_mat(tmp_path, "mix_role.mat")
        rec = _adapter(tmp_path, [_manifest_row(mat, tmp_path)]).load_split("full_train")[0]
        assert rec.metadata["record_role"] == "model_input"

    def test_charge_record_role_model_input(self, tmp_path):
        mat = _write_mat(tmp_path, "chg_role.mat")
        row = _manifest_row(mat, tmp_path, profile="Charge", paper_profile="CC",
                            file_type="charge", soc_init="", soc_anchor_source="")
        rec = _adapter(tmp_path, [row]).load_split("full_train")[0]
        assert rec.metadata["record_role"] == "model_input"


class TestLGSHAAndQC:
    def test_sha_match_with_manifest(self, tmp_path):
        mat = _write_mat(tmp_path, "sha.mat")
        digest = _sha256(mat)
        row = _manifest_row(mat, tmp_path, sha256=digest)
        rec = _adapter(tmp_path, [row]).load_split("full_train")[0]
        assert rec.metadata["source_sha256"] == digest

    def test_sha_mismatch_raises(self, tmp_path):
        mat = _write_mat(tmp_path, "sha_bad.mat")
        row = _manifest_row(mat, tmp_path, sha256="a" * 64)
        with pytest.raises(SHA256MismatchError):
            _adapter(tmp_path, [row]).load_split("full_train")

    def test_empty_manifest_sha_warns(self, tmp_path):
        mat = _write_mat(tmp_path, "no_sha.mat")
        rec = _adapter(tmp_path, [_manifest_row(mat, tmp_path)]).load_split("full_train")[0]
        assert rec.qc.status in ("WARN", "PASS")
        assert any("SHA" in issue for issue in rec.qc.issues)

    def test_missing_voltage_fail(self, tmp_path):
        p = tmp_path / "no_v.mat"
        meas = _make_meas()
        del meas["Voltage"]
        scipy.io.savemat(str(p), {"meas": meas})
        rec = _adapter(tmp_path, [_manifest_row(p, tmp_path)]).load_split("full_train")[0]
        assert rec.qc.status == "FAIL"

    def test_length_mismatch_fail(self, tmp_path):
        p = tmp_path / "len.mat"
        meas = _make_meas()
        meas["Voltage"] = np.ones(10)
        scipy.io.savemat(str(p), {"meas": meas})
        rec = _adapter(tmp_path, [_manifest_row(p, tmp_path)]).load_split("full_train")[0]
        assert rec.qc.status == "FAIL"

    def test_nan_voltage_warn(self, tmp_path):
        p = tmp_path / "nan.mat"
        meas = _make_meas()
        meas["Voltage"][3] = np.nan
        scipy.io.savemat(str(p), {"meas": meas})
        rec = _adapter(tmp_path, [_manifest_row(p, tmp_path)]).load_split("full_train")[0]
        assert rec.qc.status == "WARN"

    def test_inf_current_fail(self, tmp_path):
        p = tmp_path / "inf.mat"
        meas = _make_meas()
        meas["Current"][2] = np.inf
        scipy.io.savemat(str(p), {"meas": meas})
        rec = _adapter(tmp_path, [_manifest_row(p, tmp_path)]).load_split("full_train")[0]
        assert rec.qc.status == "FAIL"

    def test_dup_timestamp_warn(self, tmp_path):
        p = tmp_path / "dup.mat"
        time = np.arange(50, dtype=float)
        time[-1] = time[-2]
        meas = _make_meas(Time=time)
        scipy.io.savemat(str(p), {"meas": meas})
        rec = _adapter(tmp_path, [_manifest_row(p, tmp_path)]).load_split("full_train")[0]
        assert rec.qc.duplicate_timestamp_count >= 1

    def test_negative_dt_fail(self, tmp_path):
        p = tmp_path / "neg.mat"
        time = np.arange(50, dtype=float)
        time[10] = time[9] - 1.0
        meas = _make_meas(Time=time)
        scipy.io.savemat(str(p), {"meas": meas})
        rec = _adapter(tmp_path, [_manifest_row(p, tmp_path)]).load_split("full_train")[0]
        assert rec.qc.status == "FAIL"

    def test_ah_reset_warn(self, tmp_path):
        p = tmp_path / "ah.mat"
        ah = np.linspace(0.0, -2.5, 50)
        ah[-1] = -0.001
        meas = _make_meas(Ah=ah)
        scipy.io.savemat(str(p), {"meas": meas})
        rec = _adapter(tmp_path, [_manifest_row(p, tmp_path)]).load_split("full_train")[0]
        assert rec.qc.ah_reset_detected

    def test_manifest_fail_skipped(self, tmp_path):
        mat = _write_mat(tmp_path, "skip.mat")
        row = _manifest_row(mat, tmp_path, qc_status="FAIL", qc_notes="bad")
        adapter = _adapter(tmp_path, [row])
        assert adapter.load_split("full_train") == []
        summary = adapter.summarize([], split="full_train")
        assert summary.expected_manifest_entries == 1
        assert summary.skipped_manifest_failures == 1


class TestLGDeterminismAndSplits:
    def test_one_record_per_file(self, tmp_path):
        mat1 = _write_mat(tmp_path, "a.mat")
        mat2 = _write_mat(tmp_path, "b.mat")
        rows = [
            _manifest_row(mat1, tmp_path),
            {**_manifest_row(mat2, tmp_path), "filename": mat2.name,
             "relative_path": str(mat2.relative_to(tmp_path.parent)),
             "profile": "Mixed2", "paper_profile": "Mix2"},
        ]
        records = _adapter(tmp_path, rows).load_split("full_train")
        assert len(records) == 2

    def test_order_preserved(self, tmp_path):
        mat1 = _write_mat(tmp_path, "first.mat")
        mat2 = _write_mat(tmp_path, "second.mat")
        rows = [
            _manifest_row(mat1, tmp_path),
            {**_manifest_row(mat2, tmp_path), "filename": mat2.name,
             "relative_path": str(mat2.relative_to(tmp_path.parent)),
             "profile": "Mixed2", "paper_profile": "Mix2"},
        ]
        records = _adapter(tmp_path, rows).load_split("full_train")
        assert records[0].metadata["source_file"].endswith("first.mat")

    def test_no_train_test_overlap(self, tmp_path):
        mat_tr = _write_mat(tmp_path, "tr.mat")
        mat_ts = _write_mat(tmp_path, "ts.mat")
        rows = [
            _manifest_row(mat_tr, tmp_path, full_train=1, test=0),
            {**_manifest_row(mat_ts, tmp_path), "filename": mat_ts.name,
             "relative_path": str(mat_ts.relative_to(tmp_path.parent)),
             "full_train": "0", "partial_train": "0", "test": "1",
             "profile": "US06", "paper_profile": "US06"},
        ]
        adapter = _adapter(tmp_path, rows)
        train = {r.metadata["source_file"] for r in adapter.load_split("full_train")}
        test = {r.metadata["source_file"] for r in adapter.load_split("test")}
        assert train.isdisjoint(test)

    def test_sequence_id_stable(self, tmp_path):
        mat = _write_mat(tmp_path, "sid.mat")
        adapter = _adapter(tmp_path, [_manifest_row(mat, tmp_path)])
        r1 = adapter.load_split("full_train")[0]
        r2 = adapter.load_split("full_train")[0]
        assert r1.metadata["sequence_id"] == r2.metadata["sequence_id"]
        assert "sid" in r1.metadata["sequence_id"]

    def test_raw_file_immutable(self, tmp_path):
        mat = _write_mat(tmp_path, "imm.mat")
        before = _sha256(mat)
        _adapter(tmp_path, [_manifest_row(mat, tmp_path)]).load_split("full_train")
        assert _sha256(mat) == before

    def test_summarize_multi_split_no_cross_talk(self, tmp_path):
        mat_tr = _write_mat(tmp_path, "tr2.mat")
        mat_ts = _write_mat(tmp_path, "ts2.mat")
        rows = [
            _manifest_row(mat_tr, tmp_path, full_train=1, partial_train=0, test=0),
            {**_manifest_row(mat_ts, tmp_path), "filename": mat_ts.name,
             "relative_path": str(mat_ts.relative_to(tmp_path.parent)),
             "full_train": "0", "partial_train": "0", "test": "1",
             "profile": "US06", "paper_profile": "US06"},
        ]
        adapter = _adapter(tmp_path, rows)
        train_records = adapter.load_split("full_train")
        adapter.load_split("test")
        summary = adapter.summarize(train_records, split="full_train")
        assert summary.expected_manifest_entries == 1
        assert summary.split == "full_train"


@pytest.mark.integration
@pytest.mark.skipif(not _RAW_EXISTS or not _MANIFEST.exists(), reason="LG raw data unavailable")
class TestLGIntegration:
    @pytest.fixture(scope="class")
    def adapter(self):
        return LGHG2Adapter(raw_root=_REPO_ROOT, manifest_path=_MANIFEST)

    def test_full_train_count(self, adapter):
        entries = select_manifest_entries(load_manifest(_MANIFEST), "full_train")
        records = adapter.load_split("full_train")
        assert len(records) == len([e for e in entries if e.qc_status != "FAIL"])

    def test_partial_train_count(self, adapter):
        records = adapter.load_split("partial_train")
        assert len(records) == 7

    def test_test_count(self, adapter):
        records = adapter.load_split("test")
        assert len(records) == 7

    def test_required_columns_and_rows(self, adapter):
        for rec in adapter.load_split("full_train"):
            assert rec.has_required_columns
            assert rec.n_rows > 0

    def test_paper_profile_from_manifest(self, adapter):
        rec = next(r for r in adapter.load_split("full_train") if r.metadata["profile"] == "Mixed1")
        assert rec.metadata["paper_profile"] == "Mix1"

    def test_requested_split_metadata(self, adapter):
        for rec in adapter.load_split("full_train"):
            assert rec.metadata["split"] == "full_train"

    def test_no_train_test_overlap(self, adapter):
        train = {r.metadata["source_file"] for r in adapter.load_split("full_train")}
        test = {r.metadata["source_file"] for r in adapter.load_split("test")}
        assert train.isdisjoint(test)

    def test_sequence_ids_unique(self, adapter):
        records = adapter.load_split("full_train") + adapter.load_split("test")
        ids = [r.metadata["sequence_id"] for r in records]
        assert len(ids) == len(set(ids))

    def test_summary_accounting(self, adapter):
        records = adapter.load_split("full_train")
        summary = adapter.summarize(records, split="full_train")
        assert summary.expected_manifest_entries == 17
        assert summary.loaded_records == 17
        assert summary.skipped_manifest_failures == 0
        assert summary.expected_manifest_entries == summary.loaded_records + summary.skipped_manifest_failures

    def test_integration_qc_report(self, adapter, capsys):
        for split in ("full_train", "partial_train", "test"):
            records = adapter.load_split(split)
            summary = adapter.summarize(records, split=split)
            print(f"{split}: loaded={summary.loaded_records} pass={summary.pass_count} "
                  f"warn={summary.warn_count} fail={summary.fail_count} rows={summary.total_rows}")

    def test_record_roles_full_train(self, adapter):
        for rec in adapter.load_split("full_train"):
            if rec.metadata["file_type"] == "drive_cycle":
                assert rec.metadata["record_role"] == "model_input"
            elif rec.metadata["file_type"] == "charge":
                assert rec.metadata["record_role"] in {"model_input", "soc_anchor"}

    def test_excluded_records_have_excluded_role(self, adapter):
        for rec in adapter.load_split("exclude"):
            assert rec.metadata["record_role"] == "excluded"

    def test_model_input_drive_cycles_have_soc_metadata(self, adapter):
        records = adapter.load_split("full_train") + adapter.load_split("test")
        for rec in records:
            if (
                rec.metadata["record_role"] == "model_input"
                and rec.metadata["file_type"] == "drive_cycle"
            ):
                assert rec.metadata["soc_init"] is not None
                assert rec.metadata["soc_anchor_source"]
                assert rec.metadata["soc_anchor_source"] != "not_applicable"
