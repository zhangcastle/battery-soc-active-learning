"""Tests for Samsung30TAdapter — unit tests + integration smoke test."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io

from battery_benchmark.data.adapter_common import build_adapter_summary
from battery_benchmark.data.adapter_qc import AdapterSchemaError
from battery_benchmark.data.manifest_loader import (
    SHA256MismatchError,
    load_manifest,
    select_manifest_entries,
)
from battery_benchmark.data.samsung_30t import Samsung30TAdapter
from battery_benchmark.data.schema import REQUIRED_METADATA_FIELDS
from battery_benchmark.data.sequence_record import SequenceRecord

_REPO_ROOT = Path(__file__).parent.parent
_MANIFEST = _REPO_ROOT / "data/manifests/vidal_2019/samsung_30t_vidal2019_25c_manifest.csv"
_RAW_EXISTS = (_REPO_ROOT / "data/raw/Samsung_30T/25degC").exists()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_meas(n: int = 50, **overrides) -> dict:
    meas = {
        "Time": np.arange(n, dtype=float),
        "TimeStamp": np.array([f"2019-03-19 03:{i:02d}:00" for i in range(n)], dtype=object),
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
        "battery": "Samsung_INR21700-30T",
        "paper_battery_name": "Samsung T30",
        "dataset_identity_status": "STRONGLY_SUPPORTED",
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
        "pairing_status": "resolved",
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


def _adapter(tmp_path: Path, rows: list[dict]) -> Samsung30TAdapter:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return Samsung30TAdapter(raw_root=tmp_path.parent, manifest_path=manifest)


class TestSamsungAdapterInit:
    def test_adapter_initializes(self, tmp_path):
        mat = _write_mat(tmp_path, "init.mat")
        adapter = _adapter(tmp_path, [_manifest_row(mat, tmp_path)])
        assert adapter.DATASET_NAME == "samsung_30t"
        assert adapter.SEQUENCE_ID_PREFIX == "samsung_30t_25c"


class TestSamsungManifestLoad:
    def test_real_manifest_loads(self):
        entries = load_manifest(_MANIFEST)
        assert len(entries) == 34
        roles = {e.record_role for e in entries}
        assert roles == {"model_input", "soc_anchor", "excluded"}

    def test_record_role_counts(self):
        entries = load_manifest(_MANIFEST)
        model_input = sum(1 for e in entries if e.record_role == "model_input")
        soc_anchor = sum(1 for e in entries if e.record_role == "soc_anchor")
        excluded = sum(1 for e in entries if e.record_role == "excluded")
        assert model_input == 22
        assert soc_anchor == 2
        assert excluded == 10
        assert model_input + soc_anchor + excluded == len(entries)


class TestSamsungBasicLoad:
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

    def test_metadata_complete(self, tmp_path):
        mat = _write_mat(tmp_path, "meta.mat")
        digest = _sha256(mat)
        row = _manifest_row(mat, tmp_path, sha256=digest)
        meta = _adapter(tmp_path, [row]).load_split("full_train")[0].metadata
        for field in REQUIRED_METADATA_FIELDS:
            assert field in meta
        assert meta["dataset"] == "samsung_30t"
        assert meta["battery"] == "Samsung_INR21700-30T"
        assert meta["paper_battery_name"] == "Samsung T30"
        assert meta["dataset_identity_status"] == "STRONGLY_SUPPORTED"
        assert meta["capacity_hint_ah"] == 3.0
        assert meta["current_sign_convention"] == "as_recorded"
        assert meta["loader_backend"]


class TestSamsungSplitSelection:
    def test_full_train_selection(self):
        entries = select_manifest_entries(load_manifest(_MANIFEST), "full_train")
        assert len(entries) == 17

    def test_partial_train_selection(self):
        entries = select_manifest_entries(load_manifest(_MANIFEST), "partial_train")
        assert len(entries) == 7

    def test_test_selection(self):
        entries = select_manifest_entries(load_manifest(_MANIFEST), "test")
        assert len(entries) == 7

    def test_exclude_selection(self):
        entries = select_manifest_entries(load_manifest(_MANIFEST), "exclude")
        assert len(entries) == 10


class TestSamsungRequestedSplit:
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
        mat = _write_mat(tmp_path, "udds.mat")
        row = _manifest_row(
            mat, tmp_path, full_train=0, partial_train=0, test=1,
            profile="UDDS", paper_profile="UDDS",
            soc_anchor_source="preceding_full_charge",
        )
        rec = _adapter(tmp_path, [row]).load_split("test")[0]
        assert rec.metadata["split"] == "test"


class TestSamsungRecordRole:
    def test_model_input_role(self, tmp_path):
        mat = _write_mat(tmp_path, "mi.mat")
        rec = _adapter(tmp_path, [_manifest_row(mat, tmp_path)]).load_split("full_train")[0]
        assert rec.metadata["record_role"] == "model_input"
        assert rec.metadata["file_type"] == "drive_cycle"

    def test_charge_record_role_model_input(self, tmp_path):
        mat = _write_mat(tmp_path, "chg.mat")
        row = _manifest_row(
            mat, tmp_path, profile="Charge7", paper_profile="CC",
            file_type="charge", soc_init="", soc_anchor_source="preceding_full_charge",
        )
        rec = _adapter(tmp_path, [row]).load_split("full_train")[0]
        assert rec.metadata["record_role"] == "model_input"

    def test_excluded_role(self, tmp_path):
        mat = _write_mat(tmp_path, "exc.mat")
        row = _manifest_row(
            mat, tmp_path, full_train=0, partial_train=0, test=0, exclude=1,
            profile="HPPC", paper_profile="", file_type="characterization",
        )
        rec = _adapter(tmp_path, [row]).load_split("exclude")[0]
        assert rec.metadata["record_role"] == "excluded"


class TestSamsungPaperProfileAndSOC:
    def test_selected_paper_profile_non_empty(self, tmp_path):
        mat = _write_mat(tmp_path, "pp.mat")
        rec = _adapter(tmp_path, [_manifest_row(mat, tmp_path)]).load_split("full_train")[0]
        assert rec.metadata["paper_profile"] == "Mix1"

    def test_empty_paper_profile_fails(self, tmp_path):
        mat = _write_mat(tmp_path, "bad.mat")
        row = _manifest_row(mat, tmp_path, paper_profile="")
        with pytest.raises(AdapterSchemaError, match="paper_profile"):
            _adapter(tmp_path, [row]).load_split("full_train")

    def test_model_input_soc_from_manifest(self, tmp_path):
        mat = _write_mat(tmp_path, "soc.mat")
        row = _manifest_row(
            mat, tmp_path, soc_init="1.0", soc_anchor_source="preceding_full_charge",
        )
        rec = _adapter(tmp_path, [row]).load_split("full_train")[0]
        assert rec.metadata["soc_init"] == 1.0
        assert rec.metadata["soc_anchor_source"] == "preceding_full_charge"


class TestSamsungSHAAndQC:
    def test_sha_match(self, tmp_path):
        mat = _write_mat(tmp_path, "sha.mat")
        digest = _sha256(mat)
        rec = _adapter(tmp_path, [_manifest_row(mat, tmp_path, sha256=digest)]).load_split("full_train")[0]
        assert rec.metadata["source_sha256"] == digest

    def test_sha_mismatch_raises(self, tmp_path):
        mat = _write_mat(tmp_path, "sha_bad.mat")
        row = _manifest_row(mat, tmp_path, sha256="a" * 64)
        with pytest.raises(SHA256MismatchError):
            _adapter(tmp_path, [row]).load_split("full_train")

    def test_missing_field_fail(self, tmp_path):
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

    def test_nan_time_fail(self, tmp_path):
        p = tmp_path / "nan_t.mat"
        time = np.arange(50, dtype=float)
        time[5] = np.nan
        meas = _make_meas(Time=time)
        scipy.io.savemat(str(p), {"meas": meas})
        rec = _adapter(tmp_path, [_manifest_row(p, tmp_path)]).load_split("full_train")[0]
        assert rec.qc.status == "FAIL"

    def test_inf_current_fail(self, tmp_path):
        p = tmp_path / "inf.mat"
        meas = _make_meas()
        meas["Current"][2] = np.inf
        scipy.io.savemat(str(p), {"meas": meas})
        rec = _adapter(tmp_path, [_manifest_row(p, tmp_path)]).load_split("full_train")[0]
        assert rec.qc.status == "FAIL"

    def test_negative_dt_fail(self, tmp_path):
        p = tmp_path / "neg.mat"
        time = np.arange(50, dtype=float)
        time[10] = time[9] - 1.0
        meas = _make_meas(Time=time)
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
        assert rec.qc.status == "WARN"

    def test_ah_reset_warn(self, tmp_path):
        p = tmp_path / "ah.mat"
        ah = np.linspace(0.0, -2.5, 50)
        ah[-1] = -0.001
        meas = _make_meas(Ah=ah)
        scipy.io.savemat(str(p), {"meas": meas})
        rec = _adapter(tmp_path, [_manifest_row(p, tmp_path)]).load_split("full_train")[0]
        assert rec.qc.ah_reset_detected

    def test_temperature_deviation_warn(self, tmp_path):
        p = tmp_path / "temp.mat"
        meas = _make_meas(Battery_Temp_degC=np.full(50, 50.0))
        scipy.io.savemat(str(p), {"meas": meas})
        rec = _adapter(tmp_path, [_manifest_row(p, tmp_path)]).load_split("full_train")[0]
        assert rec.qc.status == "WARN"

    def test_manifest_fail_skipped(self, tmp_path):
        mat = _write_mat(tmp_path, "skip.mat")
        row = _manifest_row(mat, tmp_path, qc_status="FAIL", qc_notes="bad")
        adapter = _adapter(tmp_path, [row])
        assert adapter.load_split("full_train") == []
        summary = adapter.summarize([], split="full_train")
        assert summary.expected_manifest_entries == 1
        assert summary.skipped_manifest_failures == 1


class TestSamsungDeterminismAndSplits:
    def test_one_record_per_file(self, tmp_path):
        mat1 = _write_mat(tmp_path, "a.mat")
        mat2 = _write_mat(tmp_path, "b.mat")
        rows = [
            _manifest_row(mat1, tmp_path),
            _manifest_row(
                mat2, tmp_path, profile="Mixed2", paper_profile="Mix2",
                full_train=1, partial_train=0,
            ),
        ]
        assert len(_adapter(tmp_path, rows).load_split("full_train")) == 2

    def test_order_preserved(self, tmp_path):
        mat1 = _write_mat(tmp_path, "first.mat")
        mat2 = _write_mat(tmp_path, "second.mat")
        rows = [
            _manifest_row(mat1, tmp_path),
            _manifest_row(
                mat2, tmp_path, profile="Mixed2", paper_profile="Mix2",
                full_train=1, partial_train=0,
            ),
        ]
        records = _adapter(tmp_path, rows).load_split("full_train")
        assert records[0].metadata["source_file"].endswith("first.mat")

    def test_no_train_test_overlap(self, tmp_path):
        mat_tr = _write_mat(tmp_path, "tr.mat")
        mat_ts = _write_mat(tmp_path, "ts.mat")
        rows = [
            _manifest_row(mat_tr, tmp_path, full_train=1, test=0),
            _manifest_row(
                mat_ts, tmp_path, full_train=0, partial_train=0, test=1,
                profile="US06", paper_profile="US06",
                soc_anchor_source="preceding_full_charge",
            ),
        ]
        adapter = _adapter(tmp_path, rows)
        train = {r.metadata["source_file"] for r in adapter.load_split("full_train")}
        test = {r.metadata["source_file"] for r in adapter.load_split("test")}
        assert train.isdisjoint(test)

    def test_sequence_id_stable_and_unique(self, tmp_path):
        mat1 = _write_mat(tmp_path, "sid1.mat")
        mat2 = _write_mat(tmp_path, "sid2.mat")
        rows = [
            _manifest_row(mat1, tmp_path),
            _manifest_row(
                mat2, tmp_path, profile="Mixed2", paper_profile="Mix2",
                full_train=1, partial_train=0,
            ),
        ]
        adapter = _adapter(tmp_path, rows)
        r1 = adapter.load_split("full_train")[0]
        r2 = adapter.load_split("full_train")[0]
        assert r1.metadata["sequence_id"] == r2.metadata["sequence_id"]
        ids = [r.metadata["sequence_id"] for r in adapter.load_split("full_train")]
        assert len(ids) == len(set(ids))

    def test_raw_file_immutable(self, tmp_path):
        mat = _write_mat(tmp_path, "imm.mat")
        before = _sha256(mat)
        _adapter(tmp_path, [_manifest_row(mat, tmp_path)]).load_split("full_train")
        assert _sha256(mat) == before

    def test_repeat_load_identical(self, tmp_path):
        mat = _write_mat(tmp_path, "rep.mat")
        adapter = _adapter(tmp_path, [_manifest_row(mat, tmp_path)])
        r1 = adapter.load_split("full_train")[0]
        r2 = adapter.load_split("full_train")[0]
        assert r1.metadata["sequence_id"] == r2.metadata["sequence_id"]
        assert r1.n_rows == r2.n_rows

    def test_summarize_multi_split_no_cross_talk(self, tmp_path):
        mat_tr = _write_mat(tmp_path, "tr2.mat")
        mat_ts = _write_mat(tmp_path, "ts2.mat")
        rows = [
            _manifest_row(mat_tr, tmp_path, full_train=1, partial_train=0, test=0),
            _manifest_row(
                mat_ts, tmp_path, full_train=0, partial_train=0, test=1,
                profile="US06", paper_profile="US06",
                soc_anchor_source="preceding_full_charge",
            ),
        ]
        adapter = _adapter(tmp_path, rows)
        train_records = adapter.load_split("full_train")
        adapter.load_split("test")
        summary = adapter.summarize(train_records, split="full_train")
        assert summary.expected_manifest_entries == 1
        assert summary.split == "full_train"

    def test_summarize_split_mismatch_fails(self, tmp_path):
        mat = _write_mat(tmp_path, "mis.mat")
        rec = _adapter(tmp_path, [_manifest_row(mat, tmp_path)]).load_split("full_train")[0]
        with pytest.raises(AdapterSchemaError, match="split"):
            build_adapter_summary([rec], split="test", stats=None)


@pytest.mark.integration
@pytest.mark.skipif(not _RAW_EXISTS or not _MANIFEST.exists(), reason="Samsung raw data unavailable")
class TestSamsungIntegration:
    @pytest.fixture(scope="class")
    def adapter(self):
        return Samsung30TAdapter(raw_root=_REPO_ROOT, manifest_path=_MANIFEST)

    def test_full_train_loads(self, adapter):
        records = adapter.load_split("full_train")
        assert len(records) > 0

    def test_partial_train_loads(self, adapter):
        assert len(adapter.load_split("partial_train")) == 7

    def test_test_loads(self, adapter):
        assert len(adapter.load_split("test")) == 7

    def test_exclude_loads(self, adapter):
        assert len(adapter.load_split("exclude")) == 10

    def test_selected_counts_match_manifest(self, adapter):
        for split in ("full_train", "partial_train", "test", "exclude"):
            entries = select_manifest_entries(load_manifest(_MANIFEST), split)
            records = adapter.load_split(split)
            expected = len([e for e in entries if e.qc_status != "FAIL"])
            assert len(records) == expected

    def test_source_sha256_matches_manifest(self, adapter):
        entries = {e.relative_path: e.sha256 for e in load_manifest(_MANIFEST)}
        for rec in adapter.load_split("full_train") + adapter.load_split("test"):
            assert rec.metadata["source_sha256"] == entries[rec.metadata["source_file"]]

    def test_all_records_nonempty_with_required_columns(self, adapter):
        for split in ("full_train", "partial_train", "test", "exclude"):
            for rec in adapter.load_split(split):
                assert rec.n_rows > 0
                assert rec.has_required_columns

    def test_paper_profile_from_manifest(self, adapter):
        rec = next(r for r in adapter.load_split("full_train") if r.metadata["profile"] == "Mixed1")
        assert rec.metadata["paper_profile"] == "Mix1"

    def test_record_roles_consistent(self, adapter):
        for split in ("full_train", "partial_train", "test", "exclude"):
            for rec in adapter.load_split(split):
                if rec.metadata["record_role"] == "model_input":
                    assert rec.metadata["file_type"] in ("drive_cycle", "charge")
                elif rec.metadata["record_role"] == "excluded":
                    assert "exclude" in rec.metadata["manifest_splits"]

    def test_model_input_soc_metadata(self, adapter):
        records = adapter.load_split("full_train") + adapter.load_split("test")
        for rec in records:
            if (
                rec.metadata["record_role"] == "model_input"
                and rec.metadata["file_type"] == "drive_cycle"
            ):
                assert rec.metadata["soc_init"] is not None
                assert 0.0 <= rec.metadata["soc_init"] <= 1.0
                assert rec.metadata["soc_anchor_source"]
                assert rec.metadata["soc_anchor_source"] != "not_applicable"

    def test_no_train_test_overlap(self, adapter):
        train = {r.metadata["source_file"] for r in adapter.load_split("full_train")}
        test = {r.metadata["source_file"] for r in adapter.load_split("test")}
        assert train.isdisjoint(test)

    def test_sequence_ids_unique(self, adapter):
        records = adapter.load_split("full_train") + adapter.load_split("test")
        ids = [r.metadata["sequence_id"] for r in records]
        assert len(ids) == len(set(ids))

    def test_raw_files_immutable(self, adapter):
        rec = adapter.load_split("full_train")[0]
        path = _REPO_ROOT / rec.metadata["source_file"]
        before = _sha256(path)
        adapter.load_split("full_train")
        assert _sha256(path) == before

    def test_integration_summary_report(self, adapter, capsys):
        for split in ("full_train", "partial_train", "test", "exclude"):
            records = adapter.load_split(split)
            summary = adapter.summarize(records, split=split)
            print(
                f"{split}: expected={summary.expected_manifest_entries} "
                f"loaded={summary.loaded_records} skipped={summary.skipped_manifest_failures} "
                f"pass={summary.pass_count} warn={summary.warn_count} fail={summary.fail_count} "
                f"rows={summary.total_rows} dup_ts={summary.duplicate_timestamp_count} "
                f"ah_reset={summary.ah_reset_count}"
            )
            assert summary.expected_manifest_entries == (
                summary.loaded_records + summary.skipped_manifest_failures
            )
