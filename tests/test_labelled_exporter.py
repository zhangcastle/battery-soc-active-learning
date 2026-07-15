"""Tests for labelled dataset exporter."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from battery_benchmark.data.adapter_qc import QCResult
from battery_benchmark.data.sequence_record import SequenceRecord
from battery_benchmark.labels import (
    SOCCalculator,
    export_labelled_dataset,
    validate_exported_labelled_dataset,
)


def _labelled_record(seq_id: str = "test_seq") -> SequenceRecord:
    import numpy as np

    n = 10
    ah = np.linspace(0.0, -1.0, n)
    df = pd.DataFrame({
        "time_s": np.arange(n, dtype=float),
        "timestamp": [None] * n,
        "voltage_v": np.linspace(4.0, 3.5, n),
        "current_a": np.full(n, -1.0),
        "temperature_c": np.full(n, 25.0),
        "cycler_ah": ah,
    })
    base = SequenceRecord(
        data=df,
        metadata={
            "dataset": "lg_hg2",
            "battery": "lg_hg2",
            "source_file": "data/a.mat",
            "source_sha256": "abc123",
            "sequence_id": seq_id,
            "profile": "Mixed1",
            "paper_profile": "Mix1",
            "file_type": "drive_cycle",
            "record_role": "model_input",
            "split": "full_train",
            "temperature_nominal_c": 25.0,
            "soc_init": 1.0,
            "soc_anchor_source": "preceding_full_charge",
            "capacity_hint_ah": 3.0,
        },
        qc=QCResult(),
    )
    return SOCCalculator().calculate_vidal(base)


@pytest.fixture
def labelled_bundle():
    return {
        "lg_hg2": {
            "full_train": [_labelled_record("lg_a"), _labelled_record("lg_b")],
        }
    }


class TestExporter:
    def test_export_creates_files(self, tmp_path, labelled_bundle):
        root = export_labelled_dataset(labelled_bundle, tmp_path, overwrite=False)
        assert (root / "dataset_manifest.json").exists()
        assert list((root / "lg_hg2" / "full_train").glob("*.parquet"))

    def test_existing_dir_fails(self, tmp_path, labelled_bundle):
        out = tmp_path / "labelled"
        export_labelled_dataset(labelled_bundle, out)
        with pytest.raises(FileExistsError):
            export_labelled_dataset(labelled_bundle, out)

    def test_overwrite_succeeds(self, tmp_path, labelled_bundle):
        out = tmp_path / "labelled"
        export_labelled_dataset(labelled_bundle, out)
        export_labelled_dataset(labelled_bundle, out, overwrite=True)
        validate_exported_labelled_dataset(out / "vidal_compatible_v1")

    def test_exclude_rejected(self, tmp_path):
        rec = _labelled_record()
        rec.metadata["split"] = "exclude"
        with pytest.raises(ValueError, match="exclude"):
            export_labelled_dataset({"lg_hg2": {"exclude": [rec]}}, tmp_path)

    def test_validate_reload(self, tmp_path, labelled_bundle):
        root = export_labelled_dataset(labelled_bundle, tmp_path)
        validate_exported_labelled_dataset(root)

    def test_manifest_has_version(self, tmp_path, labelled_bundle):
        root = export_labelled_dataset(labelled_bundle, tmp_path)
        manifest = json.loads((root / "dataset_manifest.json").read_text())
        assert manifest["dataset_version"] == "vidal_compatible_v1"
        assert manifest["excluded_records_exported"] is False

    def test_sequence_ids_unique(self, tmp_path, labelled_bundle):
        root = export_labelled_dataset(labelled_bundle, tmp_path)
        manifest = json.loads((root / "dataset_manifest.json").read_text())
        ids = [r["sequence_id"] for r in manifest["records"]]
        assert len(ids) == len(set(ids))

    def test_dedupes_multi_split_records(self, tmp_path):
        rec = _labelled_record("shared_seq")
        partial = SequenceRecord(
            data=rec.data,
            metadata={**rec.metadata, "split": "partial_train"},
            qc=rec.qc,
        )
        full = SequenceRecord(
            data=rec.data,
            metadata={**rec.metadata, "split": "full_train"},
            qc=rec.qc,
        )
        bundle = {
            "lg_hg2": {
                "full_train": [full],
                "partial_train": [partial],
            }
        }
        root = export_labelled_dataset(bundle, tmp_path)
        manifest = json.loads((root / "dataset_manifest.json").read_text())
        assert len(manifest["records"]) == 1
        assert manifest["records"][0]["split"] == "full_train"
        validate_exported_labelled_dataset(root)

    def test_missing_file_validation_fails(self, tmp_path, labelled_bundle):
        root = export_labelled_dataset(labelled_bundle, tmp_path)
        parquet = next((root / "lg_hg2" / "full_train").glob("*.parquet"))
        parquet.unlink()
        with pytest.raises(FileNotFoundError):
            validate_exported_labelled_dataset(root)

    def test_row_count_mismatch_fails(self, tmp_path, labelled_bundle):
        root = export_labelled_dataset(labelled_bundle, tmp_path)
        manifest_path = root / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["records"][0]["n_rows"] = 99999
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="Row count"):
            validate_exported_labelled_dataset(root)
