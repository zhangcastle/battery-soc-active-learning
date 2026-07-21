"""Integration tests for Vidal-compatible SOC labelling across all datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from battery_benchmark.data.lg_hg2 import LGHG2Adapter
from battery_benchmark.data.panasonic import PanasonicAdapter
from battery_benchmark.data.samsung_30t import Samsung30TAdapter
from battery_benchmark.labels import (
    SOCCalculator,
    export_labelled_dataset,
    validate_exported_labelled_dataset,
)

REPO = Path(__file__).resolve().parents[3]
SPLITS = ("full_train", "partial_train", "test")

DATASETS = {
    "panasonic_18650pf": (
        PanasonicAdapter,
        REPO / "data/manifests/vidal_2019/panasonic_vidal2019_25c_manifest.csv",
        REPO / "data/raw/Panasonic_18650PF/25degC",
    ),
    "lg_hg2": (
        LGHG2Adapter,
        REPO / "data/manifests/vidal_2019/lg_hg2_vidal2019_25c_manifest.csv",
        REPO / "data/raw/LG_HG2/25degC",
    ),
    "samsung_30t": (
        Samsung30TAdapter,
        REPO / "data/manifests/vidal_2019/samsung_30t_vidal2019_25c_manifest.csv",
        REPO / "data/raw/Samsung_30T/25degC",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.integration
class TestVidalLabelsIntegration:
    @pytest.fixture(params=list(DATASETS.keys()))
    def dataset_name(self, request):
        return request.param

    @pytest.fixture
    def adapter(self, dataset_name):
        cls, manifest, raw = DATASETS[dataset_name]
        if not raw.exists():
            pytest.skip(f"Raw data missing for {dataset_name}")
        return cls(raw_root=REPO, manifest_path=manifest)

    def test_all_model_input_labelled(self, adapter, dataset_name):
        calc = SOCCalculator()
        for split in SPLITS:
            for rec in adapter.load_split(split):
                if rec.metadata["record_role"] != "model_input":
                    continue
                labelled = calc.calculate_vidal(rec)
                assert len(labelled.data) == rec.n_rows
                assert "soc" in labelled.data.columns
                assert labelled.metadata["label_source"] == "vidal_compatible_cycler_ah"

    def test_drive_and_charge_both_labelled(self, adapter):
        calc = SOCCalculator()
        types = set()
        for split in SPLITS:
            for rec in adapter.load_split(split):
                if rec.metadata["record_role"] != "model_input":
                    continue
                calc.calculate_vidal(rec)
                types.add(rec.metadata["file_type"])
        assert "drive_cycle" in types
        assert "charge" in types

    def test_raw_hash_unchanged(self, adapter):
        rec = next(
            r for split in SPLITS for r in adapter.load_split(split)
            if r.metadata["record_role"] == "model_input"
        )
        path = REPO / rec.metadata["source_file"]
        before = _sha256(path)
        SOCCalculator().calculate_vidal(rec)
        assert _sha256(path) == before

    def test_export_reload(self, adapter, dataset_name, tmp_path):
        calc = SOCCalculator()
        bundle: dict[str, dict[str, list]] = {dataset_name: {}}
        for split in SPLITS:
            records = [
                calc.calculate_vidal(r)
                for r in adapter.load_split(split)
                if r.metadata["record_role"] == "model_input"
            ]
            bundle[dataset_name][split] = records
        root = export_labelled_dataset(bundle, tmp_path, overwrite=True)
        validate_exported_labelled_dataset(root)
