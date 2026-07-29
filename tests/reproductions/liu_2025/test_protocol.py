from pathlib import Path

import pandas as pd
import pytest

from battery_benchmark.reproductions.liu_2025.protocol import (
    load_experiment_manifest,
    source_protocol_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "data/manifests/liu_2025/panasonic_25c_to_0c.csv"
DATA_ROOT = REPO_ROOT / "data/raw/Panasonic_18650PF"


@pytest.mark.integration
def test_real_manifest_has_strict_target_label_policy():
    manifest = load_experiment_manifest(MANIFEST, DATA_ROOT)
    target_adapt = manifest.loc[manifest["role"] == "target_adapt"].iloc[0]
    assert target_adapt["label_available"] is False or not target_adapt["label_available"]
    assert target_adapt["label_usage"] == "forbidden"
    assert set(manifest.loc[manifest["role"] == "target_test", "profile"]) == {
        "LA92",
        "UDDS",
        "US06",
        "HWFET",
        "NN",
    }


def test_manifest_rejects_target_adaptation_labels(tmp_path):
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    manifest.loc[manifest["role"] == "target_adapt", "label_available"] = True
    path = tmp_path / "bad.csv"
    manifest.to_csv(path, index=False)
    with pytest.raises(ValueError, match="requires label policy"):
        load_experiment_manifest(path, DATA_ROOT)


def test_manifest_rejects_duplicate_test_profiles(tmp_path):
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    duplicate = manifest.loc[manifest["record_id"] == "source_test_la92"].copy()
    duplicate["record_id"] = "source_test_la92_duplicate"
    path = tmp_path / "duplicate.csv"
    pd.concat([manifest, duplicate], ignore_index=True).to_csv(path, index=False)

    with pytest.raises(ValueError, match="one file per profile"):
        load_experiment_manifest(path, DATA_ROOT)


def test_source_protocol_hash_ignores_target_rows():
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    changed = manifest.copy()
    changed.loc[changed["role"].str.startswith("target"), "domain_id"] = "another_target"

    assert source_protocol_sha256(changed) == source_protocol_sha256(manifest)


def test_target_test_may_omit_an_unavailable_profile(tmp_path):
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    manifest = manifest.loc[manifest["record_id"] != "target_test_us06"]
    path = tmp_path / "missing_us06.csv"
    manifest.to_csv(path, index=False)

    loaded = load_experiment_manifest(path, DATA_ROOT)

    assert "US06" not in set(loaded.loc[loaded["role"] == "target_test", "profile"])
