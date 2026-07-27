"""Manifest rules for the Liu et al. 25 C -> 0 C experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from battery_benchmark.utils.hashing import sha256_file

REQUIRED_COLUMNS = {
    "record_id",
    "dataset",
    "domain_id",
    "temperature_c",
    "role",
    "profile",
    "relative_path",
    "label_available",
    "label_usage",
    "label_source",
    "source_sha256",
}
ROLE_LABEL_POLICY = {
    "source_train": (True, "train"),
    "source_test": (True, "evaluation_only"),
    "target_adapt": (False, "forbidden"),
    "target_test": (True, "evaluation_only"),
}
TEST_PROFILES = {"LA92", "UDDS", "US06", "HWFET", "NN"}


def load_experiment_manifest(
    manifest_path: str | Path,
    data_root: str | Path,
    *,
    verify_hashes: bool = False,
) -> pd.DataFrame:
    manifest_path, data_root = Path(manifest_path), Path(data_root)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(manifest)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if manifest.empty or manifest["record_id"].duplicated().any():
        raise ValueError("Manifest record_id values must be non-empty and unique")

    manifest["label_available"] = manifest["label_available"].map(
        {True: True, False: False, "true": True, "false": False}
    )
    if manifest["label_available"].isna().any():
        raise ValueError("label_available must contain only true/false")

    unknown_roles = set(manifest["role"]) - set(ROLE_LABEL_POLICY)
    if unknown_roles:
        raise ValueError(f"Unknown manifest roles: {sorted(unknown_roles)}")
    for row in manifest.itertuples(index=False):
        expected = ROLE_LABEL_POLICY[row.role]
        actual = (bool(row.label_available), row.label_usage)
        if actual != expected:
            raise ValueError(
                f"{row.record_id}: role {row.role} requires label policy {expected}, got {actual}"
            )
        if row.label_available and not row.label_source:
            raise ValueError(f"{row.record_id}: labelled rows require label_source")
        if not row.label_available and row.label_source:
            raise ValueError(f"{row.record_id}: forbidden labels must not declare label_source")

    counts = manifest["role"].value_counts().to_dict()
    if counts.get("source_train") != 4 or counts.get("target_adapt") != 1:
        raise ValueError("Expected four source_train rows and one target_adapt row")
    for role in ("source_test", "target_test"):
        profiles = set(manifest.loc[manifest["role"] == role, "profile"])
        if profiles != TEST_PROFILES:
            raise ValueError(
                f"{role} profiles must be {sorted(TEST_PROFILES)}, got {sorted(profiles)}"
            )

    resolved = []
    for row in manifest.itertuples(index=False):
        path = data_root / row.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {path}")
        if verify_hashes and sha256_file(path) != row.source_sha256:
            raise ValueError(f"SHA-256 mismatch: {row.record_id}")
        resolved.append(path)
    manifest["path"] = resolved
    return manifest
