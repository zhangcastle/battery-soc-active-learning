"""Cross-run CMU protocol consistency check (hash level).

A benchmark run is only comparable with the other CMU transfer runs
(Backbone / DARM / SFTNN / DTNN) when it read the *same* manifest and the
*same* per-cell CMU input artifacts. Directory names such as ``v1`` / ``v2`` /
``ns_relabel`` prove nothing, so this module compares SHA-256 digests:

* every cell's CMU artifact digest against the manifest's
  ``source_artifact_sha256`` column, and
* the manifest digest and the per-cell artifact digests against what each
  *reference run root* (an already-finished Backbone/DARM/SFTNN/DTNN run)
  recorded in its own ``run_metadata.json`` (``cmu_manifest_sha256``) and
  per-cell ``results.json`` (``cmu_input_sha256``).

Any mismatch, or a reference root that records nothing verifiable, raises
:class:`ProtocolMismatchError`. Nothing here reads or writes SOC labels.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from battery_benchmark.utils.hashing import sha256_file

METADATA_FILENAMES = ("run_metadata.json", "run.json")


class ProtocolMismatchError(ValueError):
    """Raised when a run's CMU inputs do not match the common protocol."""


def manifest_artifact_hashes(manifest: pd.DataFrame, cell_id: str) -> set[str]:
    """Set of ``source_artifact_sha256`` values the manifest lists for a cell."""
    return set(
        manifest.loc[manifest["cell_id"] == cell_id, "source_artifact_sha256"].astype(str)
    )


def _read_reference_metadata(root: Path) -> dict | None:
    for name in METADATA_FILENAMES:
        path = root / name
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:  # unreadable -> unverifiable
                raise ProtocolMismatchError(f"{path}: not valid JSON ({error})") from error
            if isinstance(payload, dict):
                return payload
    return None


def _reference_cell_hashes(root: Path) -> dict[str, set[str]]:
    """``cell_id -> {cmu_input_sha256}`` collected from every ``results.json``
    below ``root`` (any nesting depth: per-model, per-seed, per-repeat)."""
    found: dict[str, set[str]] = {}
    for path in root.rglob("results.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        cell, digest = payload.get("cell_id"), payload.get("cmu_input_sha256")
        if cell and digest:
            found.setdefault(str(cell), set()).add(str(digest))
    return found


def verify_cmu_protocol(
    manifest_path: Path,
    cell_artifacts: Mapping[str, Path],
    *,
    reference_run_roots: Iterable[Path] = (),
    require_reference: bool = True,
) -> dict[str, object]:
    """Verify hashes and return a provenance record for ``run_metadata.json``.

    Parameters
    ----------
    manifest_path:
        The manifest CSV this run will read.
    cell_artifacts:
        ``cell_id -> CMU dynamic-SOC parquet`` actually used by this run.
    reference_run_roots:
        Finished runs of the other frameworks to compare against.
    require_reference:
        When True (default) at least one reference root must be given and each
        must verify at least one hash; when False the record states plainly
        that the cross-run comparison was skipped.
    """
    manifest_path = Path(manifest_path)
    manifest = pd.read_csv(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)

    artifact_sha256: dict[str, str] = {}
    for cell_id, path in cell_artifacts.items():
        digest = sha256_file(path)
        expected = manifest_artifact_hashes(manifest, cell_id)
        if expected != {digest}:
            raise ProtocolMismatchError(
                f"{cell_id}: CMU input hash {digest} does not match manifest "
                f"source_artifact_sha256 {sorted(expected)}"
            )
        artifact_sha256[cell_id] = digest

    roots = [Path(root) for root in reference_run_roots]
    if require_reference and not roots:
        raise ProtocolMismatchError(
            "No reference run root given: pass --reference-run-root (a finished "
            "Backbone/DARM/SFTNN/DTNN run) or explicitly skip the cross-run "
            "check with --skip-reference-check"
        )

    references: list[dict[str, object]] = []
    for root in roots:
        if not root.is_dir():
            raise ProtocolMismatchError(f"Reference run root {root} does not exist")
        metadata = _read_reference_metadata(root)
        verified_manifest = False
        if metadata is not None and metadata.get("cmu_manifest_sha256"):
            if metadata["cmu_manifest_sha256"] != manifest_sha256:
                raise ProtocolMismatchError(
                    f"{root}: reference manifest sha256 {metadata['cmu_manifest_sha256']} "
                    f"!= this run's {manifest_sha256}"
                )
            verified_manifest = True
        reference_cells = _reference_cell_hashes(root)
        verified_cells: list[str] = []
        for cell_id, digest in artifact_sha256.items():
            recorded = reference_cells.get(cell_id)
            if not recorded:
                continue
            if recorded != {digest}:
                raise ProtocolMismatchError(
                    f"{root}: reference {cell_id} cmu_input_sha256 {sorted(recorded)} "
                    f"!= this run's {digest}"
                )
            verified_cells.append(cell_id)
        if not verified_manifest and not verified_cells:
            raise ProtocolMismatchError(
                f"{root}: records neither cmu_manifest_sha256 nor any per-cell "
                "cmu_input_sha256 -- cannot verify protocol consistency"
            )
        references.append(
            {
                "root": str(root),
                "manifest_sha256_verified": verified_manifest,
                "cells_verified": sorted(verified_cells),
            }
        )

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "cmu_artifact_sha256": artifact_sha256,
        "manifest_hash_check": "passed",
        "reference_runs": references,
        "reference_check": "passed" if roots else "skipped_by_user",
    }
