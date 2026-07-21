from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from battery_benchmark.datasets.cmu_evtol.raw import (
    ARTICLE_ID,
    ASSET_INVENTORY_COLUMNS,
    CANONICAL_COLUMNS,
    CMUEmptyFileError,
    CMUInvalidCellIDError,
    CMURawSchemaError,
    CMU_V3_CELL_IDS,
    DATASET_VERSION,
    IMPEDANCE_COLUMNS,
    RAW_COLUMNS,
    RAW_QC_SUMMARY_COLUMNS,
    build_asset_inventory,
    build_raw_qc_summary,
    load_cmu_v3_main_file,
    positive_delta_statistics,
)
from scripts.datasets.cmu_evtol.qc_raw import DEFAULT_QC_OUTPUT, resolve_output_path


REAL_RAW_ROOT = Path("data/raw/cmu_evtol_vtc6/figshare_14226830")
ASSET_MANIFEST = Path("data/manifests/cmu_evtol/cmu_evtol_vtc6_asset_inventory_v3.csv")
RAW_QC_MANIFEST = Path("data/manifests/cmu_evtol/cmu_evtol_vtc6_raw_qc_v3_120s.csv")


def _row(
    time_s: float,
    *,
    cycle_number: int = 1,
    ns: int = 0,
) -> list[float | int]:
    return [time_s, 3.8, -1000.0, 0.0, 0.0, 1.0, 250.0, 25.0, cycle_number, ns]


def _write_main(
    root: Path,
    cell_id: str = "VAH01",
    rows: list[list[float | int]] | None = None,
    header: tuple[str, ...] = RAW_COLUMNS,
) -> Path:
    path = root / f"{cell_id}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows if rows is not None else [_row(0.0)])
    return path


def _load(path: Path, threshold: float = 10.0, boundaries=None):
    return load_cmu_v3_main_file(
        path,
        gap_threshold_s=threshold,
        source_boundary_rows=boundaries,
        repo_root=path.parent,
    )


def _write_impedance(
    root: Path,
    cell_id: str = "VAH01",
    header: tuple[str, ...] = IMPEDANCE_COLUMNS,
) -> Path:
    path = root / f"{cell_id}_impedance.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow([0.02, 0.04, 0.02, 0.03, 1])
    return path


def test_strictly_increasing_time_produces_one_block_and_provenance(tmp_path: Path) -> None:
    path = _write_main(tmp_path, rows=[_row(0.0), _row(1.0), _row(2.0)])
    result = _load(path)

    assert tuple(result.data.columns) == CANONICAL_COLUMNS
    assert result.data["continuous_block_id"].tolist() == [0, 0, 0]
    assert result.data["source_row"].tolist() == [1, 2, 3]
    assert result.data["source_file"].tolist() == ["VAH01.csv"] * 3
    assert result.metadata["article_id"] == ARTICLE_ID
    assert result.metadata["dataset_version"] == DATASET_VERSION
    assert result.metadata["input_row_count"] == 3
    assert result.metadata["output_row_count"] == 3
    assert result.metadata["block_count"] == 1
    assert result.metadata["filtered_row_count"] == 0


def test_default_gap_threshold_is_explicitly_exploratory(tmp_path: Path) -> None:
    result = load_cmu_v3_main_file(
        _write_main(tmp_path, rows=[_row(0.0), _row(1.0)]),
        repo_root=tmp_path,
    )
    assert result.metadata["gap_threshold_s"] == 120.0
    assert result.metadata["gap_threshold_source"] == "exploratory_loader_default"


@pytest.mark.parametrize(
    ("times", "expected_ids", "reason"),
    [
        ([0.0, 1.0, 1.0], [0, 0, 1], "time_duplicate"),
        ([0.0, 2.0, 1.0], [0, 0, 1], "time_reversal"),
        ([0.0, 1.0, 20.0], [0, 0, 1], "gap_exceeded"),
    ],
)
def test_time_discontinuities_start_new_blocks(
    tmp_path: Path,
    times: list[float],
    expected_ids: list[int],
    reason: str,
) -> None:
    path = _write_main(tmp_path, rows=[_row(value) for value in times])
    result = _load(path, threshold=10.0)
    assert result.data["continuous_block_id"].tolist() == expected_ids
    assert result.metadata["boundary_reason_counts"][reason] == 1


def test_non_finite_time_is_retained_and_separates_both_sides(tmp_path: Path) -> None:
    path = _write_main(tmp_path, rows=[_row(0.0), _row(float("nan")), _row(2.0)])
    result = _load(path)

    assert len(result.data) == 3
    assert np.isnan(result.data.loc[1, "time_s"])
    assert result.data["continuous_block_id"].tolist() == [0, 1, 2]
    counts = result.metadata["boundary_reason_counts"]
    assert counts["current_time_non_finite"] == 1
    assert counts["previous_time_non_finite"] == 1


def test_cycle_number_and_ns_changes_do_not_create_blocks(tmp_path: Path) -> None:
    path = _write_main(
        tmp_path,
        rows=[
            _row(0.0, cycle_number=10, ns=0),
            _row(1.0, cycle_number=1, ns=9),
            _row(2.0, cycle_number=1, ns=0),
        ],
    )
    result = _load(path)
    assert result.data["continuous_block_id"].tolist() == [0, 0, 0]


def test_single_row_is_one_block(tmp_path: Path) -> None:
    result = _load(_write_main(tmp_path, rows=[_row(5.0)]))
    assert result.data["continuous_block_id"].tolist() == [0]
    assert result.metadata["block_count"] == 1


def test_explicit_source_concatenation_boundary_is_retained(tmp_path: Path) -> None:
    path = _write_main(tmp_path, rows=[_row(0.0), _row(1.0), _row(2.0)])
    result = _load(path, boundaries={2})
    assert result.data["source_row"].tolist() == [1, 2, 3]
    assert result.data["continuous_block_id"].tolist() == [0, 1, 1]
    assert result.data.loc[1, "continuous_block_boundary_reason"] == (
        "source_concat_boundary"
    )


def test_missing_required_column_fails_clearly(tmp_path: Path) -> None:
    header = tuple(column for column in RAW_COLUMNS if column != "Ns")
    path = _write_main(tmp_path, header=header, rows=[_row(0.0)[:-1]])
    with pytest.raises(CMURawSchemaError, match=r"missing=\['Ns'\]"):
        _load(path)


def test_duplicate_column_fails_before_pandas_mangles_it(tmp_path: Path) -> None:
    duplicate_header = (*RAW_COLUMNS[:-1], "cycleNumber")
    path = _write_main(tmp_path, header=duplicate_header)
    with pytest.raises(CMURawSchemaError, match="Duplicate raw CSV column"):
        _load(path)


def test_zero_byte_and_header_only_files_are_rejected(tmp_path: Path) -> None:
    zero_byte = tmp_path / "VAH01.csv"
    zero_byte.touch()
    with pytest.raises(CMUEmptyFileError, match="empty"):
        _load(zero_byte)

    header_only = _write_main(tmp_path, rows=[])
    with pytest.raises(CMUEmptyFileError, match="no data rows"):
        _load(header_only)


def test_invalid_cell_id_is_rejected(tmp_path: Path) -> None:
    path = _write_main(tmp_path, cell_id="VAH99")
    with pytest.raises(CMUInvalidCellIDError, match="Unknown CMU Version 3 cell ID"):
        _load(path)


def test_asset_inventory_allows_vah26_without_impedance(tmp_path: Path) -> None:
    main_path = _write_main(tmp_path, cell_id="VAH26", rows=[_row(0.0), _row(1.0)])
    inventory = build_asset_inventory(tmp_path, cell_ids=["VAH26"])
    row = inventory.iloc[0]

    assert tuple(inventory.columns) == ASSET_INVENTORY_COLUMNS
    assert row["main_file_present"]
    assert not row["impedance_file_present"]
    assert row["file_size_bytes"] == main_path.stat().st_size
    assert len(row["sha256"]) == 64
    assert row["row_count"] == 2
    assert json.loads(row["observed_columns"]) == list(RAW_COLUMNS)
    assert pd.isna(row["impedance_file_size_bytes"])
    assert pd.isna(row["impedance_sha256"])
    assert pd.isna(row["impedance_row_count"])
    assert pd.isna(row["impedance_observed_columns"])


def test_asset_inventory_validates_main_header_strictly(tmp_path: Path) -> None:
    malformed = tuple(column for column in RAW_COLUMNS if column != "Ns")
    _write_main(tmp_path, header=malformed, rows=[_row(0.0)[:-1]])
    with pytest.raises(CMURawSchemaError, match="Raw CSV schema mismatch.*VAH01.csv"):
        build_asset_inventory(tmp_path, cell_ids=["VAH01"])


def test_asset_inventory_records_impedance_provenance(tmp_path: Path) -> None:
    _write_main(tmp_path)
    impedance = _write_impedance(tmp_path)
    row = build_asset_inventory(tmp_path, cell_ids=["VAH01"]).iloc[0]

    assert row["impedance_file_present"]
    assert row["impedance_file_size_bytes"] == impedance.stat().st_size
    assert len(row["impedance_sha256"]) == 64
    assert row["impedance_row_count"] == 1
    assert json.loads(row["impedance_observed_columns"]) == list(IMPEDANCE_COLUMNS)


def test_malformed_impedance_header_cannot_enter_inventory(tmp_path: Path) -> None:
    _write_main(tmp_path)
    _write_impedance(tmp_path, header=IMPEDANCE_COLUMNS[:-1])
    with pytest.raises(CMURawSchemaError, match="Impedance CSV schema mismatch"):
        build_asset_inventory(tmp_path, cell_ids=["VAH01"])


def test_positive_delta_statistics_ignore_non_finite_and_non_positive() -> None:
    stats = positive_delta_statistics([0.0, 1.0, 1.0, np.nan, 5.0, 8.0, 6.0, 10.0])
    assert stats["min_positive_delta_s"] == 1.0
    assert stats["median_positive_delta_s"] == 3.0
    assert stats["p95_positive_delta_s"] == pytest.approx(3.9)
    assert stats["p99_positive_delta_s"] == pytest.approx(3.98)
    assert stats["max_positive_delta_s"] == 4.0


def test_raw_qc_summary_matches_loader_block_metadata(tmp_path: Path) -> None:
    path = _write_main(
        tmp_path,
        rows=[_row(0.0), _row(1.0), _row(1.0), _row(20.0)],
    )
    loaded = _load(path, threshold=10.0)
    summary = build_raw_qc_summary(
        tmp_path,
        gap_threshold_s=10.0,
        cell_ids=["VAH01"],
        repo_root=tmp_path,
    )
    row = summary.iloc[0]

    assert tuple(summary.columns) == RAW_QC_SUMMARY_COLUMNS
    assert row["block_count"] == loaded.metadata["block_count"] == 3
    assert row["time_duplicate_count"] == 1
    assert row["gap_exceeded_count"] == 1
    assert row["main_sha256"] == loaded.metadata["source_sha256"]
    assert row["gap_threshold_source"] == "explicit_argument"


def test_default_threshold_uses_default_120s_output() -> None:
    assert resolve_output_path(None, None) == DEFAULT_QC_OUTPUT
    assert resolve_output_path(120.0, None) == DEFAULT_QC_OUTPUT


def test_non_default_threshold_requires_explicit_output() -> None:
    with pytest.raises(ValueError, match="--output is required"):
        resolve_output_path(60.0, None)


def test_non_default_threshold_accepts_explicit_output(tmp_path: Path) -> None:
    custom = tmp_path / "cmu_raw_qc_60s.csv"
    assert resolve_output_path(60.0, custom) == custom


def test_committed_asset_and_qc_manifests_are_consistent() -> None:
    assets = pd.read_csv(ASSET_MANIFEST, dtype={"article_id": str})
    qc = pd.read_csv(RAW_QC_MANIFEST, dtype={"article_id": str})

    assert len(assets) == len(qc) == 22
    assert assets["cell_id"].is_unique
    assert qc["cell_id"].is_unique
    assert set(assets["cell_id"]) == set(qc["cell_id"]) == set(CMU_V3_CELL_IDS)

    merged = assets.merge(qc, on="cell_id", suffixes=("_asset", "_qc"), validate="one_to_one")
    assert merged["sha256"].eq(merged["main_sha256"]).all()
    assert merged["impedance_file_present_asset"].eq(
        merged["impedance_file_present_qc"]
    ).all()
    assert merged["impedance_sha256_asset"].fillna("").eq(
        merged["impedance_sha256_qc"].fillna("")
    ).all()
    assert merged["row_count_asset"].eq(merged["row_count_qc"]).all()

    assert qc["gap_threshold_s"].eq(120.0).all()
    assert qc["gap_threshold_source"].eq("exploratory_loader_default").all()
    assert int(qc["row_count"].sum()) == 15_309_317
    assert int(qc["block_count"].sum()) == 336
    assert int(qc["time_duplicate_count"].sum()) == 310
    assert int(qc["gap_exceeded_count"].sum()) == 4
    assert int(qc["time_reversal_count"].sum()) == 0
    assert int(qc["current_time_non_finite_count"].sum()) == 0
    assert int(qc["previous_time_non_finite_count"].sum()) == 0

    assert qc["git_revision"].nunique() == 1
    assert qc["git_revision"].str.fullmatch(r"[0-9a-f]{40}").all()
    assert qc["generated_at_utc"].nunique() == 1
    assert qc["git_dirty"].eq(False).all()  # noqa: E712


@pytest.mark.integration
def test_real_v3_asset_inventory_when_raw_package_is_available() -> None:
    if not REAL_RAW_ROOT.is_dir():
        pytest.skip(f"Optional CMU raw package is not present: {REAL_RAW_ROOT}")

    inventory = build_asset_inventory(REAL_RAW_ROOT)
    assert len(inventory) == 22
    assert set(inventory["cell_id"]) == set(CMU_V3_CELL_IDS)
    assert inventory["main_file_present"].all()
    missing_impedance = set(inventory.loc[~inventory["impedance_file_present"], "cell_id"])
    assert missing_impedance == {"VAH26"}
    assert (inventory["file_size_bytes"] > 0).all()
    assert (inventory["row_count"] > 0).all()
    assert inventory["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    impedance = inventory.loc[inventory["impedance_file_present"]]
    assert len(impedance) == 21
    assert (impedance["impedance_file_size_bytes"] > 0).all()
    assert (impedance["impedance_row_count"] > 0).all()
    assert impedance["impedance_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert impedance["impedance_observed_columns"].map(json.loads).map(
        lambda columns: tuple(columns) == IMPEDANCE_COLUMNS
    ).all()
