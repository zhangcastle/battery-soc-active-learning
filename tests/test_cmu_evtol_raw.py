from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from battery_benchmark.data.cmu_evtol_raw import (
    ARTICLE_ID,
    ASSET_INVENTORY_COLUMNS,
    CANONICAL_COLUMNS,
    CMUEmptyFileError,
    CMUInvalidCellIDError,
    CMURawSchemaError,
    CMU_V3_CELL_IDS,
    DATASET_VERSION,
    RAW_COLUMNS,
    build_asset_inventory,
    load_cmu_v3_main_file,
)


REAL_RAW_ROOT = Path("data/raw/cmu_evtol_vtc6/figshare_14226830")


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
