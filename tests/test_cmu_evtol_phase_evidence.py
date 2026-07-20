from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_cmu_evtol_phase_evidence import (
    EVIDENCE_COLUMNS,
    add_power_columns,
    contiguous_ns_runs,
    run_evidence_audit,
)


REAL_RAW_ROOT = Path("data/raw/cmu_evtol_vtc6/figshare_14226830")


def _canonical_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": ["VAH01"] * 6,
            "source_file": ["VAH01.csv"] * 6,
            "source_row": [1, 2, 3, 4, 5, 6],
            "time_s": [0.0, 2.0, 9.0, 10.0, 13.0, 20.0],
            "Ecell_V": [4.0] * 6,
            "I_mA": [1000.0, -2000.0, 0.0, -1000.0, -1000.0, -1000.0],
            "cycleNumber": [10, 10, 10, 1, 1, 1],
            "Ns": [4, 4, 4, 4, 4, 5],
            "continuous_block_id": [0, 0, 0, 1, 1, 1],
            "continuous_block_boundary_reason": [
                "file_start",
                "",
                "",
                "time_duplicate",
                "",
                "",
            ],
        }
    )


def test_power_sign_calculation_is_explicit() -> None:
    powered = add_power_columns(_canonical_rows())
    assert powered["electrical_power_W"].iloc[:3].tolist() == [4.0, -8.0, 0.0]
    assert powered["discharge_power_W"].iloc[:3].tolist() == [0.0, 8.0, -0.0]


def test_contiguous_ns_runs_never_cross_canonical_blocks() -> None:
    runs = contiguous_ns_runs(_canonical_rows())
    ns4 = runs.loc[runs["ns_value"].eq(4)]
    assert len(ns4) == 2
    assert ns4["continuous_block_id"].tolist() == [0, 1]
    assert ns4[["source_row_start", "source_row_end"]].values.tolist() == [
        [1, 3],
        [4, 5],
    ]


def test_run_duration_uses_last_time_minus_first_time() -> None:
    runs = contiguous_ns_runs(_canonical_rows())
    ns4 = runs.loc[runs["ns_value"].eq(4)]
    assert ns4["duration_s"].tolist() == [9.0, 3.0]


def test_evidence_schema_has_no_production_labels() -> None:
    forbidden = {"mission_id", "phase_name", "rpt_label"}
    assert forbidden.isdisjoint(EVIDENCE_COLUMNS)


@pytest.mark.integration
def test_real_phase_evidence_audit_when_raw_package_is_available() -> None:
    if not REAL_RAW_ROOT.is_dir():
        pytest.skip(f"Optional CMU raw package is not present: {REAL_RAW_ROOT}")
    evidence = run_evidence_audit(
        REAL_RAW_ROOT,
        audit_cells=("VAH01",),
        gap_cells=(),
    )
    assert not evidence.empty
    assert set(evidence["cell_id"]) == {"VAH01"}
    assert "mission_id" not in evidence.columns
    assert "phase_name" not in evidence.columns
