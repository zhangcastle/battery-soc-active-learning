"""Tests for SOC label QC."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_benchmark.data.adapter_qc import QCResult
from battery_benchmark.labels.label_qc import run_label_qc


def _labelled_df(
    file_type: str = "drive_cycle",
    n: int | None = None,
    ah: np.ndarray | None = None,
    soc: np.ndarray | None = None,
) -> pd.DataFrame:
    if ah is None:
        n = n or 20
        ah = np.linspace(0.0, -2.0, n)
    else:
        n = len(ah)
    if soc is None:
        denom = abs(float(np.min(ah)))
        soc = (ah + denom) / denom
    time_s = np.arange(n, dtype=float)
    current = np.full(n, -1.0)
    int_ah = np.zeros(n)
    for i in range(1, n):
        int_ah[i] = int_ah[i - 1] + current[i] * (time_s[i] - time_s[i - 1]) / 3600.0
    return pd.DataFrame({
        "time_s": time_s,
        "cycler_ah": ah,
        "soc_raw": soc,
        "soc": soc,
        "cycler_ah_delta": ah - ah[0],
        "current_integrated_ah": int_ah,
        "ah_reference_difference": (ah - ah[0]) - int_ah,
    })


class TestLabelQC:
    def test_normal_drive_pass_or_warn(self):
        df = _labelled_df()
        qc = run_label_qc(
            df, file_type="drive_cycle", adapter_qc=QCResult(),
            denominator_ah=2.0, formula_mode="drive_discharge",
        )
        assert qc.status in ("PASS", "WARN")
        assert qc.soc_start == pytest.approx(1.0)

    def test_normal_charge(self):
        ah = np.linspace(0.0, 2.5, 20)
        denom = 2.5
        soc = ah / denom
        df = _labelled_df(file_type="charge", ah=ah, soc=soc)
        qc = run_label_qc(
            df, file_type="charge", adapter_qc=QCResult(),
            denominator_ah=denom, formula_mode="charge",
        )
        assert qc.status in ("PASS", "WARN")

    def test_opposite_direction_fail(self):
        ah = np.linspace(0.0, -2.0, 10)
        soc = np.linspace(1.0, 0.0, 10)
        df = _labelled_df(ah=ah, soc=soc)
        df["current_integrated_ah"] = np.linspace(0.0, 2.0, 10)
        df["ah_reference_difference"] = df["cycler_ah_delta"] - df["current_integrated_ah"]
        qc = run_label_qc(
            df, file_type="drive_cycle", adapter_qc=QCResult(),
            denominator_ah=2.0, formula_mode="drive_discharge",
        )
        assert qc.status == "FAIL"

    def test_soc_oob_warn(self):
        soc = np.linspace(1.0, 1.5, 10)
        df = _labelled_df(n=10, soc=soc)
        qc = run_label_qc(
            df, file_type="drive_cycle", adapter_qc=QCResult(),
            denominator_ah=2.0, formula_mode="drive_discharge",
        )
        assert qc.soc_out_of_bounds_count > 0
        assert qc.status == "WARN"

    def test_ah_reset_warn(self):
        adapter_qc = QCResult()
        adapter_qc.ah_reset_detected = True
        df = _labelled_df()
        qc = run_label_qc(
            df, file_type="drive_cycle", adapter_qc=adapter_qc,
            denominator_ah=2.0, formula_mode="drive_discharge",
        )
        assert qc.status == "WARN"
        assert qc.ah_reset_detected

    def test_nonzero_ah_start_warn(self):
        ah = np.linspace(0.2, -1.8, 10)
        df = _labelled_df(ah=ah)
        qc = run_label_qc(
            df, file_type="drive_cycle", adapter_qc=QCResult(),
            denominator_ah=1.8, formula_mode="drive_discharge",
        )
        assert any("Non-zero Ah start" in w for w in qc.warnings)

    def test_reference_difference_stats(self):
        df = _labelled_df()
        qc = run_label_qc(
            df, file_type="drive_cycle", adapter_qc=QCResult(),
            denominator_ah=2.0, formula_mode="drive_discharge",
        )
        assert qc.max_abs_ah_reference_difference is not None

    def test_drive_positive_growth_fail(self):
        ah = np.linspace(0.0, 1.0, 10)
        df = _labelled_df(ah=ah)
        qc = run_label_qc(
            df, file_type="drive_cycle", adapter_qc=QCResult(),
            denominator_ah=1.0, formula_mode="drive_discharge",
        )
        assert qc.status == "FAIL"
