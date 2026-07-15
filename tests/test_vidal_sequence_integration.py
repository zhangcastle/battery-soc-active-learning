"""Raw integration tests for Vidal sequence SOC v2."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.integration

DATASETS = {
    "panasonic_18650pf": REPO / "data/raw/Panasonic_18650PF/25degC",
    "lg_hg2": REPO / "data/raw/LG_HG2/25degC",
    "samsung_30t": REPO / "data/raw/Samsung_30T/25degC",
}


@pytest.fixture(scope="module")
def v2_index():
    root = REPO / "outputs/labelled/vidal_sequence_v2"
    if not root.exists():
        pytest.skip("v2 outputs not generated")
    rows = []
    for idx in root.glob("*/*/sequence_index.csv"):
        rows.append(pd.read_csv(idx))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def qc_reset():
    p = REPO / "outputs/qc_reports/vidal_sequence_v2/reset_unwrap.csv"
    if not p.exists():
        pytest.skip("QC not generated")
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def qc_anchors():
    p = REPO / "outputs/qc_reports/vidal_sequence_v2/soc_anchors.csv"
    if not p.exists():
        pytest.skip("QC not generated")
    return pd.read_csv(p)


class TestV2Integration:
    def test_raw_data_present(self):
        for name, path in DATASETS.items():
            if not path.exists():
                pytest.skip(f"Missing raw for {name}")

    def test_exported_all_pass(self, v2_index):
        assert set(v2_index["label_qc_status"].unique()) <= {"PASS"}

    def test_reset_charges_unwrapped(self, qc_reset):
        targets = {
            "10-29-18_22.29 551_Charge8.mat",
            "03-19-19_06.45 734_Charge8_25degC_IN21700_30T.mat",
            "03-20-19_22.00 740_Charge11_25degC_IN21700_30T.mat",
        }
        found = set(qc_reset["filename"].unique()) & targets
        assert found == targets
        assert all(qc_reset[qc_reset["filename"].isin(targets)]["status"] == "PASS")

    def test_anchors_qc_pass(self, qc_anchors):
        anchor_files = {
            "03-18-17_00.05 3406_Charge1.mat",
            "10-28-18_14.16 551_Charge2.mat",
            "10-30-18_03.00 552_Charge9.mat",
            "03-18-19_14.47 734_Charge2_25degC_IN21700_30T.mat",
            "03-20-19_19.37 740_Charge10_25degC_IN21700_30T.mat",
        }
        sub = qc_anchors[qc_anchors["filename"].isin(anchor_files)]
        assert len(sub) >= 5
        assert (sub["anchor_qc_status"] == "PASS").all()

    def test_hwfet_charges_not_exported(self, v2_index):
        excluded = {
            "03-20-17_21.35 3416_Charge2.mat",
            "10-29-18_01.47 551_Charge4.mat",
            "03-18-19_20.36 734_Charge4_25degC_IN21700_30T.mat",
        }
        exported = set(v2_index["source_filename"])
        assert not exported & excluded

    def test_v1_unchanged(self):
        v1 = REPO / "outputs/labelled/vidal_compatible_v1"
        if v1.exists():
            assert list(v1.rglob("*.parquet"))

    def test_boundary_continuity(self):
        p = REPO / "outputs/qc_reports/vidal_sequence_v2/boundary_continuity.csv"
        df = pd.read_csv(p)
        assert df["continuity_error"].max() <= 1e-6

    def test_charge_ends_near_one(self, v2_index):
        charges = v2_index[v2_index["file_type"] == "charge"]
        assert ((charges["soc_end"] - 1.0).abs() <= 1e-4).all()
