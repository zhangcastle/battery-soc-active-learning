import hashlib
from pathlib import Path

import pandas as pd
import yaml

from battery_benchmark.data.cmu_evtol_raw import CMU_V3_CELL_IDS


ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = ROOT / "configs/protocols/cmu_evtol_mission_segmentation_v0_1.yaml"
MARKDOWN_PATH = ROOT / "docs/protocols/cmu_evtol_mission_segmentation_v0_1.md"
EVIDENCE_PATH = ROOT / "data/manifests/cmu_evtol_mission_phase_evidence_v0_1.csv"
SUPPORTED = {"VAH01", "VAH17", "VAH27", "VAH02", "VAH12", "VAH05"}


def _protocol() -> dict:
    return yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))


def test_draft_is_present_versioned_and_fully_disabled() -> None:
    protocol = _protocol()
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    assert protocol["protocol_version"] == "0.1"
    assert "Protocol v0.1" in markdown
    assert protocol["status"] == "draft"
    assert protocol["production_enabled"] is False
    assert protocol["emit_mission_records"] is False
    assert protocol["emit_phase_labels"] is False
    assert protocol["emit_rpt_labels"] is False
    assert "does not claim production segmentation is complete" in markdown


def test_cycle_number_and_block_safety_rules() -> None:
    protocol = _protocol()
    cycle = protocol["cycle_number_policy"]
    assert cycle["allowed_as_mission_primary_key"] is False
    assert cycle["allowed_as_mission_boundary"] is False
    assert cycle["allowed_as_split_key"] is False
    assert cycle["considered_globally_unique"] is False
    assert "cycle" not in protocol["proposed_mission_id"]["template"].lower()
    assert "all_three_segments_in_same_continuous_block" in protocol[
        "classification_rules"
    ]["accept_all_required"]
    assert "crosses_continuous_block" in protocol["classification_rules"][
        "reject_any"
    ]


def test_accept_reject_uncertain_and_rpt_exclusion_are_explicit() -> None:
    rules = _protocol()["classification_rules"]
    assert rules["accept_all_required"]
    assert rules["reject_any"]
    assert rules["uncertain_any"]
    assert rules["rejected_and_uncertain_qc_counts_must_be_retained"] is True
    assert rules["silent_drop_allowed"] is False
    assert rules["rpt_like_action"] == (
        "exclude_from_ordinary_mission_candidates_without_emitting_rpt_label"
    )


def test_supported_and_deferred_cells_partition_all_22_cells() -> None:
    evidence = _protocol()["evidence_cells"]
    supported = set(evidence["supported"])
    deferred = set(evidence["deferred_unvalidated"])
    assert supported == SUPPORTED
    assert len(deferred) == 16
    assert supported.isdisjoint(deferred)
    assert supported | deferred == set(CMU_V3_CELL_IDS)


def test_proposed_output_schema_excludes_deferred_research_fields() -> None:
    schema = _protocol()["proposed_output_schema"]
    assert "mission_id" in schema
    forbidden = {"soc", "soh", "capacity", "split", "phase_name"}
    assert not any(
        token in field.lower() for field in schema for token in forbidden
    )


def test_clean_main_evidence_manifest_shape_and_provenance() -> None:
    evidence = pd.read_csv(EVIDENCE_PATH)
    expected_sha = _protocol()["references"]["evidence_audit_sha256"]
    assert evidence.shape == (85, 58)
    assert hashlib.sha256(EVIDENCE_PATH.read_bytes()).hexdigest() == expected_sha
    assert evidence["git_dirty"].eq(False).all()  # noqa: E712
    assert evidence["git_revision"].str.fullmatch(r"[0-9a-f]{40}").all()
    assert evidence["git_revision"].nunique() == 1
