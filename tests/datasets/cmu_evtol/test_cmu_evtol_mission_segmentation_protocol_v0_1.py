import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from battery_benchmark.datasets.cmu_evtol.raw import CMU_V3_CELL_IDS


ROOT = Path(__file__).resolve().parents[3]
YAML_PATH = ROOT / "configs/datasets/cmu_evtol/protocols/cmu_evtol_mission_segmentation_v0_1.yaml"
MARKDOWN_PATH = ROOT / "docs/datasets/cmu_evtol/protocols/cmu_evtol_mission_segmentation_v0_1.md"
EVIDENCE_PATH = ROOT / "data/manifests/cmu_evtol/cmu_evtol_mission_phase_evidence_v0_1.csv"
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
    assert protocol["implementation_authorized"] is False
    assert "does not claim production segmentation is complete" in markdown


def test_cycle_number_and_block_safety_rules() -> None:
    protocol = _protocol()
    cycle = protocol["cycle_number_policy"]
    assert cycle["allowed_as_mission_primary_key"] is False
    assert cycle["allowed_as_mission_boundary"] is False
    assert cycle["allowed_as_split_key"] is False
    assert cycle["considered_globally_unique"] is False
    assert "cycle" not in protocol["proposed_mission_id"]["template"].lower()
    classification = protocol["classification"]
    assert "all_three_segments_in_same_continuous_block" in classification[
        "accept_requirements"
    ]
    assert "crosses_continuous_block" in classification["reason_codes"][
        "hard_reject"
    ]


def test_primary_classification_is_deterministic_and_reason_codes_do_not_overlap() -> None:
    classification = _protocol()["classification"]
    expected = ["out_of_scope", "hard_reject", "uncertain", "accept"]
    assert classification["primary_statuses"] == expected
    assert classification["precedence"] == expected
    assert len(set(classification["primary_statuses"])) == 4
    assert classification["exactly_one_primary_status_per_candidate"] is True
    assert classification["retain_all_triggered_reason_codes"] is True
    assert classification["duplicate_reason_codes_allowed"] is False

    reasons = classification["reason_codes"]
    assert "missing_required_segment" in reasons["uncertain"]
    assert "missing_required_segment" not in reasons["hard_reject"]
    assert set(reasons["hard_reject"]).isdisjoint(reasons["uncertain"])
    assert "deferred_cell_without_phase_evidence_validation" in reasons[
        "out_of_scope"
    ]
    assert classification["silent_drop_allowed"] is False
    assert classification["rpt_like_action"] == (
        "exclude_from_ordinary_mission_candidates_without_emitting_rpt_label"
    )


def test_candidate_enumeration_and_overlap_policy_are_explicit() -> None:
    enumeration = _protocol()["candidate_enumeration"]
    assert enumeration["scope"] == "one_continuous_block_at_a_time"
    assert enumeration["run_unit"] == (
        "contiguous_same_Ns_rows_within_one_continuous_block"
    )
    assert enumeration["anchor"] == "run_with_high_discharge_power_evidence"
    assert enumeration["lookahead"] == (
        "next_two_adjacent_runs_must_form_lower_then_high"
    )
    assert enumeration["crossing_continuous_block_allowed"] is False
    assert enumeration["missing_followup_action"] == "emit_uncertain_qc_candidate"
    assert enumeration["ns_4_alone_can_accept"] is False
    assert enumeration["overlap_policy"] == "disallow_shared_runs"
    assert enumeration["overlap_resolution"] == "earliest_anchor_source_row_wins"


def test_supported_and_deferred_cells_partition_all_22_cells() -> None:
    evidence = _protocol()["evidence_cells"]
    supported = set(evidence["supported"])
    deferred = set(evidence["deferred_unvalidated"])
    assert supported == SUPPORTED
    assert len(deferred) == 16
    assert supported.isdisjoint(deferred)
    assert supported | deferred == set(CMU_V3_CELL_IDS)
    assert "deferred_cell_without_phase_evidence_validation" in _protocol()[
        "classification"
    ]["reason_codes"]["out_of_scope"]


def test_proposed_output_schema_excludes_deferred_research_fields() -> None:
    schema = _protocol()["proposed_output_schema"]
    assert "mission_id" in schema
    assert {"primary_status", "primary_reason", "all_reason_codes", "qc_flags"} <= set(
        schema
    )
    forbidden = {"soc", "soh", "capacity", "split", "phase_name"}
    assert not any(
        token in field.lower() for field in schema for token in forbidden
    )


def test_diagnostic_screening_thresholds_match_evidence_audit() -> None:
    evidence = pd.read_csv(EVIDENCE_PATH)
    assert evidence["candidate_thresholds_json"].nunique() == 1
    audited = json.loads(evidence["candidate_thresholds_json"].iloc[0])
    configured = _protocol()["diagnostic_screening_thresholds"]
    for key, value in audited.items():
        assert configured[key] == value
    assert configured["status"] == "v0_1_diagnostic_candidate_screening_only"
    assert configured["condition_specific_final_acceptance_tolerance"] == "not_frozen"
    assert configured["maximum_next_implementation_scope"] == (
        "diagnostic_classifier_only"
    )


def test_clean_main_evidence_manifest_shape_and_provenance() -> None:
    evidence = pd.read_csv(EVIDENCE_PATH)
    expected_sha = _protocol()["references"]["evidence_audit_sha256"]
    assert evidence.shape == (85, 58)
    assert hashlib.sha256(EVIDENCE_PATH.read_bytes()).hexdigest() == expected_sha
    assert evidence["git_dirty"].eq(False).all()  # noqa: E712
    assert evidence["git_revision"].str.fullmatch(r"[0-9a-f]{40}").all()
    assert evidence["git_revision"].nunique() == 1
