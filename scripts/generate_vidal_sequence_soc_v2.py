#!/usr/bin/env python3
"""Generate Vidal sequence-aware SOC v2 labels."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from battery_benchmark.data.lg_hg2 import LGHG2Adapter
from battery_benchmark.data.manifest_loader import load_manifest, entry_manifest_splits
from battery_benchmark.data.panasonic import PanasonicAdapter
from battery_benchmark.data.samsung_30t import Samsung30TAdapter
from battery_benchmark.labels.vidal_capacity_policy import VidalCapacityPolicy
from battery_benchmark.labels.vidal_sequence_exporter import export_split_artifacts, write_qc_reports
from battery_benchmark.labels.vidal_sequence_soc_v2 import (
    audit_current_signs,
    build_anchor_metadata,
    load_candidate_sequences,
    reconstruct_split_sequence,
    validate_sequence_item,
)

DATASETS = {
    "panasonic_18650pf": (
        PanasonicAdapter,
        REPO / "data/manifests/panasonic_vidal2019_25c_manifest.csv",
        REPO / "data/raw/Panasonic_18650PF/25degC",
    ),
    "lg_hg2": (
        LGHG2Adapter,
        REPO / "data/manifests/lg_hg2_vidal2019_25c_manifest.csv",
        REPO / "data/raw/LG_HG2/25degC",
    ),
    "samsung_30t": (
        Samsung30TAdapter,
        REPO / "data/manifests/samsung_30t_vidal2019_25c_manifest.csv",
        REPO / "data/raw/Samsung_30T/25degC",
    ),
}

PROTOCOL_ACTIONS = {
    ("panasonic_18650pf", "03-18-17_00.05 3406_Charge1.mat"): ("CHANGE_TO_ANCHOR", "soc_anchor"),
    ("panasonic_18650pf", "03-20-17_21.35 3416_Charge2.mat"): ("CHANGE_TO_EXCLUDED", "excluded"),
    ("lg_hg2", "10-28-18_14.16 551_Charge2.mat"): ("CHANGE_TO_ANCHOR", "soc_anchor"),
    ("lg_hg2", "10-29-18_01.47 551_Charge4.mat"): ("CHANGE_TO_EXCLUDED", "excluded"),
    ("lg_hg2", "10-30-18_03.00 552_Charge9.mat"): ("CHANGE_TO_ANCHOR", "soc_anchor"),
    ("samsung_30t", "03-18-19_14.47 734_Charge2_25degC_IN21700_30T.mat"): ("CHANGE_TO_ANCHOR", "soc_anchor"),
    ("samsung_30t", "03-18-19_20.36 734_Charge4_25degC_IN21700_30T.mat"): ("CHANGE_TO_EXCLUDED", "excluded"),
    ("samsung_30t", "03-20-19_19.37 740_Charge10_25degC_IN21700_30T.mat"): ("CHANGE_TO_ANCHOR", "soc_anchor"),
}


class RecordLoader:
    def __init__(self, adapter) -> None:
        self.adapter = adapter
        self._cache: dict[str, Any] = {}

    def load_by_filename(self, filename: str):
        if filename in self._cache:
            return self._cache[filename]
        for entry in self.adapter.entries:
            if entry.filename == filename:
                splits = entry_manifest_splits(entry)
                split = splits[0] if splits else "full_train"
                rec = self.adapter.load_entry(entry, requested_split=split)
                rec.metadata["source_file"] = entry.filename
                self._cache[filename] = rec
                return rec
        return None


def build_manifest_index(manifest_path: Path) -> dict[str, dict]:
    entries = load_manifest(manifest_path)
    out = {}
    for e in entries:
        out[e.filename] = {
            "record_role": e.record_role,
            "file_type": e.file_type,
            "full_train": e.full_train,
            "partial_train": e.partial_train,
            "test": e.test,
            "exclude": e.exclude,
        }
    return out


def write_manifest_change_log(qc_root: Path) -> list[dict]:
    """Document expected manifest protocol actions (post-update verification)."""
    audit_csv = REPO / "outputs/audits/vidal_model_input_charge_review.csv"
    df = pd.read_csv(audit_csv)
    rows = []
    for _, r in df.iterrows():
        action = r["recommended_protocol_action"]
        if action not in ("CHANGE_TO_ANCHOR", "CHANGE_TO_EXCLUDED", "KEEP_MODEL_INPUT"):
            continue
        ds = r["dataset"]
        fn = r["filename"]
        new_role = {
            "KEEP_MODEL_INPUT": "model_input",
            "CHANGE_TO_ANCHOR": "soc_anchor",
            "CHANGE_TO_EXCLUDED": "excluded",
        }[action]
        old_role = r.get("current_record_role", "model_input")
        old_splits = r.get("current_split_memberships", "")
        if new_role == "excluded":
            new_splits = ""
        else:
            new_splits = old_splits
        rows.append({
            "dataset": ds,
            "filename": fn,
            "old_record_role": old_role,
            "new_record_role": new_role,
            "old_splits": old_splits,
            "new_splits": new_splits,
            "audit_action": action,
            "reason": r.get("reason", ""),
        })
    qc_root.mkdir(parents=True, exist_ok=True)
    return rows


def panasonic_missing_corresponding_charge(split: str) -> bool:
    return split == "partial_train"


def compute_boundary_charge_ends(
    dataset: str,
    split_data: dict,
    loader: RecordLoader,
    convention,
    capacity_policy: VidalCapacityPolicy,
    audit_df: pd.DataFrame,
) -> dict[str, float]:
    """Pre-compute SOC ends for boundary charges referenced outside sequence."""
    from battery_benchmark.labels.vidal_sequence_soc_v2 import compute_charge_soc, compute_drive_soc

    ends: dict[str, float] = {}
    boundary_charges = split_data.get("true_test_train_boundary_charges", [])
    ordered = split_data.get("ordered_model_sequence", [])
    first_drive = next((x for x in ordered if x["file_type"] == "drive_cycle"), None)
    if not first_drive or not boundary_charges:
        return ends

    # Use the boundary charge that immediately precedes the first drive in test split.
    charge_fn = boundary_charges[-1]
    charge_row = audit_df[
        (audit_df["dataset"] == dataset) & (audit_df["filename"] == charge_fn)
    ]
    if charge_row.empty:
        return ends
    prev_drive_fn = str(charge_row.iloc[0].get("previous_actual_drive", "") or "")
    if not prev_drive_fn:
        return ends

    drive_rec = loader.load_by_filename(prev_drive_fn)
    charge_rec = loader.load_by_filename(charge_fn)
    if drive_rec is None or charge_rec is None:
        return ends

    soc_init = 1.0
    if first_drive["filename"] in {
        a.get("next_selected_drive") for a in split_data.get("leading_anchors", [])
    }:
        return ends

    _, dmeta = compute_drive_soc(
        drive_rec, soc_init, capacity_policy.capacity_reference_ah, convention
    )
    _, cmeta, _ = compute_charge_soc(
        charge_rec,
        dmeta.soc_end,
        capacity_policy.capacity_reference_ah,
        convention,
        allow_endpoint_correction=True,
    )
    ends[first_drive["filename"]] = cmeta.soc_end
    return ends


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Vidal sequence SOC v2")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--splits", nargs="+", default=["full_train", "partial_train", "test"])
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs/labelled/vidal_sequence_v2")
    parser.add_argument("--qc-root", type=Path, default=REPO / "outputs/qc_reports/vidal_sequence_v2")
    parser.add_argument("--audit-json", type=Path, default=REPO / "outputs/audits/vidal_candidate_sequences.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-warnings", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists() and not args.overwrite and not args.dry_run:
        print(f"Output exists: {args.output_root} (use --overwrite)")
        return 1

    sequences = load_candidate_sequences(args.audit_json)
    manifest_log = write_manifest_change_log(args.qc_root)
    audit_df = pd.read_csv(REPO / "outputs/audits/vidal_model_input_charge_review.csv")

    summary_rows: list[dict] = []
    failures: list[dict] = []
    warnings: list[dict] = []
    boundary_all: list[dict] = []
    reset_all: list[dict] = []
    anchor_all: list[dict] = []
    current_sign_all: list[dict] = []

    exit_code = 0

    for dataset in args.datasets:
        if dataset not in DATASETS:
            print(f"Unknown dataset: {dataset}")
            return 1
        adapter_cls, manifest_path, raw_root = DATASETS[dataset]
        if not raw_root.exists():
            print(f"Raw data missing for {dataset}")
            return 1

        load_manifest(manifest_path)  # validate
        adapter = adapter_cls(raw_root=REPO, manifest_path=manifest_path)
        loader = RecordLoader(adapter)
        manifest_idx = build_manifest_index(manifest_path)

        model_records = []
        for split in args.splits:
            for rec in adapter.load_split(split):
                if rec.metadata["record_role"] == "model_input":
                    model_records.append(rec)

        sign_rows, convention = audit_current_signs(dataset, model_records)
        for row in sign_rows:
            current_sign_all.append(asdict(row))

        capacity_policy = VidalCapacityPolicy.resolve(
            dataset,
            adapter_capacity_hint=adapter._capacity_hint_ah,
        )

        for split in args.splits:
            if dataset not in sequences or split not in sequences[dataset]:
                continue
            split_data = sequences[dataset][split]
            ordered = split_data.get("ordered_model_sequence", [])
            leading = split_data.get("leading_anchors", [])

            for item in ordered:
                issues = validate_sequence_item(item, manifest_idx, split)
                for iss in issues:
                    failures.append({"dataset": dataset, "split": split, "filename": item["filename"], "issue": iss})

            missing_cc = dataset == "panasonic_18650pf" and panasonic_missing_corresponding_charge(split)
            boundary_ends = compute_boundary_charge_ends(
                dataset, split_data, loader, convention, capacity_policy, audit_df
            )

            results, anchors, unwrap_rows, boundary_rows = reconstruct_split_sequence(
                dataset,
                split,
                ordered,
                leading,
                loader,
                convention,
                capacity_policy,
                boundary_charge_ends=boundary_ends,
                missing_corresponding_charge=missing_cc,
            )

            for a in anchors:
                anchor_all.append(asdict(a))
            reset_all.extend(unwrap_rows)
            boundary_all.extend(boundary_rows)

            for res in results:
                row = {
                    "dataset": dataset,
                    "protocol_split": split,
                    "source_filename": res.meta.source_filename,
                    "sequence_order": res.meta.sequence_order,
                    "file_type": res.meta.file_type,
                    "label_qc_status": res.meta.label_qc_status,
                    "soc_init_source": res.meta.soc_init_source,
                    "soc_end": res.meta.soc_end,
                }
                summary_rows.append(row)
                if res.meta.label_qc_status == "FAIL":
                    failures.append({**row, "reasons": ";".join(res.meta.label_qc_reasons)})
                elif res.meta.label_qc_status == "WARNING":
                    warnings.append({**row, "reasons": ";".join(res.meta.label_qc_reasons)})

            if args.dry_run:
                print(f"[dry-run] {dataset}/{split}: {len(results)} records, {len(anchors)} anchors")
                continue

            export_info = export_split_artifacts(
                results,
                dataset=dataset,
                protocol_split=split,
                output_dir=args.output_root,
                include_warnings=args.include_warnings,
                source_manifest=manifest_path,
                capacity_policy=capacity_policy,
                current_convention=convention,
                missing_corresponding_charge=missing_cc,
                warnings=["missing_corresponding_charge_for_panasonic_cycle1"] if missing_cc else [],
                repo_root=REPO,
            )
            print(f"Exported {dataset}/{split}: {export_info['record_count']} records")

    if not args.dry_run:
        write_qc_reports(
            args.qc_root,
            summary_rows=summary_rows,
            failures=failures,
            warnings=warnings,
            boundary_rows=boundary_all,
            reset_rows=reset_all,
            anchor_rows=anchor_all,
            current_sign_rows=current_sign_all,
            manifest_log_rows=manifest_log,
        )

    if failures:
        exit_code = 1
    if args.fail_on_warning and warnings:
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
