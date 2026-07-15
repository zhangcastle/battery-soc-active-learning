#!/usr/bin/env python3
"""Audit Vidal-compatible cycler Ah behaviour across Panasonic / LG / Samsung."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from battery_benchmark.data.lg_hg2 import LGHG2Adapter
from battery_benchmark.data.panasonic import PanasonicAdapter
from battery_benchmark.data.samsung_30t import Samsung30TAdapter
from battery_benchmark.labels import SOCCalculator, label_qc_for_record

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "panasonic_18650pf": (
        PanasonicAdapter,
        REPO_ROOT / "data/manifests/panasonic_vidal2019_25c_manifest.csv",
    ),
    "lg_hg2": (
        LGHG2Adapter,
        REPO_ROOT / "data/manifests/lg_hg2_vidal2019_25c_manifest.csv",
    ),
    "samsung_30t": (
        Samsung30TAdapter,
        REPO_ROOT / "data/manifests/samsung_30t_vidal2019_25c_manifest.csv",
    ),
}

DEFAULT_SPLITS = ("full_train", "partial_train", "test")


def _audit_record(record, split: str, dataset: str) -> dict:
    calc = SOCCalculator()
    labelled = calc.calculate_vidal(record)
    lqc = label_qc_for_record(labelled)
    meta = labelled.metadata
    return {
        "dataset": dataset,
        "split": split,
        "profile": meta.get("profile", ""),
        "paper_profile": meta.get("paper_profile", ""),
        "sequence_id": meta.get("sequence_id", ""),
        "file_type": meta.get("file_type", ""),
        "n_rows": labelled.n_rows,
        "ah_start": lqc.ah_start,
        "ah_end": lqc.ah_end,
        "ah_min": lqc.ah_min,
        "ah_max": lqc.ah_max,
        "denominator_ah": lqc.denominator_ah,
        "cycler_ah_delta_end": lqc.cycler_ah_delta_end,
        "current_integrated_ah_end": lqc.current_integrated_ah_end,
        "max_abs_ah_reference_difference": lqc.max_abs_ah_reference_difference,
        "soc_start": lqc.soc_start,
        "soc_end": lqc.soc_end,
        "soc_min": lqc.soc_min,
        "soc_max": lqc.soc_max,
        "soc_out_of_bounds_count": lqc.soc_out_of_bounds_count,
        "adapter_qc_status": record.qc.status,
        "label_qc_status": lqc.status,
    }


def run_audit(
    repo_root: Path,
    output: Path,
    splits: tuple[str, ...],
    overwrite: bool = False,
) -> pd.DataFrame:
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Pass --overwrite to replace."
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen: dict[str, dict] = {}
    for dataset_name, (adapter_cls, manifest_path) in DATASETS.items():
        adapter = adapter_cls(raw_root=repo_root, manifest_path=manifest_path)
        for split in splits:
            prio = {"full_train": 0, "partial_train": 1, "test": 2}.get(split, 99)
            for record in adapter.load_split(split):
                if record.metadata.get("record_role") != "model_input":
                    continue
                seq_id = record.metadata["sequence_id"]
                row = _audit_record(record, split, dataset_name)
                current = seen.get(seq_id)
                if current is None or prio < current["prio"]:
                    seen[seq_id] = {"prio": prio, "row": row}

    rows = [entry["row"] for entry in seen.values()]

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/audits/vidal_ah_behavior.csv",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    df = run_audit(args.repo_root, args.output, tuple(args.splits), args.overwrite)
    print(f"Wrote {len(df)} audit rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
