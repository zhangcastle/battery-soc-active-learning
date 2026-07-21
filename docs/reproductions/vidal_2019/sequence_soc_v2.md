# Vidal Sequence-Aware SOC v2

> **SOC labels are benchmark reference labels derived from cycler current/Ah measurements and protocol constraints, not independent ground truth.**

## v1 vs v2

| | v1 (`vidal_compatible_v1`) | v2 (`vidal_sequence_v2`) |
|---|---|---|
| Role | Open reference per-record audit label | Formal sequence-aware benchmark label |
| Drive SOC | Per-file Ah normalization | `SOC_init + ∫I·dt / C_ref` with protocol init |
| Charge SOC | `Ah / max(Ah)` per file | Coulomb count + endpoint progress correction |
| Continuity | None | Drive→charge boundary enforced within explicit segments |
| Anchors | Treated as model_input | `soc_anchor` metadata only |
| HWFET charges | In manifest | `excluded`, not in sequences |

## Generation

```bash
python scripts/reproductions/vidal_2019/generate_sequence_soc_v2.py \
  --datasets panasonic_18650pf lg_hg2 samsung_30t \
  --splits full_train partial_train test \
  --output-root outputs/reproductions/vidal_2019/sequence_soc_v2/artifacts \
  --qc-root outputs/reproductions/vidal_2019/sequence_soc_v2/qc
```

## SOC formulas

**Current convention:** positive current = charging (inferred per dataset from drive/charge median sign).

**Trapezoidal integration:**

```
dt_i = time_s[i] - time_s[i-1]
delta_ah_i = 0.5 × (I_i + I_{i-1}) × dt_i / 3600
delta_soc_i = delta_ah_i / capacity_reference_ah   (sign from convention)
```

**Drive:**

```
soc_sequence_v2(t) = soc_init + cumulative_signed_delta_soc(t)
```

`soc_init` priority: full-charge anchor / protocol full-start → previous charge end → protocol boundary charge → FAIL.

**Charge (two columns):**

```
soc_cc_raw(t) = soc_start + cumulative_signed_delta_soc(t)
progress(t) = (cycler_ah_effective(t) - cycler_ah_effective(0)) /
              (cycler_ah_effective(end) - cycler_ah_effective(0))
soc_sequence_v2(t) = soc_start + (1 - soc_start) × progress(t)
```

Endpoint correction is applied only when full voltage + dataset-specific taper QC pass.

## Capacity policy

`VidalCapacityPolicy` (`vidal_capacity_policy.py`):

1. manifest `capacity_hint_ah`
2. adapter metadata
3. dataset config: Panasonic 2.9 Ah, LG/Samsung 3.0 Ah

## Protocol actions (from charge audit)

- **29 KEEP** → `model_input` charge, sequence-aware labels
- **5 ANCHOR** → `soc_anchor`, metadata only (`soc_anchors.csv`)
- **3 EXCLUDED** → HWFET-owned, not exported

## Reset unwrap

Applied to LG `551_Charge8`, Samsung `734_Charge8`, `740_Charge11` and auto-detected resets.
Preserves `cycler_ah_raw`, produces monotonic `cycler_ah_effective`.

## Panasonic Cycle1/Cycle2 segment boundary

No post-Cycle1 corresponding charge under ownership rules. `Charge1` retained as SOC anchor only.
Cycle2 starts a new segment with `soc_init=1.0` and
`soc_init_source=protocol_full_start_missing_charge`; this is marked WARNING
with medium-confidence provenance instead of inheriting Cycle1's terminal SOC.

## Coulomb counting inspiration

Non-uniform trapezoidal integration follows patterns from external sautee reference (current×dt, clipping as safety guard only). No sautee data structures or per-file reset logic are used.

## QC

Reports live under `outputs/reproductions/vidal_2019/sequence_soc_v2/qc/`.

Default export includes **PASS + WARNING** and always excludes **FAIL** records, because removing WARNING records independently can break drive→charge pairing. Use `--pass-only` only for dependency-aware PASS-only exports. Final exported artifacts are checked with `exported_boundary_continuity.csv`, `exported_sequence_completeness.csv`, and `orphan_records.csv`.
