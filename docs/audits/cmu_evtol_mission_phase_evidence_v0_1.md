# CMU eVTOL Mission/Phase Evidence Audit v0.1

## Scope and provenance

This is a descriptive evidence audit, not a segmentation protocol. It examines
CMU KiltHub article `14226830`, eVTOL Battery Dataset Version 3, using raw loader
schema `cmu_evtol_kilthub_v3_raw_v0.1` and the existing 120 s exploratory block
threshold.

Primary evidence cells:

- baseline: VAH01, VAH17, VAH27;
- condition evidence: VAH02 (extended cruise), VAH12 (short cruise), VAH05
  (reduced discharge power).

Gap-only context was also inspected for VAH09, VAH10, and VAH30; VAH27 appears
in both sets. Each input SHA-256, the Git revision/dirty state, generation time,
loader version, and audit thresholds are recorded in the evidence CSV.

Power is calculated without changing the dataset sign convention:

```text
electrical_power_W = Ecell_V * I_mA / 1000
discharge_power_W = max(-electrical_power_W, 0)
```

Positive electrical power is charge; negative electrical power is discharge.

## Descriptive candidate filters

The audit searched for contiguous-run triples with high/lower/high discharge
power, using 35–70 W / 5–30 W / 35–70 W and duration ranges 45–120 s /
200–1500 s / 60–150 s. Long low-rate discharge evidence used 0.5–4 W,
duration at least 3600 s, voltage drop at least 0.8 V, and end voltage no higher
than 2.7 V.

These ranges are **audit filters only**. They are not mission boundaries, phase
definitions, RPT labels, or frozen protocol constants.

## Ns evidence

All six evidence cells contain Ns values 0–9. Full per-cell statistics—including
row count, contiguous-run count, charge/discharge/zero-current fractions, real
run-duration min/median/max, power median/P10/P90, and common adjacent Ns—are in
`data/manifests/cmu_evtol_mission_phase_evidence_v0_1.csv`.

The recurring observations are:

- Ns 0 and 1 are charge-dominated in all six cells, at approximately 12 W and a
  lower taper-like power respectively.
- Ns 2 is predominantly zero current.
- Ns 3 mixes zero current and low-power discharge; it is not a single clean
  physical phase.
- Ns 4, 5, and 6 commonly occur consecutively. In ordinary repeated waveforms,
  Ns 4 and 6 are high-power discharge and Ns 5 is lower-power discharge.
- Ns 7 mixes zero current and low-power discharge, including long low-rate
  discharge candidates.
- Ns 8 and 9 are predominantly zero current.

Ns is therefore useful transition evidence, but it is not sufficient as a
standalone phase label. The first low-rate diagnostic sequence also traverses
Ns 4/5/6 with powers and durations unlike the ordinary repeated waveform.
VAH05 additionally contains many zero-duration/interrupted runs near duplicate
timestamp boundaries.

## Candidate high/lower/high waveform

The power-and-duration audit found the following complete candidate triples.
These are counts of descriptive patterns, not mission counts or IDs.

| Cell | Candidate triples | First segment | Middle segment | Last segment |
|---|---:|---:|---:|---:|
| VAH01 | 828 | 54.03 W, 74.95 s | 16.04 W, 800.00 s | 54.02 W, 105.00 s |
| VAH17 | 980 | 54.04 W, 74.95 s | 16.04 W, 800.00 s | 54.03 W, 105.00 s |
| VAH27 | 572 | 54.04 W, 74.96 s | 16.04 W, 800.00 s | 54.03 W, 105.00 s |
| VAH02 | 610 | 54.04 W, 74.95 s | 16.04 W, 1000.00 s | 54.03 W, 105.00 s |
| VAH12 | 2302 | 53.98 W, 74.95 s | 15.98 W, 400.00 s | 53.98 W, 105.00 s |
| VAH05 | 1514 | 48.58 W, 74.95 s | 14.38 W, 800.00 s | 48.58 W, 105.00 s |

The baseline cells are highly consistent in typical power and duration. The
condition cells exhibit exactly the expected descriptive axis: VAH02 lengthens
the middle segment, VAH12 shortens it, and VAH05 reduces all three discharge
powers by roughly ten percent. Representative source-row ranges are retained in
the CSV.

Incomplete, interrupted, low-power diagnostic, or out-of-range triples are not
counted and remain uncertain. No mission identifier was generated.

## Long low-rate discharge evidence

The audit found 13–31 long low-power candidates per evidence cell. Their median
properties are approximately:

- 1.96–2.02 W electrical discharge magnitude;
- 4983–5601 s duration;
- 3.55–3.58 V at the candidate-run start;
- approximately 2.50 V at the end;
- 1.05–1.08 V voltage drop within the selected run.

This differs materially from the 54/16/54 W repeated waveform. However, the
selected run generally starts below 4.2 V, so it is only a low-rate sub-run and
not proof of a complete capacity discharge from charge endpoint to cutoff.

Only one unique candidate `cycleNumber` per audited cell intersected the
impedance file's `cycle numbers`; candidate cells contained 5–10 unique cycle
numbers. This weak overlap is possible correspondence evidence, not a reliable
join rule. No RPT label or capacity lookup was produced.

## Long-gap context

| Cell | Source rows | Delta | Cycle before/after | Ns before/after | Current before/after | Voltage before/after | Interpretation |
|---|---|---:|---|---|---|---|---|
| VAH09 | 55058→55059 | 158.282 s | 62→62 | 1→1 | 1395.3→0 mA | 4.2006→4.1202 V | Interrupted charge/rest context; preserve block boundary |
| VAH10 | 415299→415300 | 25238.039 s | 117→117 | 0→0 | 3006.0→0 mA | 3.8596→3.7438 V | Very long missing interval; high reconstruction risk |
| VAH27 | 23801→23802 | 166.506 s | 18→18 | 3→3 | 0→0 mA | 4.1861→4.1849 V | Rest-like gap inside same Ns; do not bridge automatically |
| VAH30 | 24921→24922 | 186.206 s | 23→23 | 0→0 | 2987.9→0 mA | 4.1684→4.0757 V | Interrupted charge/rest context; preserve block boundary |

The gaps do not coincide with `cycleNumber` changes or necessarily with Ns
changes. Future reconstruction must never bridge them merely because Ns and
cycleNumber are unchanged. VAH27 is explicitly retained as a baseline evidence
cell with this caveat.

Duplicate-time boundary counts for the nine processed cells are also recorded,
with at most three representative source rows per cell. The audit does not
manually reinterpret all duplicates.

## Decision

**PARTIAL** — The evidence is sufficient to design and test a formal
segmentation proposal that combines canonical block boundaries, Ns transitions,
signed power, duration, and completeness checks. It is not sufficient to publish
a final universal Ns-to-phase mapping or complete RPT rule.

Remaining uncertainties include interrupted/zero-duration runs, diagnostic
sequences that reuse Ns 4/5/6, the weak impedance-cycle correspondence, exact
mission envelope and recovery rules, and how to reject partial sequences near
file/block boundaries. Those decisions require a separately reviewed formal
segmentation protocol.

This audit produced no `mission_id`, final phase name, SOC, RPT capacity table,
sequence windows, split, scaler, or training data.
