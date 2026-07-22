# CMU to Vidal adapter v1

This development-only bridge maps the frozen CMU dynamic-capacity SOC v1
artifacts to the record and window contract used by the Vidal 2019
reproduction. It does not make the CMU reference ground truth or formal
benchmark eligible.

Each `(cell_id, mission_cycle_id)` is validated and adapted independently.
Input order must already have strictly increasing `source_row`, `time_s`, and
`elapsed_candidate_s`; the adapter does not sort malformed input. Features are
`voltage_v`, `current_a`, and `Temperature__C`, in that order. The label is an
unchanged copy of `soc_dynamic`; current sign is unchanged.

The real inputs are not uniformly 1 Hz, so each candidate is interpolated on
its own 1 Hz elapsed-time grid without extrapolation or SOC clipping. Synthetic
grid points retain the bracketing original source rows and interpolated
original `time_s`. Candidates are never connected.

Windows reuse the Vidal Case 1 rule: length 50, stride 50, fixed
non-overlapping windows, and incomplete tails dropped by default. Window
metadata records the CMU cell/candidate identity and source-row/time bounds.

The adapter returns raw physical features. It has no scaler fitting path. A
future zero-shot experiment may only apply a previously fitted Panasonic
normalizer through its `transform()` method; inference and checkpoint loading
are outside this adapter.

