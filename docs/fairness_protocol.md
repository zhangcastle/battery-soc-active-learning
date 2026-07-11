# Fairness Protocol

This document defines the two comparison modes in the benchmark and the rules
that ensure comparisons are scientifically valid.

---

## Mode A: Paper-faithful reproduction

In paper-faithful mode, each method is reproduced as closely as possible to
its original publication:

- Each paper's original architecture is used (not unified backbone)
- Original preprocessing is used where documented
- Original sequence construction is used where confirmed
- Original normalisation scheme is used
- Results are compared against the paper's reported numbers

**When to use:** Replicating published results to verify correctness.

**Limitation:** Paper-faithful results are not directly comparable across methods
because each method may use a different preprocessing pipeline, data split,
or feature set.

---

## Mode B: Unified fair-comparison

In unified benchmark mode, all methods share identical conditions:

| Condition | Value |
|-----------|-------|
| Dataset split | `by_cycle` (or as specified per dataset) |
| Input features | voltage, current, temperature |
| Sequence construction | fixed non-overlapping windows |
| Backbone | as specified in the unified experiment config |
| Label budget | same fraction of target labels for all methods |
| Target-domain access | same for all methods (labelled / unlabelled as declared) |
| Evaluation metrics | MAE, RMSE, MAX error, BIAS, R² |
| Random seed | 42 (or as specified) |
| Model-selection policy | best validation MAE checkpoint |

**When to use:** Comparing methods to each other on a level playing field.

---

## Anti-leakage rules

The following rules must be enforced regardless of mode:

1. **No cycle leakage:** A cycle's rows must never appear in both train and test.
2. **No mission leakage:** A mission must not be split across train and test.
3. **No scaler leakage:** Normalisers must be fitted on training data only.
4. **No target test label use:** Target-domain test labels must never be used
   during training, adaptation, or hyperparameter selection.
5. **No retrospective oracle:** Pseudo-labels must not be generated using
   the test-set ground truth.

---

## Metric reporting requirements

- Always report MAE in absolute SOC units (not percentage), e.g. `0.021` not `2.1%`.
- Always report on both source test and target test separately.
- Physical-consistency metrics must be clearly labelled and not presented as
  alternatives to supervised accuracy metrics.
- Domain metrics (MMD, classifier accuracy) are diagnostic only.
