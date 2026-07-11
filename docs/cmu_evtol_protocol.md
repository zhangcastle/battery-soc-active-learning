# CMU eVTOL Transfer Protocol

This document specifies how the CMU eVTOL battery dataset should be used
in transfer learning experiments.

---

## 1. Overview

The CMU eVTOL dataset contains battery discharge and charge cycles from
electric vertical take-off and landing (eVTOL) aircraft test batteries.
The key challenge is that **measured ground-truth SOC is not directly available**.

---

## 2. Label availability

| Label type | Availability |
|------------|-------------|
| Measured SOC reference | Not available |
| Cycler-reported SOC | Unknown / unconfirmed |
| Coulomb-counted proxy reference | Generated offline (see §3) |
| Target adaptation labels | **Unavailable to the model during adaptation** |

---

## 3. Proxy SOC reference generation

A proxy reference for evaluation can be generated using:
1. Quality-controlled coulomb counting
2. Dynamic capacity estimation per cycle (from full discharge/charge pairs)
3. Label quality scoring to flag unreliable cycles

This proxy reference is stored as:
```yaml
label_source: coulomb_counted_dynamic_capacity
label_quality: <float 0-1>
```

**This proxy is NOT treated as measured ground truth.** Results must clearly
state the label source.

---

## 4. Evaluation scenarios

### Scenario A: Source-labelled → target-unlabelled adaptation

The model is trained on source data (e.g. Panasonic) with full SOC labels.
At adaptation time, only unlabelled eVTOL data is available.
Evaluation uses the hidden proxy reference (not given to the model).

Supported methods: `source_only`, `mmd`, `coral`, `dann`, `source_free`

### Scenario B: Supervised few-shot target fine-tuning

A small number of labelled target cycles (from the proxy reference) are
provided for fine-tuning.  The remaining target test set uses the proxy reference.

Supported methods: `finetune`, `scratch`

### Scenario C: Source-free adaptation

The source model weights are available but the source data is not.
Adaptation uses only unlabelled target data.

Supported methods: `source_free`

---

## 5. Splits

Splits must be at the **mission level** to prevent temporal leakage:
- A mission is one complete flight + charging sequence.
- All cycles within a mission must be in the same split.

```yaml
split:
  strategy: by_mission
```

---

## 6. Limitations

- Coulomb counting error accumulates across long missions.
- Dynamic capacity estimation may be unreliable for aged cells.
- Phase labels (takeoff, cruise, hover, landing) are approximate.
- Cell ageing during the dataset recording period is not accounted for.

---

## 7. Reporting requirements

When reporting CMU eVTOL results:
1. State whether target labels were available during adaptation.
2. Report the label source of the evaluation reference.
3. Report physical-consistency metrics alongside proxy accuracy metrics.
4. Note any cells removed by quality control.
