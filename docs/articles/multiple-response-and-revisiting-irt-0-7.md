# Multiple-Response Items, Revisiting, and Local Dependence

This Python article mirrors the frozen 0.7 development article and preserves its governance boundaries.

## Why preserve the response process?

Multiple-response items can contain more information than a single total or partial-credit score. `encode_response_combinations()` preserves option combinations before any scoring rule is imposed.

```python
import pandas as pd
import eyeprocesspy as ep

long = pd.DataFrame({
    "participant_id": ["p1"] * 4 + ["p2"] * 4,
    "item_id": ["item1"] * 8,
    "option_id": ["A", "B", "C", "D"] * 2,
    "selected": [True, False, True, False, False, True, False, True],
})
print(ep.encode_response_combinations(long))
```

## Option-level process evidence

`fit_multiple_response_process_irt()` provides a transparent option-level logistic reference and an explicit `external_engine` route. The bundled reference is **not** the MRM/MRM-LD likelihood and records `exact_multiple_response=False`.

## Local dependence must be checked

`audit_process_local_dependence()` computes Q3-style pairwise residual correlations. An aligned process-residual matrix can be supplied to examine whether response and process dependence share the same item/option pairs. The threshold is descriptive, not a universal significance cutoff.

## Revisiting as collateral evidence in cognitive diagnosis

`fit_revisit_process_cdm()` keeps mastery semantics anchored to the supplied Q-matrix. Revisiting, response time, and optional gaze are collateral process evidence and are not psychological diagnoses.

## Validation requirements

Before promotion, retain response/attribute recovery, local-dependence misspecification, option sparsity, process-channel ablation, negative controls, and held-person/item/session/device validation.
