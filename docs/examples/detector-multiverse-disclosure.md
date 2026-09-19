# Worked example: disclosure dwell under detector uncertainty

This example is synthetic, reproducible, CI-sized, and deliberately includes one optional-backend failure case.

## Research question

A simulated disclosure condition produces longer gaze dwell on a disclosure AOI. The analysis asks whether the positive condition effect survives defensible variation in event detection.

## 1. Simulate 60 Hz gaze data

```python
import eyeprocesspy as ep

data = ep.simulate_detector_multiverse_data(
    n_participants=8,
    sampling_rate=60,
    seed=20260918,
)
```

The generator creates explicit trial intervals, a disclosure AOI, a main-content AOI, valid/missing samples, and a known positive disclosure-dwell effect.

## 2. Declare the detector set

The runnable script compares I-VT at 25, 30 and 35 deg/s, an I-DT specification, the robust-MAD adaptive reference detector, and a REMoDNaV branch.

```python
from examples.detector_multiverse_worked import build_specs
multiverse = build_specs()
```

No branch receives an undeclared threshold. If REMoDNaV is not installed, that branch is recorded as failed and the remaining branches continue; it is never replaced silently.

## 3. Run and propagate

```python
result = ep.run_detector_multiverse(data, multiverse)
result = ep.propagate_detector_to_aoi(result, overlap="error")
result = ep.propagate_detector_to_features(result)
```

### Event agreement

![Pairwise fixation-event agreement](../assets/detector-multiverse/detector-agreement.svg)

Interpret this figure as pairwise temporal agreement, not ground-truth accuracy.

### Disclosure dwell

![Disclosure dwell across detector branches](../assets/detector-multiverse/disclosure-dwell-by-detector.svg)

The known simulated disclosure effect remains visible, while the absolute dwell estimates vary because event boundaries and fragmentation differ by detector.

## 4. Propagate to inference

```python
model_spec = {
    "engine": "statsmodels_ols",
    "formula": "dwell_time_ms ~ C(condition_id) + C(participant_id)",
    "outcome": "dwell_time_ms",
    "aoi_id": "disclosure",
}

inference = ep.run_detector_inference_multiverse(
    result,
    model_spec,
    minimum_valid_fraction=0.5,
)
```

Participant fixed effects keep the example lightweight for CI. In an empirical repeated-measures study, a mixed model may be preferable; use `statsmodels_mixedlm` or an explicit callback to the specialist modelling stack chosen for the study.

### Model-input audit

Before interpreting coefficients, inspect the row-accounting table:

```python
inference.input_audit[
    [
        "detector_id",
        "input_rows",
        "aoi_selected_rows",
        "quality_excluded_rows",
        "outcome_missing_rows",
        "model_rows_used",
        "status",
    ]
]
```

This table distinguishes AOI selection, prespecified quality filtering, outcome missingness, and model failure. Those states should not be collapsed into a single post-hoc N.

### Coefficient stability

![Condition coefficient across detector branches](../assets/detector-multiverse/condition-coefficient-stability.svg)

The synthetic run used to generate this page produced same-direction condition coefficients across all successfully modelled internal detector branches. The exact values are not scientific results and will change if the simulation parameters change.

The stability summary uses the **planned-specification denominator**. In this seeded example, six detector branches were declared, five returned the disclosure coefficient, and the unavailable REMoDNaV branch remains visible as a model-stage failure. Thus `specifications = 6`, `term_available_specifications = 5`, `model_failure_specifications = 1`, and the convergence rate is `5 / 6`, not `5 / 5`.

### Planned-denominator audit

![Planned detector denominator](../assets/detector-multiverse/robustness-denominator.svg)

This figure is a reporting aid, not a new statistic: it visualizes the same planned-specification accounting returned by the stability summary. See the [visual diagnostic atlas](../guides/detector-multiverse-visual-atlas.md) for a publication-oriented reading sequence.

## 5. Inspect what changed

```python
summary = ep.summarise_detector_robustness(
    result,
    inference=inference,
    term="C(condition_id)[T.disclosure]",
    substantive_threshold=100,
)

summary["event_summary"]
summary["feature_sensitivity"]
summary["inference_stability"]
summary["failures"]
```

The output separates event sensitivity, measurement sensitivity, model stability and branch failures. A failure is evidence about the evaluated workflow; it is not silently removed from the audit trail.

## 6. Generate a reproducible report

```python
ep.report_detector_multiverse(
    result,
    inference=inference,
    term="C(condition_id)[T.disclosure]",
    substantive_threshold=100,
    path="detector-multiverse-report.md",
)
```

The complete runnable source is `examples/detector_multiverse_worked.py`.
