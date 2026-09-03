# Complete Gazepoint Downstream Workflow

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/gazepoint-downstream-workflow.Rmd`.

## Purpose

`run_gazepoint_workflow()` executes the research-data workflow from a real Gazepoint Analysis folder:

1. canonical eye-dataset import;
2. file-pair, timebase, coordinate, sampling-rate, and signal-quality audits;
3. contiguous media-run reconstruction as person-by-item-by-trial intervals;
4. vendor-fixation and AOI summaries;
5. short-gap pupil interpolation, optional filtering, and blink detection;
6. valid-only biometric summaries while preserving native values;
7. gaze, pupil, biometric, AOI, and QC plots;
8. one-row-per-person-item-trial process tables;
9. response templates and IRT-ready long/matrix structures;
10. canonical exports, provenance, source fingerprints, and reproducible reports.

The workflow does not manufacture response scores. When observed responses are absent, the result remains process-ready but response-pending.

## Minimal workflow

```python
import eyeprocesspy as ep

source_dir = "path/to/eyeprocess-validation-corpus/cases/gazepoint-analysis-v7.2.0-demo"
output_dir = "path/to/eyeprocess-downstream-output"

result = ep.run_gazepoint_workflow(
    source_dir,
    output_dir=output_dir,
    overwrite=True,
)

ep.validate_gazepoint_workflow(result)
```

## Explicit specification

Pupil baseline correction is deliberately disabled by default. The first samples after media onset are not automatically equivalent to a pre-stimulus baseline.

```python
spec = ep.gazepoint_workflow_spec(
    expected_sampling_rate=60,
    minimum_valid_gaze=0.80,
    minimum_valid_pupil=0.70,
    pupil_interpolation="linear",
    pupil_max_gap_ms=150,
    pupil_filter="median",
    pupil_window=5,
    pupil_baseline="none",
    create_plots=True,
    create_html_report=True,
    retain_raw=True,
)
```

## Item labels and conditions

By default, `item_id` follows Gazepoint `MEDIA_ID`. A study-specific mapping can supply meaningful item and condition labels.

```python
import pandas as pd

item_map = pd.DataFrame(
    {
        "stimulus_id": ["0", "1"],
        "item_id": ["item_control", "item_treatment"],
        "condition_id": ["control", "treatment"],
    }
)

result = ep.run_gazepoint_workflow(
    source_dir,
    output_dir,
    item_map=item_map,
    spec=spec,
    overwrite=True,
)
```

## Adding observed responses

Responses can be supplied during the workflow or joined later through the generated response template.

```python
responses = pd.DataFrame(
    {
        "participant_id": ["User 3", "User 3"],
        "item_id": ["item_control", "item_treatment"],
        "response": ["yes", "no"],
        "score": [1, 1],
        "response_time": [6.1, 7.4],
    }
)

result = ep.run_gazepoint_workflow(
    source_dir,
    output_dir,
    responses=responses,
    item_map=item_map,
    spec=spec,
    overwrite=True,
)
```

Response and response-time matrices are created only when the relevant observations exist. The workflow does not fit IRT automatically; model adequacy, sample size, item count, dimensionality, and process-covariate assumptions still require evaluation.

## Output structure

A completed workflow contains canonical data, QC evidence, analysis tables, IRT-ready files where supported, plot outputs, workflow reports, specification/provenance records, source fingerprints, and rerun information. File extensions follow the Python storage backend rather than reproducing R-specific `.rds` artifacts.

## Interpretation boundaries

Fixations are not automatically attention; dwell time is not automatically difficulty; pupil dilation is not automatically cognitive load; and GSR or heart rate does not identify a specific emotion. The report preserves these interpretive safeguards alongside the analysis outputs.
