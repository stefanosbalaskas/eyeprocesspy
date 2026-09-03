# Quality, Provenance, and Responsible Interpretation

Eye and physiological signals are observations. They do not identify mental states without a defensible measurement model and external validation.

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/responsible-use.Rmd`.

```python
import eyeprocesspy as ep

ep.validate_eye_dataset(x)
ep.audit_sampling_rate(x)
ep.audit_signal_quality(x)
ep.audit_pupil_quality(x)
ep.audit_trial_coverage(x)
ep.audit_missingness(x)
ep.analysis_readiness(x)
ep.interpretive_warnings()
```

The package does not automatically equate:

- fixation with attention;
- long dwell time with item difficulty;
- pupil dilation with cognitive load;
- rapid response with guessing;
- EDA with a uniquely identified emotional state;
- a statistical process component with effort or engagement.

These boundaries apply equally to descriptive, predictive, latent-variable, Bayesian, IRT, machine-learning, and multimodal analyses. A model label does not create construct validity.

## Reproducible records

```python
ep.provenance_manifest(x)
ep.write_provenance(x, "provenance.json", format="json")
ep.report_eye_dataset(
    x,
    "analysis-report.md",
    include_plots=True,
)
```

Exploratory and confirmatory analyses should use separately declared feature, preprocessing, AOI, exclusion, and model specifications. Quality failures, missingness, transformations, exclusions, alignment parameters, and model deviations should remain recoverable from the analysis record.
