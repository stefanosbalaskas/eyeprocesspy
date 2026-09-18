# Missingness, quality, and genuine zero gaze

Trial-level mediation is especially vulnerable to conflating **zero**, **not observed**, and **poor quality**. The preparation contract keeps these states separate.

## Four states to distinguish

- **Observed non-zero**: gaze was observed and the mediator is non-zero.
- **Observed zero**: gaze was observed and the measured mediator is exactly zero. This is valid data when zero is meaningful for the metric.
- **Not observed**: the mediator is unavailable. It remains missing.
- **Poor quality**: the quality rule failed. By default the value remains visible but the trial is flagged as ineligible.

`quality_action="mask_mediator"` is an explicit opt-in that masks mediator values before decomposition. It still preserves the trial row.

## Why this matters

Replacing unobserved gaze with zero changes the estimand: it asserts that the participant had no dwell rather than that the measurement is unknown. Dropping quality failures without an audit similarly changes the analysis population. Both operations therefore require explicit analyst decisions.

## Recommended reporting

Report the number of input trials, missing mediator trials, genuine-zero mediator trials, poor-quality trials, missing behavioral responses, the exact quality threshold, whether quality failures were flagged or masked, and the inferential missingness policy used downstream.
