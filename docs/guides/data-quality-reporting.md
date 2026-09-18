# Reporting eye-tracking data quality

A manuscript-ready quality section should describe the measurement quantities that matter for the analysis rather than dumping every possible tracker statistic.

## Minimum data-quality items

Report, when relevant:

- the native/analysis coordinate system and unit;
- how validation targets were presented and segmented;
- target-referenced accuracy and its summary level;
- the precision operationalization (RMS-S2S, SD, BCEA) and BCEA probability when used;
- nominal sampling frequency and whether sampling was irregular;
- empirically observed/effective frequency or interval statistics when needed;
- when long-gap diagnostics are reported, distinguish the number of long intervals from the estimated number of nominal samples lost within those intervals;
- the definition and amount of data loss;
- whether loss causes such as blinks or tracker invalidity were distinguishable;
- preprocessing that can change these metrics;
- pre-specified review/exclusion criteria and the number of affected units;
- sensitivity analyses when conclusions may depend on quality decisions.

## Suggested wording

> Gaze-position accuracy was quantified as target-referenced Euclidean error. Precision was characterized separately using RMS sample-to-sample displacement and spatial SD; 68% BCEA was additionally reported as an area-based dispersion measure. Sampling behavior was summarized from timestamps using effective frequency, median inter-sample interval, and interval jitter alongside the device's nominal sampling rate. Data loss was defined as unavailable gaze coordinates or explicit invalidity and was summarized with loss fraction and missing-run duration. Pre-specified thresholds triggered review rather than automatic deletion, and exclusions were reported separately.

## What not to write

Avoid claims such as “data quality was good” without an operational definition, using nominal Hz as proof that the realized stream was regular, reporting a single unspecified “precision” number, or describing a vendor calibration score as if it were a universal measure of measurement uncertainty.

## References

The package terminology follows the 2023 minimal reporting guideline for eye-tracking studies (Dunn et al.; published in *Behavior Research Methods* 2024) and Niehorster et al.'s 2026 *Behavior Research Methods* tutorial on determining eye-tracking data quality (doi:10.3758/s13428-026-03039-4).
