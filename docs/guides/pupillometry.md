# Pupillometry workflow

Pupil analysis is especially sensitive to time alignment, missingness, blink handling, baseline choices, filtering, and aggregation. `eyeprocesspy` keeps pupil observations in the canonical `eye_samples` table and supports both scalar and trajectory-level/process representations.

## Inspect the canonical pupil stream first

```python
import eyeprocesspy as ep

ax = ep.plot_pupil_timeseries(eye, trial_id="T1")
ax.figure.tight_layout()
```

![Pupil time series](../assets/gallery/pupil-timeseries.svg)

Before preprocessing, confirm that eye labels, timestamps, trial IDs, validity information, and pupil units are what the study expects.

## Treat preprocessing as part of the measurement model

A manuscript-quality pipeline should make the following decisions explicit:

- left/right/binocular handling;
- validity and blink rules;
- whether missing segments are interpolated;
- maximum interpolated gap;
- filtering/smoothing choices;
- event alignment and latency window;
- baseline interval;
- baseline correction form;
- trial exclusion rules;
- whether inference uses samples, windows, scalar summaries, or functional representations.

Interpolation is a transformation of missing observations, not recovery of ground truth.

## Baseline choices

The functional-pupil workflow supports explicit baseline strategies such as subtraction, percentage change, and z-scoring. Baseline choice changes the estimand and should therefore be reported alongside the time window and event definition.

## Functional pupil representations

For time-resolved analysis, `eyeprocesspy` provides the functional-pupil family for:

- aligned pupil trajectories;
- basis representations;
- nuisance adjustment;
- trial-level coefficient extraction;
- functional/process connections to IRT workflows;
- explicit backend checks for advanced models.

Optional statistical backends are not silently substituted. If an exact backend is unavailable, the package raises a backend error rather than changing the estimator under the same function label.

## Missingness is scientific information

Pupil missingness can be informative about tracking conditions, blink behavior, head position, glasses, task phases, or other measurement conditions. Use the pupil-missingness and quality functions to quantify patterns and sensitivity rather than reporting only the interpolated result.

At minimum, consider reporting:

- fraction of available pupil observations;
- distribution and duration of missing segments;
- exclusions due to missingness;
- interpolation rule;
- sensitivity to alternative missingness thresholds;
- whether missingness differs systematically across experimental conditions.

## Align pupil with events carefully

Pupil responses are temporally smooth and delayed relative to many experimental events. Event alignment should distinguish:

- stimulus/event timestamp uncertainty;
- device clock alignment;
- pre-event baseline period;
- analysis window;
- overlapping events;
- trial boundaries.

Clock/timebase audit functions and explicit event tables should be used before interpreting fine-grained timing differences.

## Combine pupil with other process channels

The package's multimodal/process families allow pupil features or trajectories to coexist with gaze, AOI, response, and other synchronized process information. This does not mean every channel should be fused into a single latent construct. Preserve channel-specific units and validity evidence.

## Reporting checklist

For a reproducible paper, report:

1. device/export context and nominal sampling rate;
2. empirical sampling/timing checks;
3. pupil units and eye handling;
4. validity/blink definition;
5. missingness/interpolation rule;
6. filtering/smoothing;
7. baseline definition and correction;
8. event alignment and analysis window;
9. feature/trajectory representation;
10. aggregation level;
11. exclusions;
12. robustness/sensitivity checks.

## Interpretation boundary

A pupil change is not automatically a measure of cognitive load, effort, arousal, surprise, or emotion. Construct interpretation requires an experimental design and external validity evidence that distinguish competing explanations.

[See the visual gallery](../gallery.md){ .md-button }
[Explore process quality and uncertainty](process-quality-uncertainty.md){ .md-button .md-button--primary }
