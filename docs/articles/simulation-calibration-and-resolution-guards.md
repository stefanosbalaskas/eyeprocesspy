# Simulation-based calibration and measurement-resolution guards

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/simulation-calibration-and-resolution-guards.Rmd`.

Bayesian and process estimators can return plausible-looking results even when computation is miscalibrated. `sbc_rank_diagnostics()` therefore operates on ranks produced by a declared simulation/inference loop rather than pretending to perform simulation-based calibration by itself.

```python
import eyeprocesspy as ep

sbc = ep.sbc_rank_diagnostics(
    ranks,
    n_draws=n_draws,
    bins=10,
)
deviation = ep.sbc_ecdf_deviation(sbc)
```

SBC evaluates computational calibration under the declared generative model. It does not establish that the generative model is scientifically correct for real participants, tasks, or constructs.

## Measurement resolution

An analysis may request temporal or spatial distinctions finer than the empirical recording quality can credibly support. `analysis_resolution_guard()` combines event duration and effective sampling frequency with optional spatial feature size and radial error.

```python
guard = ep.analysis_resolution_guard(
    event_duration_ms=100,
    effective_hz=60,
    spatial_feature_size=0.20,
    radial_error=0.04,
    min_samples=3,
    max_error_fraction=0.5,
)
```

The thresholds are researcher-declared compatibility rules, not universal eye-tracking quality cutoffs.

## Pupil preprocessing order and baseline sensitivity

```python
order_audit = ep.audit_pupil_preprocessing_order(preprocessing_spec)
sensitivity = ep.pupil_baseline_sensitivity(
    pupil,
    time="time_ms",
    pupil="pupil",
    by=("person_id", "trial_id"),
    windows={"W500": (-500, 0), "W300": (-300, 0), "W200": (-200, 0)},
)
```

A successful SBC diagnostic supports computational calibration of a declared workflow under simulation. A passing resolution guard indicates compatibility with declared numerical rules. Neither result alone validates a psychological construct or a universal measurement threshold.

## Methodological anchors

- Talts S, Betancourt M, Simpson D, Vehtari A, Gelman A (2018), *Validating Bayesian Inference Algorithms with Simulation-Based Calibration*.
- Niehorster DC, Nyström M, Hessels RS, et al. (2026), *The fundamentals of eye tracking, Part 7: Determining data quality*.
- Mathôt S, Fabius J, Van Heusden E, Van der Stigchel S (2018), *Safe and sensible preprocessing and baseline correction of pupil-size data*.
