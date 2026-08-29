# Advanced pupillometry representations and confound control

The frozen 0.8 layer provides transparent pupil frequency/activity representations, event-kernel deconvolution, luminance/trial-order adjustment, and auditable median filtering.

```python
freq = ep.pupil_frequency_features(samples, by=["person_id", "trial_id"], time="time_ms", pupil="pupil_bc", sampling_rate_hz=60)
deconv = ep.fit_pupil_event_deconvolution(samples, events={"stimulus": 0})
```

`pupil_activity_index()` exposes velocity, frequency-contrast, and RIPA-style proxy representations. They are signal representations—not pure cognitive-load measures. `fit_pupil_confound_model(..., engine="lm")` provides the transparent reference adjustment. Exact R `mgcv`, `plm`, and `robfilter` paths remain explicit backend boundaries rather than silent substitutions. Adjusted pupil values remain model-dependent and should be described as luminance/fatigue-adjusted, not as cognition isolated from all confounding.
