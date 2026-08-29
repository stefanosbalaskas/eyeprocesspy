# Temporal process windows and AOI trajectories

`extract_process_windows()` converts sample-level gaze and pupil streams into a participant × trial × temporal-window representation while preserving the window specification as provenance.

```python
ws = ep.process_window_spec(width_ms=1000, step_ms=500, start_ms=0, end_ms=3000)
w = ep.extract_process_windows(samples, spec=ws, pupil="pupil_bc", gaze_x="x", gaze_y="y", aoi="aoi")
ep.validate_process_windows(w)
ep.summarize_process_windows(w)
```

`audit_process_window_sensitivity()` checks whether summaries depend strongly on arbitrary temporal windows. `aoi_trajectory_features()` summarizes binned AOI occupancy with polynomial trajectory coefficients. These coefficients describe temporal shape; they are not causal or latent-strategy parameters by themselves.
