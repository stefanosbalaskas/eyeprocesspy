# Response-time and process-aware IRT

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/process-response-time-irt.Rmd`.

Eye-tracking, pupil, response-time, and sequence channels are **measurement channels**. Their association with latent response models does not automatically identify cognitive strategy, effort, diagnosis, or mental state.

```python
import eyeprocesspy as ep

spec = ep.eyeprocess_joint_process_irt_spec(
    response_family="2pl",
    time_model="lognormal",
    process_channels=("dwell", "pupil", "transitions"),
    missingness="ignorable",
)
```

`eyeprocess_process_irt_data_bundle()` preserves sparse person-item data and declared process channels. Descriptive functions summarize response-time structure, speed-accuracy association, item/person process profiles, missingness patterns, and alignment without turning process measures into latent-state labels.

```python
bundle = ep.eyeprocess_process_irt_data_bundle(
    responses,
    response_times=response_times,
    process=process_channels,
)

ep.eyeprocess_irt_response_time_profile(bundle)
ep.eyeprocess_irt_speed_accuracy_profile(bundle)
ep.eyeprocess_irt_process_item_profile(bundle)
ep.eyeprocess_irt_process_person_profile(bundle)
ep.eyeprocess_irt_process_alignment(bundle)
```

When exact joint response/response-time estimation is requested, the LNIRT adapter delegates to the external engine only when it is available and the required contract can be validated. Otherwise the result is explicitly gated rather than silently replaced by a different estimator.

That distinction matters for parity: a dependency-light Python reference model can be scientifically useful without being represented as numerically identical to an R/LNIRT likelihood.
