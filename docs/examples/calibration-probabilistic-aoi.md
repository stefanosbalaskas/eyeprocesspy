# Calibration uncertainty and probabilistic AOIs

Hard AOI assignment treats a gaze point as if its coordinate were exact. This workflow instead estimates an empirical calibration-error model, propagates that uncertainty, and summarizes AOI membership probability under the fitted measurement-error model.

![Calibration error](../assets/gallery/calibration-error.svg)

The executable example is [`examples/calibration_probabilistic_aoi.py`](https://github.com/stefanosbalaskas/eyeprocesspy/blob/release/0.1.0-deep-parity/examples/calibration_probabilistic_aoi.py).

## 1. Estimate the calibration-error distribution

```python
model = ep.calibration_error_model(calibration_validation_data)
print(model["metrics"])
```

The input contains target coordinates and observed gaze coordinates. The model stores the empirical mean error, covariance, raw error cloud and descriptive error metrics.

## 2. Summarize the uncertainty ellipse

```python
ellipse = ep.gaze_uncertainty_ellipse(model, level=0.95)
print(ellipse)
```

The ellipse is a geometric summary of the fitted two-dimensional calibration error; it is not a participant diagnosis or universal accuracy threshold.

## 3. Propagate uncertainty into AOI membership

```python
assignment = ep.probabilistic_aoi_assignment(
    gaze_points,
    aois,
    model,
    draws=400,
    seed=9,
    min_probability=0.50,
)

print(assignment["assignments"])
```

![Probabilistic AOI membership](../assets/gallery/probabilistic-aoi.svg)

Each point is repeatedly perturbed under the empirical calibration-error model, producing a membership distribution across AOIs. Ambiguous boundary points can therefore remain uncertain instead of being forced into a single deterministic rectangle.

## 4. Compare hard and probabilistic assignment

```python
comparison = ep.compare_hard_probabilistic_aoi(
    gaze_points,
    aois,
    assignment,
)
```

Use disagreement as a measurement-sensitivity diagnostic. It can reveal AOIs or trials whose substantive conclusions depend strongly on calibration error or boundary placement.

!!! warning "Do not over-interpret probability"
    These probabilities quantify propagated coordinate uncertainty under the fitted calibration model. They are **not posterior probabilities that a participant psychologically attended to an AOI**.
