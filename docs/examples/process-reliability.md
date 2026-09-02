# Process-measure reliability

Eye-tracking and pupil features are often treated as stable person-level quantities without showing whether repeated observations support that interpretation. `eyeprocesspy` includes explicit reliability and agreement workflows for process measures.

![Process reliability](../assets/gallery/process-reliability.svg)

The executable example is [`examples/process_reliability.py`](https://github.com/stefanosbalaskas/eyeprocesspy/blob/release/0.1.0-deep-parity/examples/process_reliability.py).

## 1. Structure repeated observations

```python
# columns: person, session, dwell_score
repeated.head()
```

The unit of analysis should match the intended claim. A person-level reliability statement requires repeated person-level observations under a defensible session/task design.

## 2. Estimate an ICC and Bland–Altman agreement

```python
profile = ep.process_reliability_profile(
    repeated,
    person="person",
    session="session",
    measure="dwell_score",
)

print(profile["icc"])
print(profile["bland_altman"]["summary"])
```

The profile combines an absolute-agreement ICC with pairwise Bland–Altman summaries when at least two sessions are available.

## 3. Examine temporal stability

```python
stability = ep.process_temporal_stability(
    repeated,
    person="person",
    session="session",
    measure="dwell_score",
)
```

This separates rank-order association from absolute agreement. Depending on the research question, both can matter.

## 4. Plot agreement

```python
ax = ep.plot_eye_process_reliability_profile(profile)
```

The plot exposes the pair mean, difference, bias and limits of agreement. The numerical data are preserved in the result object and on the plotting axis.

!!! note "Reliability is not validity"
    A reliable gaze, pupil or process measure can still measure the wrong construct. Report the population, task, session spacing, preprocessing, aggregation rule and uncertainty; do not treat a high reliability coefficient as construct validation.
