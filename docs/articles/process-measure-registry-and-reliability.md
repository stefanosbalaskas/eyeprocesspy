# Process-measure registry, repeatability, and reliability

The process-measure registry separates an **observed feature** from the construct claims sometimes attached to it. Each entry records its channel, unit, aggregation level, interpretation, guardrail, and lifecycle/status information. This mirrors the frozen `eyeprocess 0.11.1` article and keeps measurement description separate from psychological interpretation.

```python
import eyeprocesspy as ep

registry = ep.process_measure_registry()
card = ep.process_measure_card("pupil_peak")
guardrails = ep.process_measure_guardrails()
```

A project-specific measure can be registered only when the same metadata contract is supplied. In particular, a guardrail cannot be omitted.

```python
registry = ep.register_process_measure(
    registry,
    name="custom_signal",
    channel="gaze",
    unit="a.u.",
    level="trial",
    interpretation="Observed custom signal.",
    guardrail="Do not interpret as a psychological construct without validation.",
)
```

## Repeatability is a separate question

Repeated observations can be evaluated with absolute-agreement ICC, Bland–Altman summaries, temporal stability, split-half reliability, and participant-level bootstrap uncertainty.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(9)
people = np.arange(1, 31)
trait = rng.normal(size=len(people))
rows = []
for i, person in enumerate(people):
    for session in (1, 2):
        rows.append({
            "person_id": person,
            "session": session,
            "dwell_time": 850 + 130 * trait[i] + rng.normal(scale=30),
        })
repeated = pd.DataFrame(rows)

icc = ep.process_icc(repeated, "person_id", "session", "dwell_time")
profile = ep.process_reliability_profile(
    repeated, "person_id", "session", "dwell_time"
)
ax = ep.plot_eye_process_reliability_profile(profile, type="bland_altman")
```

`process_reliability_profile()` deliberately keeps its caveat in the result contract: reliability depends on design and population. **High repeatability does not establish construct validity, diagnostic meaning, or person-level interpretability.**

See the executable counterpart in `examples/irt_process_reliability_09.py`.
