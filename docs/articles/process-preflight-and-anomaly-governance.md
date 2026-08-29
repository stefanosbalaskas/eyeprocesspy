# Process pre-flight and anomaly governance

This Python counterpart preserves the frozen R 0.11.1 workflow: a **data-quality gate before biometric/process modelling**, not a classifier of motivation, misconduct, diagnosis, or ability.

```python
import eyeprocesspy as ep
spec = ep.process_preflight_spec(min_gaze_validity=0.80, min_pupil_validity=0.70)
audit = ep.audit_biometric_preflight(trial_data, by=["person_id", "recording_id"], spec=spec)
ep.preflight_decisions(audit)
ep.preflight_failures(audit)
ep.preflight_exclusion_manifest(audit)
```

No observations are removed automatically. `apply_preflight_decision()` filters only when explicitly requested and records the retained decision levels. `audit_process_anomalies()` provides a regularized Mahalanobis review statistic; a large value can reflect calibration, lighting, glasses, tracker loss, atypical viewing, or other benign causes and is **not evidence of cheating, intent, diagnosis, or ability**.
