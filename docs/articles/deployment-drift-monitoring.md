# Post-deployment psychometric-biometric drift monitoring

Drift monitoring connects psychometric parameters with response-process and device/stimulus context. A flag is a governance signal for review, not proof that an item leaked or was compromised.

```python
import eyeprocesspy as ep
spec = ep.process_drift_spec(baseline="first_batch", difficulty_limit=.40, gaze_validity_drop=.10)
drift = ep.audit_process_drift(deployment_monitor, item="item_id", batch="deployment_batch", spec=spec)
ep.process_drift_alerts(drift)
```

`drift_by_device()`, `drift_by_site()`, `drift_by_vendor()`, and `drift_by_stimulus_version()` provide stratified audits. Plot counterparts expose trajectories, deltas, standardized heat maps, and control views while retaining the underlying plot data.
