# API lifecycle and canonical interfaces

A large scientific package needs a navigable public surface. The 0.9 lifecycle layer inventories exports, maps them to conceptual families, and carries explicit statuses such as `core`, `workflow`, `advanced`, `experimental`, `gated`, `compatibility`, and `deprecated`. Unknown future symbols remain `unreviewed`; maturity is never inferred from a name alone.

```python
import eyeprocesspy as ep

registry = ep.eye_api_lifecycle()
inventory = ep.eye_api_inventory()  # frozen R 0.11.1 public reference surface
registry = ep.register_eye_api_status(
    registry,
    "run_eye_pipeline",
    "workflow",
    canonical="run_eye_pipeline",
)
audit = ep.audit_eye_api(inventory, registry)
summary = ep.api_surface_summary(audit.table)
recommendations = ep.eye_api_recommendation(audit)
```

## Packaged lifecycle closure

The Python package ships the frozen 0.9 lifecycle registry copied from the `eyeprocess` 0.11.1 source. It contains all 1,182 R exports and the 108-row module policy. Calling `eye_api_inventory()` with its default `package="eyeprocess"` inventories that frozen reference surface. Use `package="eyeprocesspy"` when you specifically want the currently implemented Python namespace.

Lifecycle status describes software-interface maturity and governance. It does not establish construct validity, empirical adequacy, device equivalence, or the scientific interpretation of gaze, pupil, response-time, sequence, or psychometric measures.
