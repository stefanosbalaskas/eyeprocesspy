# Item-Bank Decisions, Fairness Drift, and Reference Centiles

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/item-bank-fairness-norms.Rmd`.

Item selection can be treated as a constrained multi-objective decision rather than a sequence of isolated cutoffs.

```python
import eyeprocesspy as ep

objectives = ep.item_objective_spec(
    information="test_information",
    process_burden="pupil_effort",
    fairness="absolute_dif",
    exposure="exposure_rate",
    content_constraints={"content_domain": (1, 5)},
)
pareto = ep.item_pareto_front(item_bank, objectives)
ep.plot_item_pareto(pareto)

selected = ep.optimize_item_bank(
    pareto,
    n_items=20,
    objectives=objectives,
    method="evolutionary",
)
ep.plot_selected_bank_profile(selected)
stability = ep.audit_bank_decision_stability(selected)
ep.plot_decision_stability(stability)
```

Process-DIF is reported separately from psychometric DIF and can be monitored over deployment batches.

```python
dif = ep.fit_process_dif(
    person_item_data,
    response="accuracy",
    process="dwell_ms",
    group="group",
    item="item_id",
    ability="theta",
)
ep.plot_process_dif_forest(dif)

drift = ep.monitor_dif_drift(
    monitoring_data,
    time="deployment_batch",
    group="group",
    metrics=("difficulty", "dwell_ms", "pupil_auc"),
    item="item_id",
)
ep.plot_dif_drift_heatmap(drift)
```

Conditional centiles are reference distributions, not clinical classifications.

```python
norms = ep.fit_process_norms(
    reference_sample,
    "dwell_ms",
    ("age", "item_difficulty"),
)
ep.predict_process_centiles(norms, new_people)
ep.score_process_deviation(norms, new_people, type="centile")
ep.plot_process_centiles(norms)
ep.audit_norm_transportability(norms, external_sample)
```

Fairness, exposure, burden, and content objectives should remain explicit decision criteria. Optimization cannot determine which trade-offs are substantively acceptable.
