# Multimodal Process IRT: Responses, Time, Gaze, and Missingness

This is the Python counterpart of the frozen `eyeprocess` 0.11.1 article. The 0.7 architecture treats process observations as declared **measurement channels**, not as an undifferentiated feature dump. Every channel should have a family, role, latent target, provenance, and validation programme.

## Measurement channels, not feature dumping

```python
import eyeprocesspy as ep

spec = ep.irt_model_spec(
    id="accuracy_time_gaze",
    latent=["ability", "speed", "engagement"],
    channels={
        "response": ep.irt_response_channel("2pl"),
        "rt": ep.irt_rt_channel("lognormal"),
        "gaze": ep.irt_count_channel("negative_binomial"),
    },
    status="experimental",
)
```

The registry also supports nominal choices, survival/event time, compositional AOI measurements, sequences, functional trajectories, and bounded continuous process measures.

## Registry

`list_irt_models()` exposes status and channel composition. Experimental/gated specifications cannot be fitted through `fit_irt_model()` unless `allow_experimental=True` is supplied deliberately. Promotion to reference status is an evidence record, not a convenience flag.

## Response + response time + gaze

`fit_joint_gaze_rt_irt()` supplies an auditable Python reference decomposition. The frozen R reference uses `lme4` crossed random effects; the current Python reference uses explicit fixed-effect design matrices and therefore records `algorithmic_parity=False`. It must not be described as the same estimator as the R mixed model or the cited fully joint Bayesian model.

## Graded responses

`fit_joint_graded_rt_process_irt()` retains the same response/RT/process decomposition and the same experimental status. Python currently exposes a dependency-light reference decomposition rather than pretending to reproduce the R `MASS`/`lme4` estimator.

## Nominal distractors + option gaze

`fit_nominal_gaze_irt()` retains selected-option identity and option-level gaze proportions. `option_process_information()`, `distractor_process_map()`, and `audit_distractor_attention()` summarize the extra process evidence. The interpretation remains process-based: greater gaze to an option does not establish *why* it was inspected.

## Missingness as a process

`classify_item_missingness()` separates answered, not reached, reached-not-inspected, inspected omission, started-unanswered, and other reached omissions. `fit_omission_survival_irt()` keeps omission and not-reached mechanisms separate. The Python reference currently uses transparent cause-specific exponential hazards; the R source uses clustered Cox models, so exact algorithmic parity remains pending.

## Device and algorithm facets

`fit_manyfacet_process_irt()`, `facet_effects()`, `audit_process_measurement_invariance()`, and `generalizability_process_study()` provide auditable person/item/device/session/algorithm variance diagnostics. A facet effect is evidence about transportability, not proof of vendor bias or a causal device effect.

## Bounded process measures

`fit_censored_normal_process_irt()` is a direct conditional censored-normal calibration for bounded measurements. It is explicitly **not** the full marginal EM estimator described in the cited work. Recovery validation is required before confirmatory use.
