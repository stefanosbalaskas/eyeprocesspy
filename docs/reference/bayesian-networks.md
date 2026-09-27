# Bayesian-network API

The Bayesian-network surface is a **Python-native extension** to `eyeprocesspy`; it is not part of the frozen R parity ledger. Install the optional backend with:

```bash
pip install "eyeprocesspy[bayesnet]"
```

The public import path is `eyeprocesspy.bayesian_networks`. The implementation is split across small modules so preprocessing, structure learning, fitting, inference, stability, validation, sensitivity, plotting, and reporting remain separately auditable.

> A learned directed edge is a probabilistic graph orientation under the supplied data, algorithm, score, and constraints. It is not, by itself, evidence of a causal effect.

## Data contracts

| Public object | Canonical module | Purpose |
| --- | --- | --- |
| `BayesianNodeSpec` | `bayesian_networks.schema` | Node type, modality, unit, range, role, states, and missingness semantics |
| `BayesianDataSpec` | `bayesian_networks.schema` | Prepared feature table plus IDs and provenance |
| `BayesianConstraintSpec` | `bayesian_networks.schema` | Temporal tiers, required/forbidden edges, parent cap |
| `BayesianDataValidationResult` | `bayesian_networks.schema` | Explicit validation errors and warnings |
| `BayesianNetworkResult` | `bayesian_networks.schema` | Backend-neutral learned/fitted graph result |
| `BayesianNetworkQueryResult` | `bayesian_networks.schema` | Posterior query plus evidence and provenance |
| `BayesianNetworkStabilityResult` | `bayesian_networks.schema` | Bootstrap edge/direction stability table |
| `BayesianNetworkValidationResult` | `bayesian_networks.schema` | Participant-grouped validation evidence |
| `BayesianNetworkComparisonResult` | `bayesian_networks.schema` | Cross-specification robustness tables |

## Preparation and constraints

```python
define_bn_nodes(specifications)
prepare_bayesian_network_data(
    data,
    *,
    nodes,
    node_specs=None,
    participant_id="participant_id",
    trial_id="trial_id",
    stimulus_id="stimulus_id",
    observation_level="trial",
    provenance=None,
    provenance_columns=None,
)
validate_bayesian_network_data(data_spec, *, strict=False)

define_temporal_tiers(tiers, *, nodes=None)
define_bn_constraints(
    *,
    temporal_tiers=(),
    required_edges=(),
    forbidden_edges=(),
    max_parents=None,
    nodes=None,
)
```

Preparation preserves missing values and does not discretize, impute, drop rows, or reinterpret zeros. Quality nodes are excluded from structure learning by default unless explicitly included.

## Learning, fitting, and inference

```python
learn_bayesian_network(
    data_spec,
    *,
    algorithm="hill_climb",
    score="bic",
    constraints=None,
    max_parents=None,
    random_state=42,
    backend="pgmpy",
    allow_unconstrained=False,
)

fit_bayesian_network(
    graph,
    *,
    data=None,
    estimator="bayesian",
    prior="BDeu",
    equivalent_sample_size=10.0,
    constraints=None,
    backend="pgmpy",
)

query_bayesian_network(result, *, target, evidence=None)
predict_bayesian_network(result, *, target, evidence)
```

The first backend exposes static hill-climb, PC, and GES structure learning. Discrete networks support MLE or Bayesian parameter estimation and exact variable-elimination queries. Gaussian networks support MLE fitting and Gaussian conditioning. Mixed conditional-Gaussian structure search is available without silent discretization; mixed parameter fitting/inference is intentionally not exposed in this first tranche.

## Stability and validation

```python
bootstrap_bn_structure(
    data_spec,
    *,
    algorithm="hill_climb",
    score="bic",
    constraints=None,
    n_boot=1000,
    resample_by=None,
    max_parents=None,
    random_state=42,
    backend="pgmpy",
)

validate_bayesian_network(
    data_spec,
    *,
    algorithm="hill_climb",
    score="bic",
    constraints=None,
    method="group_kfold",
    groups=None,
    n_splits=10,
    estimator="bayesian",
    prior="BDeu",
    equivalent_sample_size=10.0,
    max_parents=None,
    random_state=42,
    backend="pgmpy",
)

validate_bayesian_network_recovery(learned_edges, true_edges)
simulate_multimodal_bayesian_network_example(
    *,
    n_participants=400,
    trials_per_participant=1,
    random_state=42,
)
```

When repeated trials are present, bootstrap resampling defaults to participant level. Grouped validation relearns the structure and refits parameters inside every training fold before scoring held-out participants.

## Detector and AOI sensitivity

```python
compare_bayesian_networks(results, *, target=None, evidence=None)

compare_bn_across_detectors(
    analyses,
    *,
    algorithm="hill_climb",
    score="bic",
    constraints=None,
    max_parents=None,
    random_state=42,
    backend="pgmpy",
)

compare_bn_across_aoi_specs(analyses, **kwargs)
```

These helpers compare already-created feature tables. They do not rerun event detection or redefine AOIs inside the Bayesian-network layer.

## Plotting

```python
plot_bayesian_network(result, *, edge_strength=None, ax=None, seed=42)
plot_bn_edge_stability(stability, *, ax=None)
plot_bn_direction_stability(stability, *, ax=None)
plot_bn_posterior(query, *, ax=None)
plot_bn_detector_robustness(comparison, *, ax=None)
plot_bn_validation(validation, *, ax=None)
```

Plots return standard Matplotlib axes and attach their numerical payload to `ax.eyeprocess_plot_data` for auditability.

## Reporting

```python
report_bayesian_network(
    result,
    *,
    stability=None,
    validation=None,
    sensitivity=None,
    path=None,
)
```

The report records data level, model family, structure algorithm/score, parameter estimator, edges, bootstrap stability, grouped validation, sensitivity evidence, warnings, provenance, and the non-causal interpretation boundary.

See the [Bayesian-network guide](../guides/bayesian-networks/index.md), [multimodal worked example](../examples/bayesian-network-multimodal.md), and [sensitivity example](../examples/bayesian-network-sensitivity.md).
