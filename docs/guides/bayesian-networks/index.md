# Bayesian networks for multimodal eye-tracking analysis

`eyeprocesspy.bayesian_networks` is a downstream probabilistic-analysis layer for **already processed and scientifically qualified features**. It does not detect fixations, assign AOIs, interpolate pupil data, derive EDA features, silently discretize continuous variables, impute missing values, or exclude trials.

> **Interpretation boundary.** A learned directed edge is a probabilistic graph orientation under the supplied data, algorithm, score, and constraints. It is not, by itself, evidence of a causal effect.

## Where the layer belongs

~~~text
raw gaze / pupil / physiology
        ↓
validation + preprocessing
        ↓
event detection + AOI assignment
        ↓
feature construction + provenance
        ↓
Bayesian-network analysis
        ├── fixed theoretical DAG
        ├── constrained structure learning
        ├── parameter estimation
        ├── posterior queries
        ├── bootstrap stability
        ├── participant-grouped validation
        └── detector / AOI sensitivity
~~~

This separation lets gaze and physiological processing remain auditable while the BN consumes a canonical feature table.

## Install

The backend is optional:

~~~bash
pip install "eyeprocesspy[bayesnet]"
~~~

The first implementation uses `pgmpy`. The public `eyeprocesspy` API remains backend-neutral.

## Minimal static-BN workflow

~~~python
from eyeprocesspy.bayesian_networks import (
    define_bn_constraints,
    prepare_bayesian_network_data,
    learn_bayesian_network,
    fit_bayesian_network,
    query_bayesian_network,
)

bn_data = prepare_bayesian_network_data(
    data,
    nodes={
        "condition": {"type": "categorical", "modality": "experimental"},
        "target_dwell_fraction": {
            "type": "continuous",
            "modality": "gaze",
            "unit": "proportion",
            "range": [0, 1],
        },
        "pupil_auc": {"type": "continuous", "modality": "pupil"},
        "trust": {"type": "continuous", "modality": "questionnaire"},
        "choice": {"type": "categorical", "modality": "behavior"},
    },
    participant_id="participant_id",
    trial_id="trial_id",
    provenance={"analysis": "preregistered-main"},
)

constraints = define_bn_constraints(
    temporal_tiers=[
        ["condition"],
        ["target_dwell_fraction"],
        ["pupil_auc"],
        ["trust"],
        ["choice"],
    ],
    nodes=bn_data.structure_nodes,
    max_parents=3,
)

learned = learn_bayesian_network(
    bn_data,
    algorithm="hill_climb",
    score="bic",
    constraints=constraints,
)
~~~

For mixed categorical/continuous data, structure learning can use conditional-Gaussian scores. The first backend **does not silently discretize mixed data** and therefore does not expose mixed-BN parameter fitting or posterior inference until a defensible backend is available.

## Two legitimate entry points

A BN analysis does not have to start with structure discovery. If the graph is theory-specified, fit it directly:

~~~python
fitted = fit_bayesian_network(
    [
        ("condition", "gaze"),
        ("gaze", "pupil"),
        ("condition", "trust"),
        ("gaze", "trust"),
        ("trust", "choice"),
    ],
    data=bn_data,
    estimator="bayesian",
    prior="BDeu",
    equivalent_sample_size=10,
)
~~~

For discrete networks, posterior queries use exact variable elimination:

~~~python
posterior = query_bayesian_network(
    fitted,
    target="trust",
    evidence={"condition": "Explanation", "gaze": "High"},
)
~~~

## What to report

At minimum report the observation level, node definitions, units, missingness semantics, preprocessing provenance, structure algorithm/score, temporal/edge constraints, parameter estimator/prior, bootstrap stability, participant-grouped validation, sensitivity analyses, and the exact posterior query.

Continue with the [data contract](data-contract.md), [structure learning](structure-learning.md), [stability and validation](stability-validation.md), [sensitivity analysis](sensitivity.md), and [reporting guide](reporting.md).
