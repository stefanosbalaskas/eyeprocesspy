"""Reproducible static Bayesian-network demonstration for multimodal features."""

from eyeprocesspy.bayesian_networks import (
    define_bn_constraints,
    fit_bayesian_network,
    prepare_bayesian_network_data,
    query_bayesian_network,
    simulate_multimodal_bayesian_network_example,
)

data = simulate_multimodal_bayesian_network_example(
    n_participants=1000,
    random_state=2026,
)

bn_data = prepare_bayesian_network_data(
    data,
    nodes={
        "condition": {"type": "categorical", "modality": "experimental"},
        "gaze": {"type": "categorical", "modality": "gaze"},
        "pupil": {"type": "categorical", "modality": "pupil"},
        "eda": {"type": "categorical", "modality": "physiology"},
        "trust": {"type": "categorical", "modality": "questionnaire"},
        "choice": {"type": "categorical", "modality": "behavior"},
    },
    participant_id="participant_id",
    trial_id="trial_id",
)

constraints = define_bn_constraints(
    temporal_tiers=[
        ["condition"],
        ["gaze"],
        ["pupil", "eda"],
        ["trust"],
        ["choice"],
    ],
    nodes=bn_data.structure_nodes,
)

fitted = fit_bayesian_network(
    [
        ("condition", "gaze"),
        ("gaze", "pupil"),
        ("gaze", "eda"),
        ("condition", "trust"),
        ("gaze", "trust"),
        ("trust", "choice"),
    ],
    data=bn_data,
    constraints=constraints,
    estimator="bayesian",
    prior="BDeu",
    equivalent_sample_size=10,
)

query = query_bayesian_network(
    fitted,
    target="trust",
    evidence={
        "condition": "Explanation",
        "pupil": "Dilated",
        "eda": "High",
    },
)

print(query.posterior)
