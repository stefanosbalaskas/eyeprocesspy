from __future__ import annotations

import pytest

from eyeprocesspy.bayesian_networks import (
    define_bn_constraints,
    fit_bayesian_network,
    learn_bayesian_network,
    prepare_bayesian_network_data,
    query_bayesian_network,
    simulate_multimodal_bayesian_network_example,
)

pytest.importorskip("pgmpy")


def _prepared():
    data = simulate_multimodal_bayesian_network_example(
        n_participants=300,
        random_state=2026,
    )
    return prepare_bayesian_network_data(
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
        stimulus_id=None,
    )


def test_real_pgmpy_fixed_fit_query_and_constrained_learning():
    data = _prepared()
    constraints = define_bn_constraints(
        temporal_tiers=[
            ["condition"],
            ["gaze"],
            ["pupil", "eda"],
            ["trust"],
            ["choice"],
        ],
        nodes=data.structure_nodes,
        max_parents=3,
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
        data=data,
        constraints=constraints,
        estimator="bayesian",
        prior="BDeu",
        equivalent_sample_size=10,
    )
    posterior = query_bayesian_network(
        fitted,
        target="trust",
        evidence={
            "condition": "Explanation",
            "pupil": "Dilated",
            "eda": "High",
        },
    )
    assert posterior.posterior is not None
    assert sum(posterior.posterior.values()) == pytest.approx(1.0)

    learned = learn_bayesian_network(
        data,
        algorithm="hill_climb",
        score="bic",
        constraints=constraints,
    )
    assert set(learned.nodes) == set(data.structure_nodes)
    assert learned.score == "bic-d"
