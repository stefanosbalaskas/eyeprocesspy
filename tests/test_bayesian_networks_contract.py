from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from eyeprocesspy.bayesian_networks import (
    BayesianNodeSpec,
    define_bn_constraints,
    define_bn_nodes,
    define_temporal_tiers,
    prepare_bayesian_network_data,
    simulate_multimodal_bayesian_network_example,
    validate_bayesian_network_data,
    validate_bayesian_network_recovery,
)


def base_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2"],
            "trial_id": ["t1", "t2", "t1", "t2"],
            "condition": ["A", "B", "A", "B"],
            "gaze": [0.2, 0.6, 0.4, 0.8],
            "choice": ["no", "yes", "no", "yes"],
            "valid": [0.9, 0.8, 0.95, 0.7],
            "detector_spec_id": ["ivt"] * 4,
        }
    )


def explicit_nodes():
    return define_bn_nodes(
        {
            "condition": {
                "type": "categorical",
                "modality": "experimental",
                "states": ["A", "B"],
            },
            "gaze": {"type": "continuous", "modality": "gaze", "range": [0, 1]},
            "choice": {"type": "categorical", "modality": "behavior"},
            "valid": {
                "type": "continuous",
                "role": "quality",
                "range": [0, 1],
            },
        }
    )


def test_node_specs_and_defaults():
    q = BayesianNodeSpec(name="q", variable_type="continuous", role="quality")
    assert q.include_in_structure is False
    s = BayesianNodeSpec(name="s", variable_type="categorical", states=("a", "b"))
    assert s.include_in_structure is True
    with pytest.raises(ValueError):
        BayesianNodeSpec(name="", variable_type="continuous")
    with pytest.raises(ValueError):
        BayesianNodeSpec(name="x", variable_type="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BayesianNodeSpec(
            name="x",
            variable_type="continuous",
            role="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        BayesianNodeSpec(name="x", variable_type="continuous", value_range=(2, 1))
    with pytest.raises(ValueError):
        BayesianNodeSpec(name="x", variable_type="categorical", states=("a", "a"))


def test_define_nodes_validation():
    specs = explicit_nodes()
    assert specs["gaze"].value_range == (0, 1)
    assert define_bn_nodes({"gaze": specs["gaze"]})["gaze"] is specs["gaze"]
    with pytest.raises(ValueError):
        define_bn_nodes({})
    with pytest.raises(TypeError):
        define_bn_nodes({"x": 1})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        define_bn_nodes({"wrong": BayesianNodeSpec("x", "continuous")})


def test_prepare_explicit_contract_and_model_family():
    spec = prepare_bayesian_network_data(
        base_data(),
        nodes=explicit_nodes(),
        participant_id="participant_id",
        trial_id="trial_id",
        stimulus_id=None,
        provenance={"detector": "ivt"},
        provenance_columns=["detector_spec_id"],
    )
    assert spec.structure_nodes == ["condition", "gaze", "choice"]
    assert spec.model_family == "mixed"
    assert spec.provenance == {"detector": "ivt"}
    assert "valid" in spec.data and "detector_spec_id" in spec.data
    assert validate_bayesian_network_data(spec).valid


def test_prepare_inference_and_input_errors():
    d = base_data()
    with pytest.warns(UserWarning, match="inferred"):
        spec = prepare_bayesian_network_data(
            d,
            nodes=["condition", "gaze"],
            participant_id="participant_id",
            trial_id="trial_id",
            stimulus_id=None,
        )
    assert spec.model_family == "mixed"
    with pytest.raises(TypeError):
        prepare_bayesian_network_data([], nodes=["x"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        prepare_bayesian_network_data(d, nodes=[], participant_id=None, trial_id=None)
    with pytest.raises(ValueError):
        prepare_bayesian_network_data(d, nodes=["gaze", "gaze"], participant_id=None, trial_id=None)
    with pytest.raises(ValueError):
        prepare_bayesian_network_data(d, nodes=["missing"], participant_id=None, trial_id=None)
    with pytest.raises(ValueError):
        prepare_bayesian_network_data(
            d,
            nodes={"gaze": {"type": "continuous"}},
            node_specs={"gaze": {"type": "continuous"}},
            participant_id=None,
            trial_id=None,
        )
    with pytest.raises(ValueError):
        prepare_bayesian_network_data(
            d,
            nodes=["gaze", "choice"],
            node_specs={"gaze": {"type": "continuous"}},
            participant_id=None,
            trial_id=None,
        )
    with pytest.raises(ValueError):
        prepare_bayesian_network_data(
            d,
            nodes=["gaze"],
            observation_level="bad",
            participant_id=None,
            trial_id=None,
        )
    with pytest.raises(ValueError):
        prepare_bayesian_network_data(
            d,
            nodes=["gaze"],
            provenance_columns=["missing"],
            participant_id=None,
            trial_id=None,
        )


def test_prepare_trial_identity_contract():
    d = base_data().copy()
    with pytest.raises(ValueError, match="requires participant_id"):
        prepare_bayesian_network_data(
            d.drop(columns="participant_id"),
            nodes=["gaze"],
            trial_id="trial_id",
        )
    dup = pd.concat([d, d.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per participant/trial"):
        prepare_bayesian_network_data(
            dup,
            nodes=["gaze"],
            participant_id="participant_id",
            trial_id="trial_id",
        )


def test_validation_errors_warnings_and_strict():
    d = base_data().copy()
    specs = define_bn_nodes(
        {
            "gaze": {
                "type": "continuous",
                "range": [0, 0.5],
                "missing_allowed": False,
            },
            "choice": {"type": "categorical", "states": ["yes"]},
            "valid": {
                "type": "continuous",
                "role": "quality",
                "include_in_structure": True,
            },
        }
    )
    d.loc[0, "gaze"] = np.nan
    prepared = prepare_bayesian_network_data(
        d,
        nodes=specs,
        participant_id="participant_id",
        trial_id="trial_id",
        stimulus_id=None,
    )
    result = validate_bayesian_network_data(prepared)
    assert not result.valid
    assert any("missing values" in x for x in result.errors)
    assert any("outside declared range" in x for x in result.errors)
    assert any("states" in x for x in result.errors)
    assert any("Quality node" in x for x in result.warnings)

    zero_spec = prepare_bayesian_network_data(
        pd.DataFrame({"x": [0.0, 1.0]}),
        nodes={"x": {"type": "continuous", "zero_is_missing": True}},
        participant_id=None,
        trial_id=None,
        stimulus_id=None,
        observation_level="participant",
    )
    z = validate_bayesian_network_data(zero_spec)
    assert z.valid and z.warnings
    assert not validate_bayesian_network_data(zero_spec, strict=True).valid

    bad_numeric = prepare_bayesian_network_data(
        pd.DataFrame({"x": ["a", "2"]}),
        nodes={"x": {"type": "continuous"}},
        participant_id=None,
        trial_id=None,
        stimulus_id=None,
        observation_level="participant",
    )
    assert not validate_bayesian_network_data(bad_numeric).valid
    bad_count = prepare_bayesian_network_data(
        pd.DataFrame({"x": [-1, 2]}),
        nodes={"x": {"type": "count"}},
        participant_id=None,
        trial_id=None,
        stimulus_id=None,
        observation_level="participant",
    )
    assert not validate_bayesian_network_data(bad_count).valid


def test_validation_no_structure_node_and_missing_manual_corruption():
    spec = prepare_bayesian_network_data(
        pd.DataFrame({"q": [0.9, 0.8]}),
        nodes={"q": {"type": "continuous", "role": "quality"}},
        participant_id=None,
        trial_id=None,
        stimulus_id=None,
        observation_level="participant",
    )
    result = validate_bayesian_network_data(spec)
    assert not result.valid and "No nodes" in result.errors[0]
    spec.node_specs["missing"] = BayesianNodeSpec("missing", "continuous")
    assert any("absent" in x for x in validate_bayesian_network_data(spec).errors)


def test_constraints_contract():
    nodes = ["condition", "gaze", "trust", "choice"]
    tiers = define_temporal_tiers(
        [["condition"], ["gaze"], ["trust"], ["choice"]],
        nodes=nodes,
    )
    constraints = define_bn_constraints(
        temporal_tiers=tiers,
        required_edges=[("condition", "gaze")],
        forbidden_edges=[("choice", "gaze")],
        max_parents=3,
        nodes=nodes,
    )
    assert ("choice", "condition") in constraints.forbidden_edges
    assert constraints.max_parents == 3
    assert define_temporal_tiers([]) == ()
    with pytest.raises(ValueError):
        define_temporal_tiers([[], ["x"]])
    with pytest.raises(ValueError):
        define_temporal_tiers([["x"], ["x"]])
    with pytest.raises(ValueError):
        define_temporal_tiers([["x"]], nodes=["y"])
    with pytest.raises(ValueError):
        define_bn_constraints(required_edges=[("x",)])
    with pytest.raises(ValueError):
        define_bn_constraints(required_edges=[("x", "x")])
    with pytest.raises(ValueError):
        define_bn_constraints(max_parents=0)
    with pytest.raises(ValueError):
        define_bn_constraints(required_edges=[("x", "y")], nodes=["x"])
    with pytest.raises(ValueError):
        define_bn_constraints(
            required_edges=[("x", "y")],
            forbidden_edges=[("x", "y")],
        )
    with pytest.raises(ValueError):
        define_bn_constraints(required_edges=[("x", "y"), ("y", "x")])


def test_simulation_and_recovery():
    d = simulate_multimodal_bayesian_network_example(
        n_participants=6,
        trials_per_participant=2,
        random_state=1,
    )
    assert d.shape == (12, 8)
    assert set(d["condition"]) <= {"Explanation", "Control"}
    with pytest.raises(ValueError):
        simulate_multimodal_bayesian_network_example(n_participants=0)
    recovery = validate_bayesian_network_recovery(
        [("a", "b"), ("b", "c"), ("d", "a")],
        [("a", "b"), ("c", "b")],
    )
    assert recovery["true_positive_edges"] == 1
    assert recovery["reversed_true_edges"] == 1
    empty = validate_bayesian_network_recovery([], [])
    assert empty["precision"] == 0.0 and empty["recall"] == 0.0


def test_additional_prepare_branches():
    d = base_data().copy()
    d["flag"] = [True, False, True, False]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inferred = prepare_bayesian_network_data(
            d,
            nodes=["flag"],
            participant_id=None,
            trial_id=None,
            stimulus_id=None,
            observation_level="participant",
        )
    assert inferred.node_specs["flag"].variable_type == "categorical"
    selected = prepare_bayesian_network_data(
        d,
        nodes=["gaze"],
        node_specs=explicit_nodes(),
        participant_id=None,
        trial_id=None,
        stimulus_id=None,
        observation_level="participant",
        provenance_columns=["gaze"],
    )
    assert list(selected.node_specs) == ["gaze"]
    assert list(selected.data.columns).count("gaze") == 1
    with pytest.raises(ValueError):
        prepare_bayesian_network_data(
            d,
            nodes={"missing": {"type": "continuous"}},
            participant_id=None,
            trial_id=None,
            stimulus_id=None,
            observation_level="participant",
        )
    with pytest.raises(TypeError):
        validate_bayesian_network_data(d)  # type: ignore[arg-type]
    assert define_temporal_tiers([["x"]]) == (("x",),)


def test_node_spec_conflicting_type_keys_is_explicit_error():
    with pytest.raises(TypeError):
        define_bn_nodes({"x": {"type": "continuous", "variable_type": "continuous"}})
