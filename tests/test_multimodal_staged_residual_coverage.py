from __future__ import annotations

import importlib.metadata as importlib_metadata

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.multimodal_staged as mm
from eyeprocesspy.exceptions import (
    EyeProcessBackendError,
    EyeProcessValidationError,
)


def _channels():
    response = ep.irt_response_channel("rasch", latent="theta")
    rt = ep.irt_rt_channel(latent="tau")
    gaze = ep.irt_count_channel(value="gaze", latent="omega")
    pupil = ep.irt_continuous_channel(
        "gaussian",
        value="pupil",
        lower=-1e12,
        upper=1e12,
        latent="rho",
    )
    return response, rt, gaze, pupil


def _tiny_measurement(include_response: bool = True):
    data = pd.DataFrame(
        {
            "person": ["P1", "P2", "P3", "P4"],
            "item": ["I1", "I1", "I2", "I2"],
            "response": [0, 1, 1, 0],
            "rt": [0.5, 0.6, 0.7, 0.8],
            "gaze": [1.0, 2.0, 3.0, 4.0],
            "pupil": [0.1, 0.2, 0.3, 0.4],
        }
    )
    return ep.prepare_multimodal_irt_data(
        data,
        person="person",
        item="item",
        response="response" if include_response else None,
        rt="rt",
        gaze="gaze",
        pupil="pupil",
    )


def _m4_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rt": np.linspace(0.4, 0.9, 6),
            "gaze": np.arange(1.0, 7.0),
            "pupil": np.linspace(-0.2, 0.3, 6),
            "pupil_quality": np.array([-1.0, -0.7, -0.2, 0.2, 0.7, 1.0]),
            "device": ["device_A", "device_B"] * 3,
        }
    )


def test_multimodal_private_coercion_key_and_graph_residuals():
    frame = pd.DataFrame({"a": [1, 2]})
    pd.testing.assert_frame_equal(mm._df({"data": frame}), frame)

    with pytest.raises(EyeProcessValidationError, match="coercible"):
        mm._df(object())
    with pytest.raises(EyeProcessValidationError, match="missing required"):
        mm._req(frame, ["a", "b"])
    with pytest.raises(EyeProcessValidationError, match="Expected a data frame"):
        mm._extract_data(object())

    bad = pd.DataFrame(
        {"person": ["P1", None], "item": ["I1", "I2"], "trial": ["T1", "T2"]}
    )
    with pytest.raises(EyeProcessValidationError, match="non-missing"):
        mm._validate_key(bad, "person", "item", "trial")

    graph = pd.DataFrame({"person": ["P1"], "item": ["I1"]})
    assert mm._connected_design(graph, "person", "item", pd.Series([False])) == 0


def test_multimodal_audit_empty_channel_duplicate_nonnumeric_and_issue_paths():
    no_channels = ep.prepare_multimodal_irt_data(
        pd.DataFrame({"person": ["P1", "P2"], "item": ["I1", "I2"]}),
        person="person",
        item="item",
    )
    audit = ep.audit_multimodal_measurement(no_channels)
    assert "no_measurement_channels" in audit.issues

    text_response = ep.prepare_multimodal_irt_data(
        pd.DataFrame(
            {
                "person": ["P1", "P2"],
                "item": ["I1", "I2"],
                "response": ["yes", "no"],
            }
        ),
        person="person",
        item="item",
        response="response",
    )
    assert np.isnan(ep.audit_multimodal_measurement(text_response).channel_table.finite_fraction.iloc[0])

    problematic = ep.prepare_multimodal_irt_data(
        pd.DataFrame(
            {
                "person": ["P1", "P2"],
                "item": ["I1", "I2"],
                "rt": [0.5, -0.1],
                "gaze": [1.0, -2.0],
            }
        ),
        person="person",
        item="item",
        rt="rt",
        gaze="gaze",
    )
    issues = set(ep.audit_multimodal_measurement(problematic).issues)
    assert {"nonpositive_response_time", "negative_gaze_measurement"} <= issues

    duplicated = _tiny_measurement()
    duplicated["data"] = pd.concat(
        [duplicated["data"], duplicated["data"].iloc[[0]]], ignore_index=True
    )
    assert "duplicated_person_item_trial_keys" in ep.audit_multimodal_measurement(
        duplicated
    ).issues


def test_multimodal_generic_spec_all_models_and_guards():
    response, rt, gaze, pupil = _channels()

    m0 = ep.multimodal_irt_spec(response=response, model="M0", backend="existing")
    m1 = ep.multimodal_irt_spec(response=response, rt=rt, model="M1")
    m2 = ep.multimodal_irt_spec(response=response, rt=rt, gaze=gaze, model="M2")
    m3 = ep.multimodal_irt_spec(
        response=response, rt=rt, gaze=gaze, pupil=pupil, model="M3", backend="cmdstanpy"
    )
    assert [m0.model, m1.model, m2.model, m3.model] == ["M0", "M1", "M2", "M3"]

    shared_rt = dict(rt)
    shared_rt["latent"] = "theta"
    shared = ep.multimodal_irt_spec(response=response, rt=shared_rt, model="M1")
    assert shared.latent == ["theta"]

    with pytest.raises(EyeProcessValidationError, match="model must"):
        ep.multimodal_irt_spec(response=response, model="M9")
    with pytest.raises(EyeProcessValidationError, match="Unsupported backend"):
        ep.multimodal_irt_spec(response=response, backend="other")
    with pytest.raises(EyeProcessValidationError, match="At least one"):
        ep.multimodal_irt_spec()
    with pytest.raises(EyeProcessValidationError, match="existing eyeprocess"):
        ep.multimodal_irt_spec(
            response={"superclass": "wrong", "latent": "theta"}, model="M0"
        )
    with pytest.raises(EyeProcessValidationError, match="requires channels"):
        ep.multimodal_irt_spec(response=response, model="M2")

    missing_latent = dict(response)
    missing_latent["latent"] = ""
    with pytest.raises(EyeProcessValidationError, match="explicit latent"):
        ep.multimodal_irt_spec(response=missing_latent, model="M0")


def test_multimodal_generic_simulation_information_backend_ppc_and_validation(monkeypatch):
    with pytest.raises(EyeProcessValidationError, match="must exceed 1"):
        ep.simulate_multimodal_irt(n_person=1, n_item=2)
    with pytest.raises(EyeProcessValidationError, match="positive-definite"):
        ep.simulate_multimodal_irt(n_person=2, n_item=2, latent_cor=np.eye(3))
    with pytest.raises(EyeProcessValidationError, match="missing_fraction"):
        ep.simulate_multimodal_irt(n_person=2, n_item=2, missing_fraction=1)

    sim = ep.simulate_multimodal_irt(
        n_person=3,
        n_item=3,
        latent_cor=np.eye(4),
        missing_fraction=0.4,
        seed=123,
    )
    assert sim.eyeprocess_class == "eye_multimodal_simulation"

    baseline = np.array([0.0, 2.0, 4.0, 6.0])
    augmented = np.array([1.0, 2.0, 3.0, 4.0])
    precision = ep.process_information(baseline, augmented, metric="precision_gain")
    entropy = ep.process_information(baseline, augmented, metric="entropy_reduction")
    assert precision.value.iloc[0] > 0
    assert entropy.value.iloc[0] > 0
    assert ep.validate_multimodal_irt(precision).valid

    with pytest.raises(EyeProcessValidationError, match="same columns"):
        ep.process_information(np.ones((4, 2)), np.ones((4, 1)))
    with pytest.raises(EyeProcessValidationError, match="variances"):
        ep.process_information(np.ones(4), np.arange(4.0))
    with pytest.raises(EyeProcessValidationError, match="Invalid information metric"):
        ep.process_information(baseline, augmented, metric="wrong")

    def fake_find_spec(name):
        return object() if name in {"cmdstanpy", "arviz"} else None

    def fake_version(name):
        if name == "cmdstanpy":
            return "9.9.9"
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(mm, "find_spec", fake_find_spec)
    monkeypatch.setattr(importlib_metadata, "version", fake_version)
    status = ep.multimodal_backend_status().set_index("backend")
    assert status.loc["cmdstanpy", "installed"]
    assert status.loc["cmdstanpy", "version"] == "9.9.9"
    assert status.loc["arviz", "installed"]
    assert pd.isna(status.loc["arviz", "version"])

    ppc_default = ep.multimodal_ppc({"data": sim.data})
    ppc_one = ep.multimodal_ppc({"data": sim.data}, variables=["response"])
    assert not ppc_default.summary.empty
    assert {"response", "rt"} <= set(ppc_default.summary["variable"])
    assert ppc_one.summary.variable.tolist() == ["response"]
    with pytest.raises(EyeProcessValidationError, match="Unsupported object"):
        ep.multimodal_ppc(object())

    measurement = _tiny_measurement()
    assert ep.validate_multimodal_irt(measurement).valid
    with pytest.raises(EyeProcessValidationError, match="Unsupported object class"):
        ep.validate_multimodal_irt(object())


def test_multimodal_ablation_and_structural_identifiability_residuals():
    no_response = _tiny_measurement(include_response=False)
    with pytest.raises(EyeProcessValidationError, match="Response channel"):
        ep.ablate_multimodal_channels(no_response)

    free = ep.ablate_multimodal_channels(no_response, include_response=False)
    assert any("rt" in obj.channels for obj in free.scenarios.values())

    with pytest.raises(EyeProcessValidationError, match="eye_multimodal_measurement"):
        ep.ablate_multimodal_channels(object())

    high_missing = _tiny_measurement()
    high_missing["data"].loc[:2, "pupil"] = np.nan
    audit = ep.audit_multimodal_identifiability(high_missing, min_person=1, min_item=1)
    assert "channel_missingness_ge_50_percent" in audit.issues


def test_m2_spec_backend_simulation_mapping_and_ablation_residuals():
    assert ep.multimodal_m2_spec(
        backend="cmdstanpy", prior_profile="paper_centered"
    ).backend == "cmdstanpy"
    with pytest.raises(EyeProcessValidationError, match="canonical CmdStan"):
        ep.multimodal_m2_spec(backend="brms")
    with pytest.raises(EyeProcessValidationError, match="prior_profile"):
        ep.multimodal_m2_spec(prior_profile="bad")
    with pytest.raises(EyeProcessBackendError, match="requires CmdStanPy"):
        ep.fit_multimodal_m2(pd.DataFrame())

    mapped = ep.simulate_multimodal_m2(
        n_person=2,
        n_item=3,
        mu_item={"b": 0.0, "beta": 4.0, "m": 3.5},
        sd_person={"theta": 1.0, "tau": 0.5, "omega": 0.5},
        cor_person=np.eye(3),
        sd_item={"b": 0.75, "beta": 0.35, "m": 0.6},
        cor_item=np.eye(3),
        gaze_shape={"shape": 2.0},
        dropout={"response": 0.0, "rt": 0.0, "gaze": 0.0},
        seed=11,
    )
    assert len(mapped.data) == 6

    with pytest.raises(EyeProcessValidationError, match="dropout must contain"):
        ep.simulate_multimodal_m2(n_person=2, n_item=3, dropout=(0, 0), seed=1)
    with pytest.raises(EyeProcessValidationError, match="dropout fractions"):
        ep.simulate_multimodal_m2(n_person=2, n_item=3, dropout=(-0.1, 0, 0), seed=1)

    ablation = ep.multimodal_m2_ablation(mapped)
    assert set(ablation.scenarios) == {"M0", "M1", "M2"}

    ppc_fit = mm._result("eye_multimodal_m2_fit", data=mapped.data)
    assert ep.multimodal_m2_ppc(ppc_fit).status == "posterior_predictive"
    assert ep.validate_multimodal_m2(mapped, include_ppc=False).ppc is None


def test_m3_spec_backend_missingness_modes_functional_bridge_and_ablation():
    with pytest.raises(EyeProcessValidationError, match="prior_profile"):
        ep.multimodal_m3_spec(prior_profile="bad")
    with pytest.raises(EyeProcessValidationError, match="pupil representation"):
        ep.multimodal_m3_spec(pupil_representation="raw")
    custom = ep.multimodal_m3_spec(
        backend="cmdstanpy",
        pupil_representation="functional_score",
        nuisance={"baseline": False},
    )
    assert custom.pupil_nuisance == {"baseline": False}
    with pytest.raises(EyeProcessBackendError, match="requires CmdStanPy"):
        ep.fit_multimodal_m3(pd.DataFrame())

    for offset, mechanism in enumerate(["quality", "gaze", "ability", "device"]):
        sim = ep.simulate_multimodal_m3(
            n_person=2,
            n_item=3,
            pupil_missingness=mechanism,
            dropout=(0, 0, 0, 0),
            seed=100 + offset,
        )
        assert sim.truth["pupil_missingness"] == mechanism

    with pytest.raises(EyeProcessValidationError, match="Invalid pupil_signal"):
        ep.simulate_multimodal_m3(n_person=2, n_item=3, pupil_signal="bad")
    with pytest.raises(EyeProcessValidationError, match="Invalid pupil_missingness"):
        ep.simulate_multimodal_m3(n_person=2, n_item=3, pupil_missingness="bad")

    d = ep.simulate_multimodal_m3(
        n_person=2,
        n_item=3,
        pupil_missingness="none",
        dropout=(0, 0, 0, 0),
        seed=19,
    ).data
    with pytest.raises(EyeProcessValidationError, match="one finite-or-NA"):
        ep.multimodal_m3_functional_bridge(d, 1.2)
    with pytest.raises(EyeProcessValidationError, match="one finite-or-NA"):
        ep.multimodal_m3_functional_bridge(d, "absent_score")
    bridge = ep.multimodal_m3_functional_bridge(
        d, np.linspace(0, 1, len(d)).tolist()
    )
    assert np.isfinite(bridge.data.pupil).all()
    assert set(ep.multimodal_m3_ablation(d).scenarios) == {"M0", "M1", "M2", "M3"}


def test_m4_spec_private_transition_missingness_and_backend_guards():
    valid = ep.multimodal_m4_spec(
        n_states=2,
        transition_structure="iid",
        trait_conditioning=(),
        initial_trait_conditioning=False,
        backend="cmdstanpy",
    )
    assert valid.transition_structure == "iid"

    for kwargs, pattern in [
        ({"state_channels": ()}, "non-empty subset"),
        ({"transition_structure": "bad"}, "transition_structure"),
        ({"trait_conditioning": ("bad",)}, "trait_conditioning"),
        ({"backend": "brms"}, "only CmdStan"),
        ({"missingness": "MNAR"}, "ignorable"),
        ({"min_sequence_length": 0}, "positive integer"),
    ]:
        with pytest.raises(EyeProcessValidationError, match=pattern):
            ep.multimodal_m4_spec(**kwargs)

    with pytest.raises(EyeProcessBackendError, match="requires CmdStanPy"):
        ep.fit_multimodal_m4(
            pd.DataFrame(),
            spec=ep.multimodal_m4_spec(
                n_states=1,
                transition_structure="iid",
                trait_conditioning=(),
                initial_trait_conditioning=False,
                backend="cmdstanpy",
            ),
        )

    traits = np.array([[0.0, 0.0], [0.5, -0.2]])
    pi, transition = mm._m4_transition_array(traits, 2, "rapid_switch")
    np.testing.assert_allclose(pi.sum(axis=1), 1.0)
    np.testing.assert_allclose(transition.sum(axis=2), 1.0)

    frame = _m4_frame()
    state = np.array([1, 2, 1, 2, 1, 2])
    none_by_rate = mm._m4_apply_missingness(
        np.random.default_rng(1), frame, "mcar", 0, state, 2
    )
    pd.testing.assert_frame_equal(none_by_rate, frame)

    for seed, mechanism in enumerate(
        ["mcar", "quality", "gaze", "pupil_quality", "device", "state_dependent"], 10
    ):
        out = mm._m4_apply_missingness(
            np.random.default_rng(seed), frame, mechanism, 0.4, state, 2
        )
        assert len(out) == len(frame)

    with pytest.raises(EyeProcessValidationError, match="more than one true state"):
        mm._m4_apply_missingness(
            np.random.default_rng(1), frame, "state_dependent", 0.2, np.ones(6), 1
        )
    with pytest.raises(EyeProcessValidationError, match="Unknown M4 missingness mechanism"):
        mm._m4_apply_missingness(
            np.random.default_rng(1), frame, "unknown", 0.2, state, 2
        )


def test_m4_simulation_guard_redundancy_and_missingness_paths():
    with pytest.raises(EyeProcessValidationError, match="Unknown M4 simulation"):
        ep.simulate_multimodal_m4(n_person=2, n_item=3, scenario="bad")
    with pytest.raises(EyeProcessValidationError, match="Unknown M4 missingness"):
        ep.simulate_multimodal_m4(n_person=2, n_item=3, missingness="bad")
    with pytest.raises(EyeProcessValidationError, match="at least 2 persons"):
        ep.simulate_multimodal_m4(n_person=1, n_item=3)
    with pytest.raises(EyeProcessValidationError, match="1 through 4"):
        ep.simulate_multimodal_m4(n_person=2, n_item=3, n_states=5)
    with pytest.raises(EyeProcessValidationError, match="state_dependent"):
        ep.simulate_multimodal_m4(
            n_person=2,
            n_item=3,
            scenario="null",
            missingness="state_dependent",
        )

    for seed, scenario in enumerate(
        ["rt_redundant", "gaze_redundant", "pupil_redundant"], 30
    ):
        sim = ep.simulate_multimodal_m4(
            n_person=2, n_item=3, scenario=scenario, seed=seed
        )
        assert sim.truth["n_states"] == 2

    for seed, mechanism in enumerate(
        ["mcar", "quality", "gaze", "pupil_quality", "device", "state_dependent"], 40
    ):
        sim = ep.simulate_multimodal_m4(
            n_person=2,
            n_item=3,
            n_states=2,
            missingness=mechanism,
            missing_rate=0.3,
            seed=seed,
        )
        assert sim.missingness == mechanism


def test_m4_state_diagnostics_identifiability_and_execution_boundaries():
    sim = ep.simulate_multimodal_m4(n_person=2, n_item=3, seed=70)
    assert ep.multimodal_m4_state_diagnostics(sim).source == "synthetic_truth"

    fake_fit = mm._result(
        "eye_multimodal_m4_fit",
        state_probability=pd.DataFrame(
            {"state_1_probability": [0.8], "state_2_probability": [0.2]}
        ),
        spec=ep.multimodal_m4_spec(
            n_states=2,
            transition_structure="iid",
            trait_conditioning=(),
            initial_trait_conditioning=False,
            backend="cmdstanpy",
        ),
    )
    posterior = ep.multimodal_m4_state_diagnostics(fake_fit)
    assert posterior.source == "posterior"
    assert ep.multimodal_m4_process_information(fake_fit).status == (
        "posterior_state_information"
    )
    with pytest.raises(EyeProcessValidationError, match="State diagnostics require"):
        ep.multimodal_m4_state_diagnostics(mm._result("wrong"))

    mixed = sim.data.copy()
    mixed.loc[1, "sequence_id"] = mixed.loc[0, "sequence_id"]
    mixed.loc[1, "person_id"] = "OTHER"
    with pytest.raises(EyeProcessValidationError, match="exactly one person"):
        ep.audit_multimodal_m4_identifiability(mixed, include_posterior=False)

    empty = pd.DataFrame(
        columns=[
            "person_id",
            "item_id",
            "sequence_id",
            "trial_index",
            "response",
            "rt",
            "gaze",
            "pupil",
        ]
    )
    empty_audit = ep.audit_multimodal_m4_identifiability(
        empty,
        spec=ep.multimodal_m4_spec(
            n_states=1,
            transition_structure="iid",
            trait_conditioning=(),
            initial_trait_conditioning=False,
            backend="cmdstanpy",
        ),
        include_posterior=False,
    )
    assert empty_audit.overall == "REVIEW"

    with pytest.raises(EyeProcessBackendError, match="negative-control fits"):
        ep.multimodal_m4_negative_controls(sim, run=True)
    with pytest.raises(EyeProcessBackendError, match="state-count sensitivity"):
        ep.multimodal_m4_sensitivity(sim, run=True)
    with pytest.raises(EyeProcessBackendError, match="M4 ablations"):
        ep.multimodal_m4_ablation(sim, run=True)

    ablation = ep.multimodal_m4_ablation(sim, include_channel_ablations=True)
    assert {"rt", "gaze", "pupil"} <= set(ablation.design.ablation)
    assert ep.validate_multimodal_m4(sim, include_ppc=False).ppc is None
