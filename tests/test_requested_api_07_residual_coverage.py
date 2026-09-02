from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.requested_api_07 as ra
from eyeprocesspy.irt import EyeResult


def _latent_space() -> EyeResult:
    return EyeResult(
        {
            "person_coordinates": np.array([[0.0, 0.0], [1.0, 0.5]]),
            "item_coordinates": np.array([[0.2, 1.0], [1.1, 1.5]]),
            "person_ids": ["P1", "P2"],
            "item_ids": ["I1", "I2"],
        },
        eyeprocess_class="eye_latent_space_irt",
    )


def test_private_ols_and_plot_distractor_alternate_paths():
    fit = ra._ols_fit(np.column_stack([np.ones(4), np.arange(4.0)]), np.array([1.0, 2.0, 3.0, 4.0]))
    assert fit.eyeprocess_class == "eye_reference_linear_fit"
    assert fit.converged is True

    ax = ep.plot_distractor_information(pd.DataFrame())
    assert ax.get_title() == "Distractor process information"
    plt.close(ax.figure)

    rc = pd.DataFrame({"response_category": ["A", "B"], "gaze_channel": ["g1", "g2"], "coefficient": [0.1, -0.2]})
    fig, ax = plt.subplots()
    out = ep.plot_distractor_information(rc, ax=ax)
    assert out is ax
    plt.close(fig)

    generic = pd.DataFrame({"score": [1.0, 2.0], "label": ["a", "b"]})
    ax = ep.plot_distractor_information(generic)
    assert ax.get_xlabel() == "score"
    plt.close(ax.figure)

    with pytest.raises(ep.EyeProcessValidationError, match="No numeric"):
        ep.plot_distractor_information(pd.DataFrame({"label": ["a"]}))


def test_gaze_missingness_validation_supplied_theta_continuous_and_reached():
    d = pd.DataFrame(
        {
            "participant_id": ["P1", "P1", "P2", "P2", "P3", "P3"],
            "item_id": ["I1", "I2"] * 3,
            "gaze_exposure": [0.1, 0.2, 0.4, 0.7, 1.0, 1.2],
            "theta": [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0],
            "response": [1.2, np.nan, 2.1, 2.4, 3.2, 3.5],
            "reached": [True, False, True, False, True, True],
        }
    )
    fit = ep.fit_gaze_informed_missingness_irt(d, theta="theta", reached="reached")
    assert fit.theta_source == "supplied"
    assert fit.response_model.eyeprocess_class == "eye_reference_linear_fit"
    assert set(fit.reached_summary["reached"]) == {False, True}

    bad = d.copy(); bad.loc[0, "gaze_exposure"] = -0.1
    with pytest.raises(ep.EyeProcessValidationError, match="non-negative"):
        ep.fit_gaze_informed_missingness_irt(bad, theta="theta")

    nonbinary = d.copy(); nonbinary["response"] = [1.0, 2.0, 1.0, np.nan, 0.0, 1.0]
    with pytest.raises(ep.EyeProcessValidationError, match="binary"):
        ep.fit_gaze_informed_missingness_irt(nonbinary)


def test_named_facet_and_changepoint_guards_and_empty_plot():
    with pytest.raises(ep.EyeProcessValidationError, match="fit_manyfacet"):
        ep.device_facet_effects({})

    fake = EyeResult({"facets": {"session": "session"}}, eyeprocess_class="eye_manyfacet_process_irt")
    with pytest.raises(ep.EyeProcessValidationError, match="Facet device"):
        ep.device_facet_effects(fake)

    with pytest.raises(ep.EyeProcessValidationError, match="change-point"):
        ep.plot_process_changepoint({})

    cp = EyeResult({"results": pd.DataFrame()}, eyeprocess_class="eye_irt_changepoints")
    fig, ax = plt.subplots()
    out = ep.plot_process_changepoint(cp, ax=ax)
    assert out is ax
    assert out.get_title() == "Process change points"
    plt.close(fig)


def test_person_item_space_validation_labels_and_interaction_selection():
    obj = _latent_space()
    with pytest.raises(ep.EyeProcessValidationError, match="fit_latent_space"):
        ep.plot_person_item_space({})
    with pytest.raises(ep.EyeProcessValidationError, match="two positive"):
        ep.plot_person_item_space(obj, dimensions=[0, 1])
    with pytest.raises(ep.EyeProcessValidationError, match="unavailable"):
        ep.plot_person_item_space(obj, dimensions=[1, 3])

    ax = ep.plot_person_item_space(obj, labels=True)
    assert len(ax.texts) == 4
    plt.close(ax.figure)

    by_index = ep.explain_latent_interaction(obj, person=[1], item=[2], top=1)
    assert by_index.iloc[0]["person"] == "P1"
    assert by_index.iloc[0]["item"] == "I2"
    by_name = ep.explain_latent_interaction(obj, person="P2", item="I1")
    assert by_name.iloc[0]["person"] == "P2"

    with pytest.raises(ep.EyeProcessValidationError, match="fit_latent_space"):
        ep.explain_latent_interaction({})
    bad_dims = EyeResult(
        {"person_coordinates": np.zeros((1, 2)), "item_coordinates": np.zeros((1, 3))},
        eyeprocess_class="eye_latent_space_irt",
    )
    with pytest.raises(ep.EyeProcessValidationError, match="incompatible"):
        ep.explain_latent_interaction(bad_dims)
    with pytest.raises(ep.EyeProcessValidationError, match="No matching"):
        ep.explain_latent_interaction(obj, person="missing")


def test_irf_uncertainty_guards_and_covariance_path():
    with pytest.raises(ep.EyeProcessValidationError, match="fit_gpirt"):
        ep.plot_irf_uncertainty({})

    external = EyeResult({"engine": "external"}, eyeprocess_class="eye_gpirt")
    with pytest.raises(ep.EyeProcessBackendError, match="External GPIRT"):
        ep.plot_irf_uncertainty(external)

    model = EyeResult(
        {
            "theta_degree": 1,
            "coefficients": np.array([0.0, 1.0]),
            "covariance": np.eye(2) * 0.04,
        },
        eyeprocess_class="eye_reference_fit",
    )
    gp = EyeResult(
        {"engine": "spline_reference", "item_names": ["I1"], "models": [model]},
        eyeprocess_class="eye_gpirt",
    )
    ax = ep.plot_irf_uncertainty(gp, item="I1", theta_grid=[-1.0, 0.0, 1.0])
    assert len(ax.gp3_data) == 3
    assert (ax.gp3_data["upper"] >= ax.gp3_data["lower"]).all()
    plt.close(ax.figure)

    with pytest.raises(ep.EyeProcessValidationError, match="Unknown item"):
        ep.plot_irf_uncertainty(gp, item=2)


def test_latent_distribution_guard_paths():
    with pytest.raises(ep.EyeProcessValidationError, match="eight"):
        ep.audit_latent_distribution([1, 2, 3])
    with pytest.raises(ep.EyeProcessValidationError, match="variance"):
        ep.audit_latent_distribution(np.ones(10))
    with pytest.raises(ep.EyeProcessValidationError, match="20"):
        ep.compare_latent_distribution_models(np.arange(10.0))


def test_event_time_external_invalid_engine_and_negative_time_paths():
    d = pd.DataFrame(
        {
            "event_time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 1],
            "theta": [-1.0, -0.2, 0.2, 1.0],
            "participant_id": ["P1", "P2", "P3", "P4"],
            "item_id": ["I1", "I1", "I2", "I2"],
        }
    )
    with pytest.raises(ep.EyeProcessBackendError, match="external_engine"):
        ep.fit_event_time_irt(d, engine="external")
    ext = ep.fit_event_time_irt(d, engine="external", external_engine=lambda **kwargs: {"ok": True})
    assert ext.engine == "external"
    with pytest.raises(ep.EyeProcessValidationError, match="engine"):
        ep.fit_event_time_irt(d, engine="unknown")
    bad = d.copy(); bad.loc[0, "event_time"] = -1.0
    with pytest.raises(ep.EyeProcessValidationError, match="non-negative"):
        ep.fit_event_time_irt(bad)


def test_simulation_dispatch_and_truth_extraction_all_forms():
    mapped = ep.simulate_from_model({"simulate_fun": lambda n=2: {"data": list(range(n))}}, n=3)
    assert mapped["data"] == [0, 1, 2]
    with pytest.raises(ep.EyeProcessValidationError, match="registered IRT"):
        ep.simulate_from_model(object())

    df_truth = pd.DataFrame({"parameter": ["a"], "truth": [1.0], "extra": [9]})
    out = ep.extract_parameter_truth({"truth": df_truth})
    assert out.columns.tolist() == ["parameter", "truth"]
    arr = ep.extract_parameter_truth({"parameters": [1.0, 2.0]})
    assert arr["parameter"].tolist() == ["parameter_1", "parameter_2"]
    with pytest.raises(ep.EyeProcessValidationError, match="parameter truth"):
        ep.extract_parameter_truth({})
    with pytest.raises(ep.EyeProcessValidationError, match="missing required"):
        ep.extract_parameter_truth({"truth": pd.DataFrame({"parameter": ["a"]})})


def test_validation_replicate_callable_guard_and_failure_taxonomy():
    with pytest.raises(ep.EyeProcessValidationError, match="must be functions"):
        ep.fit_validation_replicate(1, 1, lambda x: x, lambda x, y: x)

    failed = ep.fit_validation_replicate(
        2,
        generator=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("generator failed")),
        fitter=lambda x: x,
        extractor=lambda fit, sim: fit,
        scenario={"kind": "custom"},
        engine="test",
    )
    assert not bool(failed.loc[0, "converged"])
    assert failed.loc[0, "scenario"] == "custom"
    assert "generator failed" in failed.loc[0, "failure_message"]


def test_vendor_contract_semantics_alias_units_and_guards():
    with pytest.raises(ep.EyeProcessValidationError, match="vendor"):
        ep.vendor_schema_contract("")

    contract = ep.vendor_schema_contract(
        "Demo",
        required_fields=["time"],
        optional_fields=["x"],
        aliases={"pupil": ["pupil_mm"]},
        units={"pupil": "mm"},
    )
    with pytest.raises(ep.EyeProcessValidationError, match="vendor_schema_contract"):
        ep.validate_vendor_semantics(pd.DataFrame(), {})

    d = pd.DataFrame({"time": [1.0], "pupil_mm": [3.0]})
    z = ep.validate_vendor_semantics(d, contract, metadata={"pupil_units": "mm"})
    assert bool(z.aliases.loc[0, "matched"])
    assert bool(z.units.loc[0, "pass"])


def test_event_roundtrip_without_hed_and_sample_extractor_branches():
    ev = pd.DataFrame({"event_id": [1], "event": ["start"], "timestamp": [0.0]})
    z = ep.event_roundtrip_audit(ev, ev, key="event_id")
    assert z.hed is None

    frame = pd.DataFrame({"x": [1]})
    pd.testing.assert_frame_equal(ra._extract_samples(frame), frame)
    for key in ("samples", "data", "gaze"):
        pd.testing.assert_frame_equal(ra._extract_samples({key: frame}), frame)


def test_roundtrip_callable_and_cross_version_guards():
    source = pd.DataFrame({"x": [1]})
    with pytest.raises(ep.EyeProcessValidationError, match="must be functions"):
        ep.roundtrip_eye_bids(source, exporter=1, importer=lambda x: x)
    with pytest.raises(ep.EyeProcessValidationError, match="must be functions"):
        ep.cross_version_adapter_regression(source, baseline_adapter=1, candidate_adapter=lambda x: x)
