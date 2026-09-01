import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.exceptions import EyeProcessBackendError
from eyeprocesspy.irt import EyeResult


def _process_data() -> pd.DataFrame:
    rows = []
    for p in range(12):
        for j in range(5):
            ability = (p - 5.5) / 3.0
            item_shift = (j - 2) * 0.25
            response = int((p + 2 * j) % 5 not in {0, 1})
            rows.append(
                {
                    "participant_id": f"p{p + 1}",
                    "item_id": f"i{j + 1}",
                    "item_order": j + 1,
                    "response": response,
                    "rt": float(np.exp(1.0 + 0.08 * j - 0.04 * ability)),
                    "response_time": float(np.exp(1.0 + 0.08 * j - 0.04 * ability)),
                    "fixation_count": int(2 + ((p + j) % 6)),
                    "theta": ability,
                    "process_a": float(1.0 + 0.2 * p + 0.1 * j),
                    "process_b": float(2.0 + 0.15 * p - 0.05 * j),
                    "group": "A" if p < 6 else "B",
                    "device": "d1" if p % 2 == 0 else "d2",
                    "session": f"s{1 + p % 3}",
                    "site": "north" if p < 6 else "south",
                    "reached": True,
                }
            )
    return pd.DataFrame(rows)


def _recovery_data() -> pd.DataFrame:
    rows = []
    for engine, shift in [("reference", 0.015), ("alternate", -0.02)]:
        for replicate in range(1, 7):
            for parameter, truth in [("a", 1.0), ("b", 0.0)]:
                estimate = truth + shift + (replicate - 3.5) * 0.004
                rows.append(
                    {
                        "replicate": replicate,
                        "parameter": parameter,
                        "truth": truth,
                        "estimate": estimate,
                        "lower": estimate - 0.20,
                        "upper": estimate + 0.20,
                        "converged": True,
                        "engine": engine,
                        "scenario": "baseline",
                    }
                )
    return pd.DataFrame(rows)


def test_registry_channels_simulation_validation_comparison_and_promotion():
    channels = {
        "rt": ep.irt_rt_channel(),
        "survival": ep.irt_survival_channel(family="weibull"),
        "nominal": ep.irt_nominal_channel(categories=["A", "B", "C"]),
        "composition": ep.irt_compositional_channel(["aoi_a", "aoi_b"]),
        "sequence": ep.irt_sequence_channel(family="ngram"),
        "functional": ep.irt_functional_channel(value="pupil", time="time"),
        "continuous": ep.irt_continuous_channel(lower=0.0, upper=1.0),
    }
    assert all(z.superclass == "eye_irt_channel" for z in channels.values())

    spec = ep.irt_model_spec(
        id="p4_completion_custom_model",
        latent="ability",
        channels={"response": ep.irt_response_channel()},
        simulate_fun=lambda n=4: pd.DataFrame({"response": [0, 1] * (n // 2)}),
    )
    registered = ep.register_irt_model(spec, overwrite=True)
    assert registered.id == spec.id
    assert ep.get_irt_model(spec.id).id == spec.id
    assert ep.get_irt_model("joint_gaze_rt").status == "reference"

    simulated = ep.simulate_irt_model(spec, n=4)
    assert simulated.response.tolist() == [0, 1, 0, 1]
    validation = ep.validate_irt_model(spec, {"pass": True, "grade": "strong"})
    assert validation.pass_value is True

    comparison = ep.compare_irt_models(
        {"logLik": -10.0, "AIC": 26.0, "BIC": 29.0},
        {"logLik": -9.0, "AIC": 24.0, "BIC": 27.0},
        names=["first", "second"],
    )
    assert comparison.model.tolist() == ["first", "second"]
    assert comparison.attrs["eyeprocess_class"] == "eye_irt_model_comparison"

    promoted = ep.promote_irt_model(spec, {"pass": True, "grade": "strong"}, target="reference")
    assert promoted.to == "reference"
    assert promoted.evidence_pass is True


def test_process_irt_reference_models_have_numerical_contracts():
    data = _process_data()
    joint = ep.fit_joint_gaze_rt_irt(data)
    assert joint.eyeprocess_class == "eye_joint_gaze_rt_irt"
    assert joint.data_n == len(data)
    assert np.isfinite(joint.person_covariance).all()

    sae = ep.fit_speed_accuracy_engagement_irt(data)
    assert sae.eyeprocess_class == "eye_speed_accuracy_engagement_irt"

    graded = ep.fit_joint_graded_rt_process_irt(data, process="fixation_count")
    assert graded.eyeprocess_class == "eye_joint_graded_rt_process_irt"
    assert graded.data_n == len(data)

    multi = ep.fit_multimodal_trait_irt(
        data,
        response="response",
        rt="rt",
        gaze="fixation_count",
        person="participant_id",
        item="item_id",
    )
    assert multi.eyeprocess_class == "eye_multimodal_trait_irt"
    assert multi.trait_label == "trait"

    person_fit = ep.process_person_fit(joint, data=data)
    assert person_fit.attrs["eyeprocess_class"] == "eye_process_person_fit"
    assert len(person_fit) == data.participant_id.nunique()
    assert person_fit.empirical_percentile.between(0, 1).all()


def test_exposure_omission_changepoint_equating_and_revisit_contracts():
    data = _process_data()
    data["exposure_predictor"] = data.theta + 0.25 * (data.group == "B")
    data.loc[data.index % 9 == 0, "reached"] = False

    exposure = ep.estimate_visual_exposure_probability(
        data,
        exposed="reached",
        predictors=["exposure_predictor", "group"],
    )
    assert exposure.eyeprocess_class == "eye_visual_exposure_model"
    assert np.all((exposure.fitted_probability >= 0) & (exposure.fitted_probability <= 1))

    omission = data.copy()
    omission.loc[omission.index % 11 == 0, "response"] = np.nan
    omission_fit = ep.fit_omission_survival_irt(omission)
    assert omission_fit.eyeprocess_class == "eye_omission_survival_irt"
    assert {"events", "exposure", "hazard"}.issubset(omission_fit.omission_model.columns)

    cp_rt = ep.fit_changepoint_rt_irt(data, min_segment=2, min_delta_sic=0)
    cp_multi = ep.fit_changepoint_multimodal_irt(data, min_segment=2, min_delta_sic=0)
    assert cp_rt.eyeprocess_class == "eye_changepoint_rt_irt"
    assert cp_multi.eyeprocess_class == "eye_changepoint_multimodal_irt"
    assert len(cp_rt.changepoints.results) == data.participant_id.nunique()

    eq_data = data.assign(
        process_value=1.2 + 0.6 * data.theta + 0.03 * data.fixation_count,
        reference_value=0.4 + 1.1 * (1.2 + 0.6 * data.theta + 0.03 * data.fixation_count),
    )
    equating = ep.cross_device_process_equating_audit(
        eq_data,
        value="process_value",
        reference_value="reference_value",
        device="device",
    )
    assert equating.attrs["eyeprocess_class"] == "eye_cross_device_equating_audit"
    assert len(equating) == 2
    assert np.isfinite(equating.rmse).all()

    response_matrix = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=float)
    q_matrix = np.array([[1, 0], [0, 1], [1, 1]], dtype=int)
    process = pd.DataFrame(
        {
            "participant_id": np.repeat(["p1", "p2", "p3", "p4"], 3),
            "revisited": [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0],
            "response_time": np.linspace(1.0, 2.1, 12),
            "gaze": np.arange(1, 13),
        }
    )
    revisit = ep.fit_revisit_process_cdm(response_matrix, q_matrix, process, gaze="gaze")
    assert revisit.eyeprocess_class == "eye_revisit_process_cdm"
    assert revisit.mastery_profiles.shape == (4, 2)


def test_advanced_process_irt_external_gates_and_reference_diagnostics():
    response_matrix = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=float)
    q_matrix = np.array([[1, 0], [0, 1], [1, 1]], dtype=int)

    with pytest.raises(EyeProcessBackendError, match="GDINA"):
        ep.fit_cognitive_diagnosis_process(response_matrix, q_matrix)
    cognitive = ep.fit_cognitive_diagnosis_process(
        response_matrix,
        q_matrix,
        engine="external",
        external_engine=lambda response_matrix, q_matrix: {"shape": response_matrix.shape, "q": q_matrix.shape},
    )
    assert cognitive.eyeprocess_class == "eye_cognitive_diagnosis_process"
    assert cognitive.response_model["shape"] == (4, 3)

    with pytest.raises(EyeProcessBackendError, match="variational"):
        ep.fit_variational_irt(response_matrix)
    variational = ep.fit_variational_irt(
        response_matrix,
        external_engine=lambda response_matrix: {"mean": float(np.mean(response_matrix))},
    )
    assert variational.eyeprocess_class == "eye_variational_irt"
    assert variational.engine == "external"

    latent = EyeResult(
        {
            "person_coordinates": np.array([[0.0, 0.0], [1.0, 0.2], [0.2, 1.0], [1.1, 1.2]]),
            "item_coordinates": np.array([[0.0, 0.0], [0.8, 0.3], [0.4, 1.1]]),
            "person_ids": ["p1", "p2", "p3", "p4"],
            "item_ids": ["i1", "i2", "i3"],
        },
        eyeprocess_class="eye_latent_space_irt",
    )
    residual_map = ep.process_residual_map(latent)
    assert set(residual_map.entity_type) == {"person", "item"}
    similarity = ep.validate_latent_space_process_similarity(
        latent,
        np.array([[0.0, 0.1], [0.9, 0.3], [0.3, 0.9], [1.0, 1.3]]),
    )
    assert similarity.eyeprocess_class == "eye_latent_space_process_validation"
    assert np.isfinite(similarity.spearman_distance_correlation)


def test_process_dif_surrogate_and_adjusted_audit_are_explicit():
    data = _process_data()
    surrogate = ep.process_dif_nuisance_surrogate(
        data,
        process_features=["process_a", "process_b"],
    )
    assert len(surrogate) == data.participant_id.nunique()
    assert np.isfinite(surrogate.process_nuisance).all()

    audit = ep.audit_process_adjusted_dif(
        data,
        response="response",
        ability="theta",
        group="group",
        process_features=["process_a", "process_b"],
    )
    assert audit.eyeprocess_class == "eye_process_adjusted_dif"
    assert {"term", "unadjusted", "adjusted", "absolute_reduction"}.issubset(audit.coefficients.columns)


def test_recovery_audits_and_engine_comparison_are_numerical():
    raw = _recovery_data()
    recovery = ep.as_irt_recovery_results(raw)
    assert recovery.attrs["eyeprocess_class"] == "eye_irt_recovery_results"
    assert np.allclose(recovery.error, recovery.estimate - recovery.truth)

    bias = ep.audit_bias(recovery, threshold=0.10)
    rmse = ep.audit_rmse(recovery, threshold=0.10)
    coverage = ep.audit_coverage(recovery, minimum=0.90)
    width = ep.audit_interval_width(recovery, maximum=0.50)
    identifiability = ep.audit_identifiability(recovery)
    assert bias["pass"].all()
    assert rmse["pass"].all()
    assert coverage["pass"].all()
    assert width["pass"].all()
    assert identifiability["pass"].all()

    comparison = ep.compare_validation_engines(recovery)
    assert comparison.attrs["eyeprocess_class"] == "eye_validation_engine_comparison"
    assert set(comparison.engine) == {"reference", "alternate"}
    assert comparison.rank_within_parameter.min() == 1

    recommended = ep.recommended_validation_replications(
        target_mcse=0.10,
        metric="coverage",
        anticipated_probability=0.95,
        minimum=10,
    )
    assert recommended == 10


def test_posterior_sbc_contract_executes_self_consistency_replications():
    contract = ep.posterior_sbc_contract(
        lambda replicate, observed: {
            "truth": {"mu": float(np.mean(observed))},
            "draws": pd.DataFrame(
                {"mu": np.linspace(-1.0, 1.0, 101) + float(np.mean(observed)) + 0.001 * replicate}
            ),
        }
    )
    sbc = ep.run_posterior_sbc(np.array([-0.2, 0.0, 0.2]), contract, replications=4, seed=9)
    assert sbc.eyeprocess_class == "eye_posterior_sbc"
    assert len(sbc.ranks) == 4
    assert sbc.failures.empty
    assert sbc.ranks.normalized_rank.between(0, 1).all()


def test_external_and_leave_group_validations_execute_all_folds():
    data = _process_data().copy()
    fitter = lambda train: {"mean": float(train.response.mean())}
    predictor = lambda fit, test: np.repeat(fit["mean"], len(test))
    scorer = lambda test, pred: pd.DataFrame(
        {"brier": [float(np.mean((test.response.to_numpy(float) - np.asarray(pred, float)) ** 2))]}
    )

    external = ep.external_validate_irt(data.iloc[:40], data.iloc[40:], fitter, predictor, scorer)
    session = ep.leave_session_out_validation(data, "session", fitter, predictor, scorer)
    site = ep.leave_site_out_validation(data, "site", fitter, predictor, scorer)
    item = ep.leave_item_out_validation(data, "item_id", fitter, predictor, scorer)
    assert external.attrs["eyeprocess_class"] == "eye_external_irt_validation"
    assert not external.failed.any()
    assert len(session) == data.session.nunique() and not session.failed.any()
    assert len(site) == data.site.nunique() and not site.failed.any()
    assert len(item) == data.item_id.nunique() and not item.failed.any()


def test_validation_stress_families_preserve_scenarios_and_replications():
    runner = lambda scenario, replicate: pd.DataFrame(
        {
            "metric": [float(replicate)],
            "scenario_echo": [str(scenario["scenario"].iloc[0])],
        }
    )
    generic = ep.stress_test_misspecification(
        pd.DataFrame({"scenario": ["baseline", "misspecified"], "strength": [0.0, 0.5]}),
        runner,
        replications=2,
        seed=1,
    )
    latent = ep.stress_test_latent_distribution(runner, replications=1, seed=2)
    speed = ep.stress_test_speededness(runner, proportions=[0.0, 0.25], replications=1, seed=3)
    missing = ep.stress_test_missingness(
        runner,
        mechanisms=["MCAR", "MNAR_omission"],
        rates=[0.10],
        replications=1,
        seed=4,
    )
    preprocessing = ep.stress_test_preprocessing(runner, ["raw", "filtered"], replications=1, seed=5)

    assert len(generic) == 4 and not generic.failed.any()
    assert len(latent) == 6 and not latent.failed.any()
    assert len(speed) == 2 and not speed.failed.any()
    assert len(missing) == 2 and not missing.failed.any()
    assert len(preprocessing) == 2 and not preprocessing.failed.any()
