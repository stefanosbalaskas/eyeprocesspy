from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy.context_structure_08 as cs
import eyeprocesspy.plots_irt_08 as p


class Box(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _box(class_name, **kwargs):
    return Box(eyeprocess_class=class_name, **kwargs)


def _close(ax):
    plt.close(ax.figure)


def test_private_axis_and_empty_helpers_with_supplied_axis():
    fig, ax = plt.subplots()
    assert p._ax(ax) is ax
    out = p._empty("Title", "Message", ax)
    assert out is ax
    assert out.get_title() == "Title"
    plt.close(fig)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("plot_eye_gated_process_model", {}),
        ("plot_eye_mixture_irt_process", {}),
        ("plot_eye_latent_process_alignment", {}),
        ("plot_eye_nonparametric_rasch_audit", {}),
        ("plot_eye_item_reduction_sensitivity", {}),
        ("plot_eye_biometric_imputation_sensitivity", {}),
        ("plot_eye_process_rasch_tree", {}),
        ("plot_eye_bayesian_process_dashboard", {}),
        ("plot_eye_gaze_anchored_3pl_audit", {}),
        ("plot_eye_multiblock_process_map", {}),
        ("plot_eye_process_profile_mixture", {}),
        ("plot_eye_process_external_validity", {}),
        ("plot_eye_item_parameter_seed", {}),
        ("plot_eye_candidate_item_bank_audit", {}),
        ("plot_eye_visual_context_irt", {}),
    ],
)
def test_plot_class_guards(name, kwargs):
    with pytest.raises(p.EyeProcessValidationError):
        getattr(p, name)(Box(), **kwargs)


def test_simple_placeholder_plots_and_latent_alignment_branches():
    ax = p.plot_eye_gated_process_model(
        _box("eye_gated_process_model", id="g1", status="prepare_only")
    )
    assert "Gated frontier model" in ax.get_title()
    _close(ax)

    ax = p.plot_eye_mixture_irt_process(
        _box("eye_mixture_irt_process", n_classes=3)
    )
    assert "Mixture IRT" in ax.get_title()
    _close(ax)

    ax = p.plot_eye_nonparametric_rasch_audit(
        _box("eye_nonparametric_rasch_audit"),
        method="PCM",
    )
    assert "Nonparametric Rasch" in ax.get_title()
    _close(ax)

    ax = p.plot_eye_process_rasch_tree(_box("eye_process_rasch_tree"))
    assert "Rasch tree" in ax.get_title()
    _close(ax)

    empty = _box(
        "eye_latent_process_alignment",
        summary=pd.DataFrame(),
        process_features=["dwell"],
    )
    ax = p.plot_eye_latent_process_alignment(empty)
    assert "No process summaries" in ax.texts[0].get_text()
    _close(ax)

    no_vars = _box(
        "eye_latent_process_alignment",
        summary=pd.DataFrame({"class": [1], "other": [0.5]}),
        process_features=["dwell"],
    )
    ax = p.plot_eye_latent_process_alignment(no_vars)
    assert "No process summaries" in ax.texts[0].get_text()
    _close(ax)

    rich = _box(
        "eye_latent_process_alignment",
        summary=pd.DataFrame(
            {
                "class": [1, 2],
                "dwell": [0.2, 0.8],
                "revisit": [0.7, 0.3],
            }
        ),
        process_features=["dwell", "revisit", "missing"],
    )
    ax = p.plot_eye_latent_process_alignment(rich)
    assert len(ax.lines) == 2
    assert len(ax.gp3_data) == 2
    _close(ax)


def test_item_reduction_and_biometric_imputation_branches():
    none = _box("eye_item_reduction_sensitivity", eliminated_items=[])
    ax = p.plot_eye_item_reduction_sensitivity(none)
    assert "No items" in ax.texts[0].get_text()
    _close(ax)

    some = _box(
        "eye_item_reduction_sensitivity",
        eliminated_items=["i1", "i2"],
    )
    ax = p.plot_eye_item_reduction_sensitivity(some)
    assert len(ax.collections) == 1
    assert ax.gp3_data == ["i1", "i2"]
    _close(ax)

    missingness = pd.DataFrame(
        {"variable": ["pupil", "eda"], "missing_prop": [0.1, 0.3]}
    )
    ax = p.plot_eye_biometric_imputation_sensitivity(
        _box("eye_biometric_imputation_sensitivity", missingness=missingness)
    )
    assert len(ax.patches) == 2
    assert ax.gp3_data.equals(missingness)
    _close(ax)


def test_bayesian_dashboard_loo_and_posterior_dispatch_branches():
    bad_loo = _box(
        "eye_bayesian_process_dashboard",
        loo_table=pd.DataFrame({"model": ["m1"], "other": [1]}),
        posterior=pd.DataFrame(),
    )
    ax = p.plot_eye_bayesian_process_dashboard(bad_loo, type="loo")
    assert "LOO summary unavailable" in ax.texts[0].get_text()
    _close(ax)

    nonnumeric_loo = _box(
        "eye_bayesian_process_dashboard",
        loo_table=pd.DataFrame({"model": ["m1"], "elpd_loo": ["bad"]}),
        posterior=pd.DataFrame(),
    )
    ax = p.plot_eye_bayesian_process_dashboard(nonnumeric_loo, type="loo")
    assert "LOO summary unavailable" in ax.texts[0].get_text()
    _close(ax)

    good_loo = _box(
        "eye_bayesian_process_dashboard",
        loo_table=pd.DataFrame(
            {"model": ["m1", "m2"], "elpd_loo": [-10.0, -8.0]}
        ),
        posterior=pd.DataFrame(),
    )
    ax = p.plot_eye_bayesian_process_dashboard(good_loo, type="loo")
    assert len(ax.patches) == 2
    _close(ax)

    empty_post = _box(
        "eye_bayesian_process_dashboard",
        loo_table=pd.DataFrame(),
        posterior=pd.DataFrame(),
    )
    ax = p.plot_eye_bayesian_process_dashboard(empty_post, type="rhat")
    assert "Posterior summary unavailable" in ax.texts[0].get_text()
    _close(ax)

    missing_col = _box(
        "eye_bayesian_process_dashboard",
        loo_table=pd.DataFrame(),
        posterior=pd.DataFrame({"model": ["m1"], "other": [1.0]}),
    )
    ax = p.plot_eye_bayesian_process_dashboard(missing_col, type="rhat")
    assert "rhat unavailable" in ax.texts[0].get_text()
    _close(ax)

    posterior = pd.DataFrame(
        {
            "model": ["m1", "m1", "m2", "m2"],
            "rhat": [1.0, 1.01, 1.0, 1.005],
            "ess_bulk": [500, 600, 450, 550],
        }
    )
    good_post = _box(
        "eye_bayesian_process_dashboard",
        loo_table=pd.DataFrame(),
        posterior=posterior,
    )
    ax = p.plot_eye_bayesian_process_dashboard(good_post, type="rhat")
    assert len(ax.lines) >= 1
    assert "R-hat" in ax.get_title()
    _close(ax)

    ax = p.plot_eye_bayesian_process_dashboard(good_post, type="ess_bulk")
    assert len(ax.lines) >= 1
    assert "ESS" in ax.get_title()
    _close(ax)


def test_gaze_anchored_3pl_all_dispatch_and_guard_paths():
    base = pd.DataFrame(
        {
            "item_id": ["i1", "i2"],
            "lower_asymptote": [0.1, 0.2],
            "a": [1.2, 0.8],
            "b": [-0.3, 0.5],
            "dwell": [0.4, 0.7],
        }
    )
    x = _box(
        "eye_gaze_anchored_3pl_audit",
        item_parameters=base,
        process_features=["dwell", "absent"],
    )

    ax = p.plot_eye_gaze_anchored_3pl_audit(x, type="lower_asymptote")
    assert len(ax.patches) == 2
    _close(ax)

    no_lower = _box(
        "eye_gaze_anchored_3pl_audit",
        item_parameters=base.drop(columns=["lower_asymptote"]),
        process_features=["dwell"],
    )
    ax = p.plot_eye_gaze_anchored_3pl_audit(no_lower, type="lower_asymptote")
    assert "unavailable" in ax.texts[0].get_text()
    _close(ax)

    ax = p.plot_eye_gaze_anchored_3pl_audit(
        x,
        type="difficulty_discrimination",
    )
    assert len(ax.collections) == 1
    _close(ax)

    no_ab = _box(
        "eye_gaze_anchored_3pl_audit",
        item_parameters=base.drop(columns=["a"]),
        process_features=["dwell"],
    )
    ax = p.plot_eye_gaze_anchored_3pl_audit(
        no_ab,
        type="difficulty_discrimination",
    )
    assert "unavailable" in ax.texts[0].get_text()
    _close(ax)

    ax = p.plot_eye_gaze_anchored_3pl_audit(x, type="process")
    assert ax.get_xlabel() == "dwell"
    _close(ax)

    no_features = _box(
        "eye_gaze_anchored_3pl_audit",
        item_parameters=base,
        process_features=["not_present"],
    )
    ax = p.plot_eye_gaze_anchored_3pl_audit(no_features, type="process")
    assert "No process features" in ax.texts[0].get_text()
    _close(ax)

    with pytest.raises(p.EyeProcessValidationError, match="Unknown process feature"):
        p.plot_eye_gaze_anchored_3pl_audit(
            x,
            type="process",
            feature="absent",
        )


def test_multiblock_individual_variable_and_block_branches():
    rich = _box(
        "eye_multiblock_process_map",
        engine="pca",
        person_coordinates=pd.DataFrame(
            {"dim1": [-1.0, 1.0], "dim2": [0.5, -0.5], "label": ["a", "b"]}
        ),
        variable_coordinates=pd.DataFrame(
            {
                "variable": ["v1", "v2"],
                "dim1": [0.3, -0.2],
                "dim2": [0.7, 0.1],
            }
        ),
        block_coordinates=pd.DataFrame(
            {"block": ["gaze", "pupil"], "dim1": [0.2, -0.4], "dim2": [0.1, 0.5]}
        ),
    )
    ax = p.plot_eye_multiblock_process_map(rich, type="individuals")
    assert len(ax.collections) == 1
    _close(ax)

    ax = p.plot_eye_multiblock_process_map(rich, type="variables")
    assert len(ax.collections) == 1
    assert len(ax.texts) == 2
    _close(ax)

    ax = p.plot_eye_multiblock_process_map(rich, type="blocks")
    assert len(ax.patches) == 2
    _close(ax)

    sparse = _box(
        "eye_multiblock_process_map",
        engine="pca",
        person_coordinates=pd.DataFrame({"dim1": [1.0]}),
        variable_coordinates=pd.DataFrame({"variable": ["v1"], "dim1": [0.1]}),
        block_coordinates=pd.DataFrame(columns=["block"]),
    )
    ax = p.plot_eye_multiblock_process_map(sparse, type="individuals")
    assert "Fewer than two" in ax.texts[0].get_text()
    _close(ax)

    ax = p.plot_eye_multiblock_process_map(sparse, type="variables")
    assert "Fewer than two" in ax.texts[0].get_text()
    _close(ax)

    ax = p.plot_eye_multiblock_process_map(sparse, type="blocks")
    assert "No numeric" in ax.texts[0].get_text()
    _close(ax)


def test_process_profile_posterior_profiles_parallel_and_scatter_branches():
    assignment = pd.DataFrame(
        {
            "profile": [1, 2],
            "profile_probability_1": [0.8, 0.2],
            "profile_probability_2": [0.2, 0.8],
        }
    )
    x = _box(
        "eye_process_profile_mixture",
        assignment=assignment,
        summary=pd.DataFrame(
            {"profile": [1, 2], "dwell": [0.3, 0.8], "revisit": [0.7, 0.2]}
        ),
        variables=["dwell", "revisit"],
        scaled_data=np.array([[0.1, 0.2], [0.8, 0.7]]),
        status="ok",
    )
    ax = p.plot_eye_process_profile_mixture(x, type="posterior")
    assert len(ax.lines) == 2
    _close(ax)

    no_probs = _box(
        "eye_process_profile_mixture",
        assignment=pd.DataFrame({"profile": [1, 2]}),
        summary=x["summary"],
        variables=x["variables"],
        scaled_data=x["scaled_data"],
        status="ok",
    )
    ax = p.plot_eye_process_profile_mixture(no_probs, type="posterior")
    assert "No posterior" in ax.texts[0].get_text()
    _close(ax)

    ax = p.plot_eye_process_profile_mixture(x, type="parallel")
    assert len(ax.lines) == 2
    _close(ax)

    ax = p.plot_eye_process_profile_mixture(x, type="scatter")
    assert len(ax.collections) == 2
    _close(ax)

    one_feature = _box(
        "eye_process_profile_mixture",
        assignment=pd.DataFrame({"profile": [1, 2]}),
        summary=x["summary"],
        variables=x["variables"],
        scaled_data=np.array([[0.1], [0.8]]),
        status="ok",
    )
    ax = p.plot_eye_process_profile_mixture(one_feature, type="scatter")
    assert "Fewer than two features" in ax.texts[0].get_text()
    _close(ax)


def test_external_validity_all_plot_dispatch_paths():
    associations = pd.DataFrame(
        {"predictor": ["dwell", "pupil"], "correlation": [0.3, -0.2]}
    )
    x = _box(
        "eye_process_external_validity",
        associations=associations,
        baseline_model={"r_squared": 0.2},
        full_model={
            "r_squared": 0.4,
            "training_index": [0, 1, 2],
            "fitted": [1.1, 1.9, 3.1],
            "residuals": [-0.1, 0.1, -0.1],
        },
        incremental_r2=0.2,
        data=pd.DataFrame({"criterion": [1.0, 2.0, 3.0]}),
        criterion="criterion",
    )

    ax = p.plot_eye_process_external_validity(x, type="associations")
    assert len(ax.collections) == 1
    _close(ax)

    ax = p.plot_eye_process_external_validity(x, type="incremental")
    assert len(ax.patches) == 2
    _close(ax)

    ax = p.plot_eye_process_external_validity(x, type="observed_fitted")
    assert len(ax.collections) == 1
    _close(ax)

    ax = p.plot_eye_process_external_validity(x, type="residuals")
    assert len(ax.collections) == 1
    _close(ax)


def test_item_parameter_seed_training_and_candidate_paths(monkeypatch):
    training = pd.DataFrame(
        {"difficulty": [-1.0, 1.0], "discrimination": [0.8, 1.2]}
    )
    x = _box(
        "eye_item_parameter_seed",
        training_data=training,
        difficulty="difficulty",
        discrimination="discrimination",
    )
    ax = p.plot_eye_item_parameter_seed(x)
    assert len(ax.collections) == 1
    _close(ax)

    predicted = pd.DataFrame(
        {
            "predicted_pre_pilot_difficulty": [-0.5, 0.5],
            "predicted_pre_pilot_discrimination": [0.9, 1.1],
        }
    )
    monkeypatch.setattr(
        cs,
        "predict_item_parameter_priors",
        lambda model, candidate: predicted.copy(),
    )
    ax = p.plot_eye_item_parameter_seed(
        x,
        candidate_data=pd.DataFrame({"candidate": ["c1", "c2"]}),
    )
    assert len(ax.collections) == 1
    assert ax.gp3_data.equals(predicted)
    _close(ax)


def test_candidate_bank_normal_review_and_visual_context_branches():
    both = pd.DataFrame(
        {
            "predicted_pre_pilot_difficulty": [-0.5, 0.5],
            "predicted_pre_pilot_discrimination": [0.9, 1.1],
            "review_required": [False, True],
        }
    )
    ax = p.plot_eye_candidate_item_bank_audit(
        _box("eye_candidate_item_bank_audit", table=both)
    )
    assert len(ax.collections) == 2
    _close(ax)

    only_review = both.loc[[1]].copy()
    ax = p.plot_eye_candidate_item_bank_audit(
        _box("eye_candidate_item_bank_audit", table=only_review)
    )
    assert len(ax.collections) == 1
    _close(ax)

    only_normal = both.loc[[0]].copy()
    ax = p.plot_eye_candidate_item_bank_audit(
        _box("eye_candidate_item_bank_audit", table=only_normal)
    )
    assert len(ax.collections) == 1
    _close(ax)

    registry = {
        "mapping": pd.DataFrame(
            {"visual_context_id": ["A", "A", "B"], "item_id": ["i1", "i2", "i3"]}
        )
    }
    visual = _box("eye_visual_context_irt", registry=registry)
    ax = p.plot_eye_visual_context_irt(visual, type="context_registry")
    assert len(ax.patches) == 2
    _close(ax)

    ax = p.plot_eye_visual_context_irt(visual, type="model")
    assert "Exact mirt coefficients" in ax.texts[0].get_text()
    _close(ax)
