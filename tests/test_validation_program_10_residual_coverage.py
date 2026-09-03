from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy.validation_program_10 as vp
from eyeprocesspy.exceptions import EyeProcessValidationError


class _BadMapping(dict):
    def __getitem__(self, key):
        raise RuntimeError("broken mapping")


def _spec(**kwargs):
    defaults = {
        "replications": 1,
        "confidence": 0.95,
        "max_abs_bias": 10.0,
        "min_coverage": 0.0,
        "max_failure_rate": 1.0,
    }
    defaults.update(kwargs)
    return vp.model_validation_spec(**defaults)


def _simulator(mode="ok"):
    if mode == "simulation_error":
        raise RuntimeError("simulation failed")
    return {"mode": mode, "truth": {"theta": 1.0}}


def _fitter(simulation):
    if simulation["mode"] == "fit_error":
        raise RuntimeError("fit failed")
    return {"theta": 1.0}


def _extractor(fit):
    return {"theta": fit["theta"]}


def _truth(simulation):
    return simulation["truth"]


def test_string_vendor_spec_and_corpus_validation_edges():
    with pytest.raises(EyeProcessValidationError, match="non-empty"):
        vp._as_nonempty_strings(7, "values")
    with pytest.raises(EyeProcessValidationError, match="non-empty"):
        vp._as_nonempty_strings("", "values")
    with pytest.raises(EyeProcessValidationError, match="non-empty"):
        vp._as_nonempty_strings([np.nan], "values")

    assert vp.vendor_validation_spec(required_vendors="TOBII").required_vendors == ("tobii",)

    with pytest.raises(EyeProcessValidationError, match="positive integer"):
        vp.vendor_validation_spec(min_cases_per_vendor="bad")
    with pytest.raises(EyeProcessValidationError, match="positive integer"):
        vp.vendor_validation_spec(min_cases_per_vendor=1.5)
    with pytest.raises(EyeProcessValidationError, match="between zero and one"):
        vp.vendor_validation_spec(min_pass_rate="bad")
    with pytest.raises(EyeProcessValidationError, match="between zero and one"):
        vp.vendor_validation_spec(min_pass_rate=np.nan)

    frame = pd.DataFrame({"vendor": ["tobii"], "status": ["pass"]})
    copied, manifest = vp._extract_corpus_frames(frame)
    assert copied.equals(frame)
    assert copied is not frame
    assert manifest is None

    wrapped = SimpleNamespace(summary=frame, manifest=frame.copy())
    copied, manifest = vp._extract_corpus_frames(wrapped)
    assert copied.equals(frame)
    assert manifest is not None and manifest.equals(frame)

    with pytest.raises(TypeError, match="summary"):
        vp._extract_corpus_frames(object())
    with pytest.raises(TypeError, match="manifest"):
        vp._extract_corpus_frames({"summary": frame, "manifest": "bad"})

    assert not vp._complete_text(pd.Series([], dtype="string"))
    assert not vp._complete_text(pd.Series(["ok", " "], dtype="string"))
    assert vp._complete_text(pd.Series(["ok", "yes"], dtype="string"))

    missing_flags = vp._logical_flags(frame, "independent_source")
    assert missing_flags.isna().all()
    present_flags = vp._logical_flags(
        pd.DataFrame({"independent_source": [True, False]}),
        "independent_source",
    )
    assert present_flags.tolist() == [True, False]


def test_vendor_audit_required_optional_extra_and_failure_paths(tmp_path):
    with pytest.raises(TypeError, match="spec"):
        vp.audit_vendor_validation(
            pd.DataFrame({"vendor": ["x"], "status": ["pass"]}),
            spec={},
        )
    with pytest.raises(EyeProcessValidationError, match="missing required"):
        vp.audit_vendor_validation(pd.DataFrame({"vendor": ["x"]}))

    data = pd.DataFrame(
        {
            "vendor": ["CUSTOM", None, ""],
            "status": ["pass", "pass", "pass"],
        }
    )
    spec = vp.vendor_validation_spec(
        required_vendors=["required"],
        min_cases_per_vendor=1,
        min_pass_rate=1.0,
        require_versions=False,
        require_devices=False,
        require_independent_sources=False,
        require_licence_reviewed=False,
    )
    audit = vp.audit_vendor_validation(data, spec)
    assert set(audit["vendor"]) == {"required", "custom"}
    assert audit.loc[audit["vendor"].eq("required"), "status"].iloc[0] == "fail"
    custom = audit.loc[audit["vendor"].eq("custom")].iloc[0]
    assert custom["status"] == "pass"
    assert bool(custom["versions_complete"])
    assert bool(custom["devices_complete"])
    assert bool(custom["independent_sources_complete"])
    assert bool(custom["licences_reviewed"])

    incomplete = pd.DataFrame(
        {
            "vendor": ["x"],
            "status": ["fail"],
            "software_version": [""],
            "device_model": [pd.NA],
        }
    )
    strict = vp.vendor_validation_spec(
        required_vendors=["x"],
        min_cases_per_vendor=1,
        min_pass_rate=1.0,
    )
    failed = vp.audit_vendor_validation(incomplete, strict).iloc[0]
    assert failed["status"] == "fail"
    assert not bool(failed["versions_complete"])
    assert not bool(failed["devices_complete"])
    assert failed["independent_cases"] == 0
    assert failed["licence_reviewed_cases"] == 0

    summary = pd.DataFrame(
        {"case_id": ["a"], "vendor": ["x"], "status": ["pass"]}
    )
    manifest = pd.DataFrame(
        {
            "case_id": ["a"],
            "independent_source": [True],
            "licence_reviewed": [True],
        }
    )
    no_merge = summary.assign(independent_source=True, licence_reviewed=True)
    out = vp.audit_vendor_validation(
        {"summary": no_merge, "manifest": manifest.iloc[0:0]},
        vp.vendor_validation_spec(
            required_vendors=["x"],
            min_cases_per_vendor=1,
            require_versions=False,
            require_devices=False,
        ),
    )
    assert out["status"].iloc[0] == "pass"

    with pytest.raises(EyeProcessValidationError, match="eye_vendor_validation"):
        vp.write_vendor_validation_report(pd.DataFrame(), tmp_path / "bad.md")


def test_model_spec_grid_and_conversion_edges():
    with pytest.raises(EyeProcessValidationError, match="positive integer"):
        vp.model_validation_spec(replications="bad")
    with pytest.raises(EyeProcessValidationError, match="positive integer"):
        vp.model_validation_spec(replications=1.5)

    for kwargs in (
        {"confidence": "bad"},
        {"confidence": np.inf},
        {"min_coverage": -0.01},
        {"max_failure_rate": 1.01},
    ):
        with pytest.raises(EyeProcessValidationError, match="between zero and one"):
            vp.model_validation_spec(**kwargs)

    with pytest.raises(EyeProcessValidationError, match="finite non-negative"):
        vp.model_validation_spec(max_abs_bias="bad")
    with pytest.raises(EyeProcessValidationError, match="finite non-negative"):
        vp.model_validation_spec(max_abs_bias=np.inf)

    assert vp._grid_rows(None) == [{}]
    assert vp._grid_rows({}) == [{}]

    rows = vp._grid_rows({"model": "a", "n": [1, 2]})
    assert rows == [{"model": "a", "n": 1}, {"model": "a", "n": 2}]

    df_rows = vp._grid_rows(pd.DataFrame({"x": [1], "y": ["a"]}))
    assert df_rows == [{"x": 1, "y": "a"}]
    assert vp._grid_rows([{"x": 1}, {"x": 2}]) == [{"x": 1}, {"x": 2}]

    with pytest.raises(EyeProcessValidationError, match="at least one scenario"):
        vp._grid_rows(pd.DataFrame())
    with pytest.raises(EyeProcessValidationError, match="at least one scenario"):
        vp._grid_rows({"x": []})
    with pytest.raises(EyeProcessValidationError, match="at least one scenario"):
        vp._grid_rows([])
    with pytest.raises(EyeProcessValidationError, match="DataFrame"):
        vp._grid_rows([1, 2])
    with pytest.raises(EyeProcessValidationError, match="DataFrame"):
        vp._grid_rows(3.14)


def test_estimate_truth_failure_and_bind_helpers():
    failure = vp._failure_row(
        scenario=2,
        replication=3,
        parameter=".fit",
        error=RuntimeError("boom"),
        scenario_values={"mode": "x"},
    )
    assert failure.loc[0, "error"] == "boom"
    assert failure.loc[0, "mode"] == "x"
    assert not bool(failure.loc[0, "converged"])

    frame = pd.DataFrame({"parameter": ["a"], "estimate": [1.0]})
    copied = vp._estimate_frame(frame)
    assert copied is not frame and copied.equals(frame)

    series = pd.Series([1.0, 2.0], index=["a", "b"])
    assert vp._estimate_frame(series)["parameter"].tolist() == ["a", "b"]
    duplicate = pd.Series([1.0, 2.0], index=["a", "a"])
    assert vp._estimate_frame(duplicate) is None
    assert vp._estimate_frame({"a": 1.0}).loc[0, "estimate"] == 1.0
    assert vp._estimate_frame(_BadMapping(a=1.0)) is None
    assert vp._estimate_frame(1.0) is None

    truth_series = pd.Series([1.0, 2.0], index=["a", "b"])
    assert vp._truth_mapping(truth_series) == {"a": 1.0, "b": 2.0}
    duplicate_truth = pd.Series([1.0, 2.0], index=["a", "a"])
    assert vp._truth_mapping(duplicate_truth) is None
    assert vp._truth_mapping({"a": "2"}) == {"a": 2.0}
    assert vp._truth_mapping({"a": "bad"}) is None
    assert vp._truth_mapping([1.0]) is None

    assert vp._bind_rows([]).empty
    bound = vp._bind_rows(
        [
            pd.DataFrame({"a": [1], "b": [2]}),
            pd.DataFrame({"b": [3], "c": [4]}),
        ]
    )
    assert list(bound.columns) == ["a", "b", "c"]
    assert pd.isna(bound.loc[0, "c"])
    assert pd.isna(bound.loc[1, "a"])


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        ("simulation_error", "simulation failed"),
        ("fit_error", "fit failed"),
    ],
)
def test_run_model_validation_reraises_simulation_and_fit(stage, message):
    with pytest.raises(RuntimeError, match=message):
        vp.run_model_validation(
            _simulator,
            _fitter,
            _extractor,
            _truth,
            grid={"mode": stage},
            spec=_spec(),
            continue_on_error=False,
        )


def test_run_model_validation_argument_and_spec_errors():
    with pytest.raises(EyeProcessValidationError, match="must be callable"):
        vp.run_model_validation(None, _fitter, _extractor, _truth, spec=_spec())

    with pytest.raises(TypeError, match="spec"):
        vp.run_model_validation(
            _simulator,
            _fitter,
            _extractor,
            _truth,
            spec={},
        )


def test_run_model_validation_extractor_invalid_and_reraise_paths():
    with pytest.raises(RuntimeError, match="extract failed"):
        vp.run_model_validation(
            _simulator,
            _fitter,
            lambda fit: (_ for _ in ()).throw(RuntimeError("extract failed")),
            _truth,
            spec=_spec(),
            continue_on_error=False,
        )

    recorded = vp.run_model_validation(
        _simulator,
        _fitter,
        lambda fit: None,
        _truth,
        spec=_spec(),
    )
    assert recorded["runs"]["parameter"].iloc[0] == ".extract"
    assert "Extractor must return" in str(recorded["runs"]["error"].iloc[0])

    with pytest.raises(EyeProcessValidationError, match="Extractor must return"):
        vp.run_model_validation(
            _simulator,
            _fitter,
            lambda fit: pd.DataFrame({"parameter": ["theta"]}),
            _truth,
            spec=_spec(),
            continue_on_error=False,
        )


def test_run_model_validation_truth_invalid_and_reraise_paths():
    recorded = vp.run_model_validation(
        _simulator,
        _fitter,
        _extractor,
        lambda simulation: ["not", "named"],
        spec=_spec(),
    )
    assert recorded["runs"]["parameter"].iloc[0] == ".truth"
    assert "Truth extractor must return named values" in str(
        recorded["runs"]["error"].iloc[0]
    )

    with pytest.raises(EyeProcessValidationError, match="named values"):
        vp.run_model_validation(
            _simulator,
            _fitter,
            _extractor,
            lambda simulation: {"theta": "not-numeric"},
            spec=_spec(),
            continue_on_error=False,
        )


def test_run_model_validation_series_success_repr_and_cartesian_grid():
    result = vp.run_model_validation(
        lambda group="a", n=1: {
            "group": group,
            "n": n,
            "truth": pd.Series([1.0], index=["theta"]),
        },
        lambda simulation: pd.Series([1.0], index=["theta"]),
        lambda fit: fit,
        lambda simulation: simulation["truth"],
        grid={"group": ["a", "b"], "n": 1},
        spec=_spec(replications=2),
        seed=42,
    )
    runs = result["runs"]
    assert len(runs) == 4
    assert runs["lower"].isna().all()
    assert runs["upper"].isna().all()
    assert runs["covered"].isna().all()
    assert "<eye_model_validation runs=4" in repr(result)


def test_run_model_validation_post_bind_missing_column_fallbacks(monkeypatch):
    monkeypatch.setattr(
        vp,
        "_bind_rows",
        lambda rows: pd.DataFrame({"parameter": ["theta"]}),
    )
    result = vp.run_model_validation(
        _simulator,
        _fitter,
        _extractor,
        _truth,
        spec=_spec(),
    )
    runs = result["runs"]
    for column in ("estimate", "truth", "lower", "upper"):
        assert column in runs.columns
        assert runs[column].isna().all()
    assert not runs["converged"].any()
    assert runs["covered"].isna().all()


def _summary_frame(
    *,
    estimate=1.0,
    truth=1.0,
    converged=True,
    lower=np.nan,
    upper=np.nan,
    covered=pd.NA,
):
    bias = estimate - truth if np.isfinite(estimate) and np.isfinite(truth) else np.nan
    return pd.DataFrame(
        {
            "replication": [1],
            "parameter": ["theta"],
            "estimate": [estimate],
            "truth": [truth],
            "lower": [lower],
            "upper": [upper],
            "converged": [converged],
            "error": [pd.NA],
            "bias": [bias],
            "squared_error": [bias**2 if np.isfinite(bias) else np.nan],
            "covered": pd.Series([covered], dtype="boolean"),
        }
    )


def test_model_validation_summary_input_empty_and_single_key_paths():
    with pytest.raises(EyeProcessValidationError, match="eye_model_validation"):
        vp.model_validation_summary({})
    with pytest.raises(EyeProcessValidationError, match="eye_model_validation"):
        vp.model_validation_summary({"spec": _spec(), "runs": "bad"})

    empty = vp.model_validation_summary(
        {"spec": _spec(), "runs": pd.DataFrame()}
    )
    assert empty.empty

    summary = vp.model_validation_summary(
        {"spec": _spec(), "runs": _summary_frame()}
    )
    assert summary["successful"].iloc[0] == 1
    assert pd.isna(summary["coverage"].iloc[0])
    assert summary["status"].iloc[0] == "pass"


@pytest.mark.parametrize(
    ("frame", "spec_kwargs"),
    [
        (_summary_frame(converged=False), {}),
        (_summary_frame(estimate=np.nan), {}),
        (_summary_frame(estimate=2.0, truth=1.0), {"max_abs_bias": 0.1}),
        (
            _summary_frame(lower=2.0, upper=3.0, covered=False),
            {"min_coverage": 0.9},
        ),
        (_summary_frame(converged=False), {"max_failure_rate": 0.0}),
    ],
)
def test_model_validation_summary_failure_gate_branches(frame, spec_kwargs):
    summary = vp.model_validation_summary(
        {"spec": _spec(**spec_kwargs), "runs": frame}
    )
    assert summary["status"].iloc[0] == "fail"


def test_model_validation_summary_coverage_pass_branch():
    frame = _summary_frame(lower=0.5, upper=1.5, covered=True)
    summary = vp.model_validation_summary(
        {"spec": _spec(min_coverage=0.9), "runs": frame}
    )
    assert summary["coverage"].iloc[0] == 1.0
    assert summary["status"].iloc[0] == "pass"
