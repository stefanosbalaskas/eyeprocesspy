from __future__ import annotations

import inspect

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.exceptions import EyeProcessBackendError, EyeProcessValidationError
from eyeprocesspy.irt import EyeResult


FROZEN_ADAPTER_EXPORTS = [
    "eyeprocess_api_version", "object_schema", "validate_model_object", "upgrade_eyeprocess_model", "eyeprocess_deprecation",
    "external_model_engines", "engine_adapter_status", "fit_external_engine", "validate_engine_adapter", "compare_engine_adapters",
    "fit_mirt_adapter", "fit_tam_adapter", "fit_brms_adapter", "fit_lnirt_adapter", "fit_traminer_adapter", "fit_seqhmm_adapter",
    "fit_gdina_adapter", "fit_openmx_adapter", "fit_diffirt_engine_adapter", "fit_eyetrackingr_adapter", "fit_pupillometryr_adapter",
    "as_procdata_sequence", "as_traminer_sequence", "as_seqhmm_data", "fit_diffirt_adapter", "fit_openmx_process_model",
    "compare_model_engines",
]


def test_adapter_exports_resolve_and_public_signatures_are_stable():
    for name in FROZEN_ADAPTER_EXPORTS:
        assert callable(getattr(ep, name))
    assert list(inspect.signature(ep.fit_mirt_adapter).parameters)[:3] == ["data", "model", "purpose"]
    assert list(inspect.signature(ep.fit_external_engine).parameters)[:4] == ["engine", "data", "specification", "purpose"]
    # The final R/028 definition overrides the earlier R/020 fit_gdina_adapter.
    assert list(inspect.signature(ep.fit_gdina_adapter).parameters)[:4] == ["data", "Q", "model", "purpose"]


def test_api_and_model_contract_helpers():
    v = ep.eyeprocess_api_version()
    assert str(v) == "0.5.0"
    assert v.object_schema == "2.0.0"
    schema = ep.object_schema("eyeprocess_model")
    assert schema["version"] == "1.0.0"
    assert "scientific validity" in schema["invariant"]

    legacy = EyeResult({"spec": {"engine": "reference"}, "model": {"coef": [1.0]}}, eyeprocess_class="eyeprocess_model")
    upgraded = ep.upgrade_eyeprocess_model(legacy)
    assert upgraded["specification"] == legacy["spec"]
    assert upgraded["fit"] == legacy["model"]
    assert upgraded["model_contract_version"] == "1.0.0"
    validation = ep.validate_model_object(upgraded)
    assert validation["valid"] is True


def test_external_engine_registry_and_explicit_not_available_contracts():
    registry = ep.external_model_engines()
    assert list(registry.columns) == ["engine", "package", "domain", "available"]
    assert set(["mirt", "TAM", "brms", "LNIRT", "GDINA", "OpenMx", "diffIRT", "TraMineR", "seqHMM"]).issubset(set(registry.engine))
    assert registry.available.eq(False).all()

    status = ep.engine_adapter_status("MIRT")
    assert status.iloc[0].engine == "mirt"
    assert "without silently selecting" in status.iloc[0].contract
    with pytest.raises(EyeProcessValidationError, match="Unknown engine"):
        ep.engine_adapter_status("mystery")

    result = ep.fit_external_engine("mirt", np.array([[1, 0], [0, 1]]), purpose="contract test")
    assert result.eyeprocess_class == "eye_engine_adapter_result"
    assert result.status == "not_available"
    assert result.fit is None
    contract = ep.validate_engine_adapter(result)
    assert contract.valid is True
    assert ep.validate_engine_adapter(result, require_fit=True).valid is False


def test_convenience_adapters_preserve_engine_and_declared_purpose():
    data = pd.DataFrame({"I1": [1, 0], "I2": [0, 1]})
    results = [
        ep.fit_mirt_adapter(data, purpose="IRT comparison"),
        ep.fit_tam_adapter(data, purpose="Rasch comparison"),
        ep.fit_brms_adapter("y ~ x", pd.DataFrame({"y": [0, 1], "x": [1, 2]}), purpose="Bayesian comparison"),
        ep.fit_lnirt_adapter(data, purpose="joint response time"),
        ep.fit_traminer_adapter(data, purpose="sequence comparison"),
        ep.fit_seqhmm_adapter(data, purpose="state comparison"),
        ep.fit_openmx_adapter({"model": "placeholder"}, purpose="SEM comparison"),
        ep.fit_diffirt_engine_adapter(data, purpose="diffusion comparison"),
        ep.fit_eyetrackingr_adapter(data, purpose="interop"),
        ep.fit_pupillometryr_adapter(data, purpose="interop"),
    ]
    assert all(r.status == "not_available" for r in results)
    cmp = ep.compare_engine_adapters(results)
    assert len(cmp) == len(results)
    assert set(cmp.status) == {"not_available"}
    with pytest.raises(EyeProcessValidationError, match="purpose"):
        ep.fit_mirt_adapter(data, purpose="")


def test_final_gdina_adapter_validates_eye_dataset_q_matrix_before_backend_gate():
    x = ep.simulate_eye_dataset(n_person=4, n_item=3, sampling_rate=10, trial_duration=.3, seed=108)
    with pytest.raises(EyeProcessValidationError, match="one row per"):
        ep.fit_gdina_adapter(x, np.ones((1, 1)))
    with pytest.raises(EyeProcessBackendError, match="GDINA"):
        ep.fit_gdina_adapter(x, np.ones((3, 1)))
    raw = ep.fit_gdina_adapter(np.array([[1, 0], [0, 1]]), np.ones((2, 1)), purpose="CDM")
    assert raw.status == "not_available"
    assert raw.engine == "GDINA"


def test_strict_legacy_diffirt_and_openmx_adapters_validate_before_gate():
    x = ep.simulate_eye_dataset(n_person=3, n_item=2, sampling_rate=10, trial_duration=.3, seed=109)
    with pytest.raises(EyeProcessValidationError, match="model"):
        ep.fit_diffirt_adapter(x, model="unsupported")
    with pytest.raises(EyeProcessBackendError, match="diffIRT"):
        ep.fit_diffirt_adapter(x, model="D")
    with pytest.raises(EyeProcessValidationError, match="must be a function"):
        ep.fit_openmx_process_model(x, model_builder=None)
    with pytest.raises(EyeProcessBackendError, match="OpenMx"):
        ep.fit_openmx_process_model(x, model_builder=lambda d: d)


def test_sequence_interoperability_matches_frozen_contract_shape():
    x = ep.simulate_eye_dataset(n_person=2, n_item=2, sampling_rate=10, trial_duration=.3, seed=110)
    x = x.copy()
    x["gaze_samples"] = x["gaze_samples"].copy()
    x["gaze_samples"]["aoi_id"] = x["gaze_samples"]["true_aoi"]

    proc = ep.as_procdata_sequence(x, source="samples")
    assert proc.attrs["eyeprocess_class"] == "eye_procdata_sequence"
    assert {"recording_id", "trial_id", "action_index", "action"}.issubset(proc.columns)

    wide = ep.as_traminer_sequence(x, source="samples")
    assert wide.attrs["eyeprocess_class"] == "eye_traminer_sequence"
    assert any(c.startswith("state_") for c in wide.columns)
    with pytest.raises(EyeProcessBackendError, match="TraMineR"):
        ep.as_traminer_sequence(x, source="samples", create_object=True)

    hmm = ep.as_seqhmm_data(x, source="samples")
    assert hmm.eyeprocess_class == "eye_seqhmm_data"
    assert len(hmm.sequences) == len(hmm.lengths)
    assert set(hmm.alphabet).issubset({"prompt", "options", "evidence"})


def test_compare_model_engines_matches_reference_tolerance_and_preserves_failures():
    data = pd.DataFrame({"x": np.arange(8, dtype=float), "y": 1.0 + 2.0 * np.arange(8, dtype=float)})

    def fit_a(z):
        return np.polyfit(z.x, z.y, 1)

    def fit_b(z):
        return np.polyfit(z.x, z.y + 1e-10, 1)

    def extract(fit):
        return {"beta": float(fit[0])}

    cmp = ep.compare_model_engines(data, {"a": fit_a, "b": fit_b}, extract, reference="a", tolerance=1e-8)
    assert cmp.eyeprocess_class == "eye_engine_comparison"
    assert cmp.estimates.equivalent.dropna().all()
    assert cmp.reference == "a"

    ax = ep.plot_eye_engine_comparison(cmp, parameter="beta")
    assert len(ax.eyeprocess_plot_data) == 2
    assert ax.get_xlabel() == "Estimate"

    cmp2 = ep.compare_model_engines(data, {"a": fit_a, "bad": lambda z: (_ for _ in ()).throw(RuntimeError("boom"))}, extract)
    assert "boom" in cmp2.estimates.error.fillna("").str.cat(sep=" ")
