"""Stable external-engine adapter contracts from eyeprocess 0.11.1.

The frozen R package deliberately separates *adapter availability* from scientific
validation and never silently substitutes a different estimator.  This Python
port preserves that contract.  R-only engines therefore return explicit
``not_available`` adapter results (or raise for the older strict wrappers)
rather than being replaced by superficially similar Python packages.
"""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import pickle
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .dataset import EyeDataset, is_eye_dataset
from .exceptions import EyeProcessBackendError, EyeProcessValidationError
from .irt import EyeResult, _result, _stable_hash
from .legacy_models import model_data, response_matrix

__all__ = [
    "eyeprocess_api_version",
    "object_schema",
    "validate_model_object",
    "upgrade_eyeprocess_model",
    "eyeprocess_deprecation",
    "external_model_engines",
    "engine_adapter_status",
    "fit_external_engine",
    "validate_engine_adapter",
    "compare_engine_adapters",
    "fit_mirt_adapter",
    "fit_tam_adapter",
    "fit_brms_adapter",
    "fit_lnirt_adapter",
    "fit_traminer_adapter",
    "fit_seqhmm_adapter",
    "fit_gdina_adapter",
    "fit_openmx_adapter",
    "fit_diffirt_engine_adapter",
    "fit_eyetrackingr_adapter",
    "fit_pupillometryr_adapter",
    "as_procdata_sequence",
    "as_traminer_sequence",
    "as_seqhmm_data",
    "fit_diffirt_adapter",
    "fit_openmx_process_model",
    "compare_model_engines",
    "plot_eye_engine_comparison",
]


class EyeProcessAPIVersion(str):
    """String-like public API version carrying the frozen R contract metadata."""

    object_schema = "2.0.0"
    storage_schema = "2.0.0"
    model_contract = "1.0.0"


_ENGINE_REGISTRY = pd.DataFrame(
    [
        ("mirt", "mirt", "IRT"),
        ("TAM", "TAM", "IRT"),
        ("brms", "brms", "Bayesian multilevel"),
        ("LNIRT", "LNIRT", "joint accuracy-RT"),
        ("GDINA", "GDINA", "diagnostic classification"),
        ("OpenMx", "OpenMx", "SEM"),
        ("diffIRT", "diffIRT", "diffusion IRT"),
        ("TraMineR", "TraMineR", "sequence analysis"),
        ("seqHMM", "seqHMM", "hidden Markov"),
        ("eyetrackingR", "eyetrackingR", "eye-tracking"),
        ("PupillometryR", "PupillometryR", "pupillometry"),
    ],
    columns=["engine", "package", "domain"],
)

# These registry entries name exact R engines.  A similarly named Python
# project is not treated as the same estimator.  An rpy2 bridge may be added as
# an explicit optional backend in a later parity tranche, but core availability
# remains False until exact-engine execution is validated.
_EXACT_ENGINE_AVAILABLE: dict[str, bool] = {name.lower(): False for name in _ENGINE_REGISTRY["engine"]}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EyeProcessValidationError(f"{name} must be one non-empty name." if name == "engine" else f"A non-empty declared scientific `{name}` is required.")
    return value.strip()


def _tag(df: pd.DataFrame, cls: str) -> pd.DataFrame:
    out = df.copy()
    out.attrs["eyeprocess_class"] = cls
    return out


def eyeprocess_api_version() -> EyeProcessAPIVersion:
    """Return the frozen public API contract version (R: ``eyeprocess_api_version``)."""
    return EyeProcessAPIVersion("0.5.0")


def object_schema(object: Any = "eye_dataset") -> dict[str, Any]:
    """Describe a stable eyeprocess object schema."""
    if not isinstance(object, str):
        if isinstance(object, EyeDataset):
            object = "eye_dataset"
        elif isinstance(object, Mapping) and (
            getattr(object, "eyeprocess_class", None) in {
                "eyeprocess_model", "eye_dynamic_irtree", "eye_theory_strategy_irt",
                "eye_gaze_diffusion_irt", "eye_functional_pupil_irt"
            }
            or "fit" in object or "model" in object
        ):
            object = "eyeprocess_model"
        else:
            raise EyeProcessValidationError("Could not infer an object schema.")
    choices = {
        "eye_dataset": {
            "version": "2.0.0", "class": "eye_dataset",
            "required_components": ["recordings", "provenance"],
            "canonical_tables": [
                "recordings", "streams", "gaze_samples", "eye_samples", "episodes", "events",
                "intervals", "responses", "coordinate_spaces", "aoi_definitions", "aoi_geometry",
                "biometrics", "calibrations", "features", "quality", "provenance",
            ],
            "identifiers": ["participant_id", "recording_id", "trial_id", "item_id", "sample_id", "event_id"],
            "invariant": "Native fields and time/coordinate transformations remain traceable through provenance.",
        },
        "eyeprocess_model": {
            "version": "1.0.0", "class": "eyeprocess_model",
            "required_components": ["engine", "specification", "fit", "diagnostics", "provenance"],
            "optional_components": ["parameters", "predictions", "data_signature", "evidence_status"],
            "invariant": "Availability of a fit is not evidence of scientific validity.",
        },
        "validation_plan": {
            "version": "1.0.0", "class": "eye_validation_job_plan",
            "required_columns": ["job_id", "scenario_id", "replication", "seed", "status"],
            "invariant": "Every job identity and seed are deterministic functions of the plan.",
        },
        "validation_collection": {
            "version": "1.0.0", "class": "eye_validation_collection",
            "required_components": ["jobs", "results", "estimates", "diagnostics", "paths"],
            "invariant": "Replications are aggregated without silently dropping failures.",
        },
        "vendor_corpus": {
            "version": "1.0.0", "class": "eye_vendor_corpus",
            "support_levels": ["declared", "fixture-tested", "empirically-validated"],
            "invariant": "Production claims require independent, version-specific empirical exports.",
        },
        "eye_storage": {
            "version": "2.0.0", "class": "eye_partitioned_storage",
            "required_files": ["_partitions.csv", "_transactions.csv", "_eyeprocess_storage.json or _eyeprocess_storage.dput"],
            "invariant": "Writes are atomic and each partition is fingerprinted.",
        },
    }
    if object not in choices:
        raise EyeProcessValidationError(f"Unknown object schema `{object}`.")
    return copy.deepcopy(choices[object])


def validate_model_object(object: Any, strict: bool = False) -> EyeResult:
    """Validate a fitted model against the stable 1.0.0 model contract."""
    list_like = isinstance(object, Mapping)
    fingerprint = _stable_hash(object)
    if not list_like:
        findings = pd.DataFrame([{
            "check": "list_like", "passed": False, "severity": "error",
            "message": "Model objects must be list-like."
        }])
        out = _result("eye_model_contract_validation", valid=False, findings=findings,
                      model_family=None, schema_version="1.0.0", object_fingerprint=fingerprint)
        if strict:
            raise EyeProcessValidationError("Model objects must be list-like.")
        return out

    cls = getattr(object, "eyeprocess_class", None)
    family_map = {
        "eye_dynamic_irtree": "dynamic_irtree",
        "eye_theory_strategy_irt": "theory_strategy",
        "eye_gaze_diffusion_irt": "gaze_diffusion",
        "eye_functional_pupil_irt": "functional_pupil",
        "eyeprocess_model": "generic",
    }
    family = family_map.get(cls)
    spec = object.get("specification", object.get("spec"))
    fit = object.get("fit", object.get("model"))
    interpretation = object.get("interpretation")
    if interpretation is None and isinstance(spec, Mapping):
        interpretation = spec.get("interpretation")
    diagnostics = object.get("diagnostics")
    if diagnostics is None and isinstance(fit, Mapping):
        diagnostics = fit.get("diagnostics")
    try:
        pickle.dumps(object)
        serializable = True
    except Exception:
        serializable = False
    rows = [
        ("recognized_class", family is not None, "error", cls or "Unrecognized model class."),
        ("has_specification", spec is not None, "error", "Model should retain its full specification."),
        ("has_fit", fit is not None, "error", "Model fit component is absent."),
        ("has_interpretation", interpretation is not None, "warning", "Interpretive safeguard is absent."),
        ("has_diagnostics", diagnostics is not None, "warning", "Convergence or diagnostic evidence is absent."),
        ("serializable", serializable, "error", "Object cannot be serialized."),
    ]
    findings = pd.DataFrame(rows, columns=["check", "passed", "severity", "message"])
    failed = findings.loc[(~findings["passed"]) & ((findings["severity"] == "error") | bool(strict))]
    valid = failed.empty
    out = _result("eye_model_contract_validation", valid=bool(valid), findings=findings,
                  model_family=family, schema_version="1.0.0", object_fingerprint=fingerprint)
    if strict and not valid:
        raise EyeProcessValidationError(" ".join(failed["message"].astype(str)))
    return out


def upgrade_eyeprocess_model(x: Any, target_version: str = "1.0.0") -> EyeResult:
    """Upgrade a list-like legacy model to the stable eyeprocess model contract."""
    if not isinstance(x, Mapping):
        raise EyeProcessValidationError("Model objects must be list-like.")
    if str(target_version) != "1.0.0":
        raise EyeProcessValidationError("This release can upgrade models only to contract version 1.0.0.")
    out = EyeResult(copy.deepcopy(dict(x)), eyeprocess_class="eyeprocess_model")
    if out.get("specification") is None and out.get("spec") is not None:
        out["specification"] = out["spec"]
    if out.get("fit") is None and out.get("model") is not None:
        out["fit"] = out["model"]
    if out.get("engine") is None:
        spec = out.get("spec", out.get("specification", {}))
        out["engine"] = spec.get("engine", "unknown") if isinstance(spec, Mapping) else "unknown"
    if out.get("diagnostics") is None and isinstance(out.get("model"), Mapping):
        out["diagnostics"] = out["model"].get("diagnostics")
    if out.get("provenance") is None:
        out["provenance"] = {"upgraded_utc": _now(), "source_class": getattr(x, "eyeprocess_class", type(x).__name__)}
    out["model_contract_version"] = str(target_version)
    return out


def eyeprocess_deprecation(old: Any, replacement: Any, since: Any, remove_after: Any, reason: str = "") -> pd.DataFrame:
    """Return a structured deprecation record."""
    return pd.DataFrame([{"old": old, "replacement": replacement, "since": since, "remove_after": remove_after, "reason": reason}])


def external_model_engines() -> pd.DataFrame:
    """List frozen external-engine adapters and exact-engine availability."""
    out = _ENGINE_REGISTRY.copy()
    out["available"] = [bool(_EXACT_ENGINE_AVAILABLE[e.lower()]) for e in out["engine"]]
    return _tag(out, "eye_external_engine_registry")


def engine_adapter_status(engine: str) -> pd.DataFrame:
    """Report one adapter's availability and stable contract."""
    value = _text(engine, "engine")
    reg = external_model_engines()
    row = reg.loc[reg["engine"].str.lower() == value.lower()].copy()
    if row.empty:
        raise EyeProcessValidationError(f"Unknown engine `{value}`.")
    row["contract"] = "returns fitted, not_available, or failed without silently selecting a model"
    return row.reset_index(drop=True)


def _not_available(engine: str, purpose: str, message: str | None = None) -> EyeResult:
    return _result(
        "eye_engine_adapter_result",
        status="not_available",
        engine=engine,
        purpose=purpose,
        message=message or f"Optional package `{engine}` is not installed.",
        timestamp_utc=_now(),
        result_class="eye_engine_not_available",
        fit=None,
    )


def fit_external_engine(engine: str, data: Any, specification: Any = None, purpose: str | None = None, **kwargs: Any) -> EyeResult:
    """Fit an exact external R engine through the frozen stable adapter contract.

    In the pure-Python parity core these R engines are deliberately unavailable;
    the function therefore returns a structured ``not_available`` result instead
    of choosing a different estimator.
    """
    if purpose is None:
        raise EyeProcessValidationError("A non-empty declared scientific `purpose` is required.")
    purpose = _text(purpose, "purpose")
    status = engine_adapter_status(engine)
    canonical = str(status.loc[0, "engine"])
    if not bool(status.loc[0, "available"]):
        out = _not_available(canonical, purpose)
        out["data_signature"] = _stable_hash({
            "shape": getattr(data, "shape", None),
            "names": list(data.columns) if isinstance(data, pd.DataFrame) else None,
            "specification": specification,
        })
        out["specification"] = specification
        out["arguments"] = dict(kwargs)
        return out
    # Kept defensive: availability cannot become True without an explicitly
    # validated exact-engine runner being added to this module.
    return _result(
        "eye_engine_adapter_result", status="failed", engine=canonical, purpose=purpose,
        error=f"Adapter implementation is unavailable for `{canonical}`.",
        data_signature=_stable_hash({"specification": specification}), timestamp_utc=_now(),
        result_class="eye_engine_adapter_failure", fit=None,
    )


def validate_engine_adapter(result: Any, require_fit: bool = False) -> EyeResult:
    """Validate the stable external-engine adapter result contract."""
    if not isinstance(result, Mapping) or getattr(result, "eyeprocess_class", None) != "eye_engine_adapter_result":
        raise EyeProcessValidationError("Expected an engine-adapter result.")
    def scalar_text(v: Any) -> bool:
        return isinstance(v, str) and bool(v.strip())
    findings = pd.DataFrame({
        "check": ["status", "engine", "purpose", "timestamp", "fit_when_required"],
        "passed": [
            result.get("status") in {"fitted", "not_available", "failed"},
            scalar_text(result.get("engine")),
            scalar_text(result.get("purpose")),
            scalar_text(result.get("timestamp_utc")),
            (not bool(require_fit)) or result.get("status") == "fitted",
        ],
    })
    return _result("eye_engine_adapter_validation", valid=bool(findings["passed"].all()), findings=findings,
                   engine=result.get("engine"), status=result.get("status"))


def compare_engine_adapters(*args: Any) -> pd.DataFrame:
    """Compare multiple external-engine adapter results."""
    results: Sequence[Any]
    if len(args) == 1 and isinstance(args[0], (list, tuple)) and not (
        isinstance(args[0], Mapping) and getattr(args[0], "eyeprocess_class", None) == "eye_engine_adapter_result"
    ):
        results = list(args[0])
    else:
        results = list(args)
    if not results:
        raise EyeProcessValidationError("At least one adapter result is required.")
    rows = []
    for x in results:
        if not isinstance(x, Mapping) or getattr(x, "eyeprocess_class", None) != "eye_engine_adapter_result":
            raise EyeProcessValidationError("All results must be engine-adapter objects.")
        rows.append({
            "engine": x.get("engine"), "status": x.get("status"), "purpose": x.get("purpose"),
            "data_signature": x.get("data_signature", pd.NA),
            "error": x.get("error", x.get("message", "")),
        })
    return _tag(pd.DataFrame(rows), "eye_engine_adapter_comparison")


def fit_mirt_adapter(data: Any, model: Any = 1, purpose: str | None = None, **kwargs: Any) -> EyeResult:
    return fit_external_engine("mirt", data, model, purpose, **kwargs)


def fit_tam_adapter(data: Any, purpose: str | None = None, **kwargs: Any) -> EyeResult:
    return fit_external_engine("TAM", data, None, purpose, **kwargs)


def fit_brms_adapter(formula: Any, data: Any, purpose: str | None = None, **kwargs: Any) -> EyeResult:
    return fit_external_engine("brms", data, formula, purpose, **kwargs)


def fit_lnirt_adapter(data: Any, purpose: str | None = None, **kwargs: Any) -> EyeResult:
    return fit_external_engine("LNIRT", data, None, purpose, **kwargs)


def fit_traminer_adapter(data: Any, purpose: str | None = None, **kwargs: Any) -> EyeResult:
    return fit_external_engine("TraMineR", data, None, purpose, **kwargs)


def fit_seqhmm_adapter(data: Any, purpose: str | None = None, **kwargs: Any) -> EyeResult:
    return fit_external_engine("seqHMM", data, None, purpose, **kwargs)


def fit_openmx_adapter(model: Any, purpose: str | None = None, **kwargs: Any) -> EyeResult:
    return fit_external_engine("OpenMx", data=None, specification=model, purpose=purpose, **kwargs)


def fit_diffirt_engine_adapter(data: Any, purpose: str | None = None, **kwargs: Any) -> EyeResult:
    return fit_external_engine("diffIRT", data, None, purpose, **kwargs)


def fit_eyetrackingr_adapter(data: Any, purpose: str | None = None, **kwargs: Any) -> EyeResult:
    return fit_external_engine("eyetrackingR", data, None, purpose, **kwargs)


def fit_pupillometryr_adapter(data: Any, purpose: str | None = None, **kwargs: Any) -> EyeResult:
    return fit_external_engine("PupillometryR", data, None, purpose, **kwargs)


def fit_gdina_adapter(data: Any, Q: Any, model: str = "GDINA", purpose: str = "cognitive diagnosis", **kwargs: Any) -> EyeResult:
    """Final 0.11.1 GDINA adapter (the R/028 override of the older R/020 form)."""
    q = np.asarray(Q)
    if q.ndim != 2:
        raise EyeProcessValidationError("`Q` must be a two-dimensional Q-matrix.")
    if is_eye_dataset(data):
        responses = response_matrix(data)
        if q.shape[0] != responses.shape[1]:
            raise EyeProcessValidationError("The Q-matrix must contain one row per response-matrix item.")
        raise EyeProcessBackendError("Exact GDINA parity requires the R `GDINA` engine; no silent Python substitute is used.")
    return fit_external_engine("GDINA", data, q, purpose, model=model, **kwargs)


def _scanpath_sequence(x: Any, source: str = "visits", collapse_consecutive: bool = True) -> pd.DataFrame:
    if not is_eye_dataset(x):
        raise EyeProcessValidationError("Expected an eye_dataset.")
    if source not in {"visits", "fixations", "samples"}:
        raise EyeProcessValidationError("source must be one of 'visits', 'fixations', or 'samples'.")
    if source == "samples":
        d = x["gaze_samples"].copy()
        if "aoi_id" not in d.columns:
            raise EyeProcessValidationError("AOIs have not been assigned to gaze samples.")
        d = d[["recording_id", "trial_id", "timestamp_seconds", "aoi_id"]].rename(columns={"timestamp_seconds": "time"})
    else:
        d = x["episodes"].copy()
        kind = "aoi_visit" if source == "visits" else "fixation"
        if "episode_type" in d.columns:
            d = d.loc[d["episode_type"].astype(str) == kind]
        cols = ["recording_id", "trial_id", "start_time", "aoi_id"]
        if not set(cols).issubset(d.columns):
            d = pd.DataFrame(columns=["recording_id", "trial_id", "time", "aoi_id"])
        else:
            d = d[cols].rename(columns={"start_time": "time"})
    if d.empty:
        return pd.DataFrame(columns=["recording_id", "trial_id", "sequence", "length"])
    d = d.loc[d["aoi_id"].notna()].sort_values(["recording_id", "trial_id", "time"], kind="stable")
    rows = []
    for (recording_id, trial_id), z in d.groupby(["recording_id", "trial_id"], sort=False, dropna=False):
        states = z["aoi_id"].astype(str).tolist()
        if collapse_consecutive and states:
            states = [s for i, s in enumerate(states) if i == 0 or s != states[i - 1]]
        rows.append({"recording_id": recording_id, "trial_id": trial_id, "sequence": " > ".join(states), "length": len(states)})
    return pd.DataFrame(rows)


def as_procdata_sequence(x: Any, source: str = "visits", collapse_consecutive: bool = True) -> pd.DataFrame:
    seqs = _scanpath_sequence(x, source=source, collapse_consecutive=collapse_consecutive)
    rows = []
    for _, row in seqs.iterrows():
        states = [] if not row["sequence"] else str(row["sequence"]).split(" > ")
        for i, state in enumerate(states, start=1):
            rows.append({"recording_id": row["recording_id"], "trial_id": row["trial_id"],
                         "action_index": i, "action": state, "timestamp_order": i})
    return _tag(pd.DataFrame(rows, columns=["recording_id", "trial_id", "action_index", "action", "timestamp_order"]), "eye_procdata_sequence")


def as_traminer_sequence(x: Any, source: str = "visits", collapse_consecutive: bool = True, create_object: bool = False) -> pd.DataFrame:
    seqs = _scanpath_sequence(x, source=source, collapse_consecutive=collapse_consecutive)
    states = [([] if not s else str(s).split(" > ")) for s in seqs.get("sequence", pd.Series(dtype=str))]
    width = max((len(s) for s in states), default=0)
    rows = []
    for (_, r), s in zip(seqs.iterrows(), states):
        row = {"recording_id": r["recording_id"], "trial_id": r["trial_id"]}
        row.update({f"state_{i+1}": s[i] if i < len(s) else pd.NA for i in range(width)})
        rows.append(row)
    cols = ["recording_id", "trial_id", *[f"state_{i+1}" for i in range(width)]]
    wide = _tag(pd.DataFrame(rows, columns=cols), "eye_traminer_sequence")
    if create_object:
        raise EyeProcessBackendError("Creating a native TraMineR sequence object requires the exact R `TraMineR` engine.")
    return wide


def as_seqhmm_data(x: Any, source: str = "visits", collapse_consecutive: bool = True) -> EyeResult:
    seqs = _scanpath_sequence(x, source=source, collapse_consecutive=collapse_consecutive)
    states = [([] if not s else str(s).split(" > ")) for s in seqs.get("sequence", pd.Series(dtype=str))]
    alphabet = sorted({state for seq in states for state in seq})
    index = seqs[["recording_id", "trial_id"]].copy() if not seqs.empty else pd.DataFrame(columns=["recording_id", "trial_id"])
    return _result("eye_seqhmm_data", sequences=states, lengths=[len(s) for s in states], alphabet=alphabet, index=index)


def fit_diffirt_adapter(x: Any, model: str = "D", **kwargs: Any) -> EyeResult:
    """Strict legacy diffusion-IRT adapter from R/020."""
    if not is_eye_dataset(x):
        raise EyeProcessValidationError("Expected an eye_dataset.")
    if model not in {"D", "Q"}:
        raise EyeProcessValidationError("model must be one of 'D' or 'Q'.")
    raise EyeProcessBackendError("Exact diffusion-IRT parity requires the R `diffIRT` engine.")


def fit_openmx_process_model(x: Any, model_builder: Callable, include_features: bool = True, **kwargs: Any) -> EyeResult:
    """Strict legacy OpenMx process-model adapter from R/020."""
    if not is_eye_dataset(x):
        raise EyeProcessValidationError("Expected an eye_dataset.")
    if not callable(model_builder):
        raise EyeProcessValidationError("`model_builder` must be a function.")
    # R checks OpenMx availability before constructing the model data/model.
    raise EyeProcessBackendError("Exact latent-variable parity requires the R `OpenMx` engine.")


def compare_model_engines(
    data: Any,
    engines: Mapping[str, Callable],
    extractors: Mapping[str, Callable] | Callable,
    reference: str | None = None,
    tolerance: float = 0.05,
) -> EyeResult:
    """Compare estimates from named fitting functions against a reference engine."""
    if not isinstance(engines, Mapping) or not engines or not all(isinstance(k, str) and callable(v) for k, v in engines.items()):
        raise EyeProcessValidationError("`engines` must be a named mapping of fitting functions.")
    names = list(engines)
    if callable(extractors):
        extractor_map = {name: extractors for name in names}
    elif isinstance(extractors, Mapping) and all(name in extractors and callable(extractors[name]) for name in names):
        extractor_map = dict(extractors)
    else:
        raise EyeProcessValidationError("Supply one extractor function per engine or one shared extractor.")
    if reference is None:
        reference = names[0]
    if reference not in names:
        raise EyeProcessValidationError("reference must name one supplied engine.")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise EyeProcessValidationError("tolerance must be a finite non-negative scalar.")

    fits: dict[str, Any] = {}
    frames: list[pd.DataFrame] = []
    for name, fun in engines.items():
        try:
            fit = fun(data)
            fits[name] = fit
        except Exception as exc:  # parity: preserve heterogeneous failures
            fits[name] = exc
            frames.append(pd.DataFrame([{"engine": name, "parameter": pd.NA, "estimate": np.nan, "error": str(exc)}]))
            continue
        try:
            z = extractor_map[name](fit)
            if isinstance(z, Mapping):
                frame = pd.DataFrame({"parameter": list(z.keys()), "estimate": list(z.values())})
            elif isinstance(z, pd.Series):
                frame = pd.DataFrame({"parameter": z.index.astype(str), "estimate": z.to_numpy()})
            elif isinstance(z, pd.DataFrame):
                frame = z.copy()
            else:
                arr = np.asarray(z)
                raise EyeProcessValidationError(f"extractor result for {name} must provide named estimates, not shape {arr.shape}.")
            if not {"parameter", "estimate"}.issubset(frame.columns):
                raise EyeProcessValidationError(f"extractor result for {name} must contain parameter and estimate.")
            frame = frame[["parameter", "estimate"]].copy()
            frame["engine"] = name
            frame["error"] = pd.NA
            frames.append(frame)
        except Exception as exc:
            frames.append(pd.DataFrame([{"engine": name, "parameter": pd.NA, "estimate": np.nan, "error": str(exc)}]))

    estimates = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=["engine", "parameter", "estimate", "error"])
    ref = estimates.loc[(estimates["engine"] == reference) & estimates["parameter"].notna(), ["parameter", "estimate"]].copy()
    ref = ref.rename(columns={"estimate": "reference_estimate"})
    estimates = estimates.merge(ref, on="parameter", how="left")
    estimates["estimate"] = pd.to_numeric(estimates["estimate"], errors="coerce")
    estimates["reference_estimate"] = pd.to_numeric(estimates["reference_estimate"], errors="coerce")
    estimates["absolute_difference"] = (estimates["estimate"] - estimates["reference_estimate"]).abs()
    finite = np.isfinite(estimates["absolute_difference"].to_numpy(float, na_value=np.nan))
    estimates["equivalent"] = pd.array([bool(v <= tolerance) if ok else pd.NA for v, ok in zip(estimates["absolute_difference"], finite)], dtype="boolean")
    return _result("eye_engine_comparison", fits=fits, estimates=estimates, reference=reference, tolerance=tolerance)


def plot_eye_engine_comparison(x: Any, parameter: str | None = None, ax: Any = None) -> Any:
    """Python counterpart of ``plot.eye_engine_comparison`` with plot-data attached."""
    if not isinstance(x, Mapping) or getattr(x, "eyeprocess_class", None) != "eye_engine_comparison":
        raise EyeProcessValidationError("x must be an eye_engine_comparison result.")
    d = x["estimates"].copy()
    available = d.loc[d["parameter"].notna(), "parameter"].astype(str).unique().tolist()
    if parameter is None:
        if not available:
            raise EyeProcessValidationError("No parameter estimates are available to plot.")
        parameter = available[0]
    z = d.loc[d["parameter"].astype(str) == str(parameter)].copy()
    if z.empty:
        raise EyeProcessValidationError(f"Parameter `{parameter}` was not found.")
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    y = np.arange(len(z))
    ax.scatter(pd.to_numeric(z["estimate"], errors="coerce"), y)
    ax.set_yticks(y, z["engine"].astype(str))
    ax.set_xlabel("Estimate")
    ax.set_title(f"Engine comparison: {parameter}")
    ref = pd.to_numeric(z["reference_estimate"], errors="coerce").dropna().unique()
    if len(ref):
        ax.axvline(float(ref[0]), linestyle="--")
    ax.eyeprocess_plot_data = z.reset_index(drop=True)
    return ax
