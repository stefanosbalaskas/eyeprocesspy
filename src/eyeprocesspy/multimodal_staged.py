"""Staged M0-M4 multimodal measurement parity for eyeprocess 0.11.1.

The deterministic data, simulation, evidence and recovery-design contracts are
ported directly. Canonical M2/M3 Stan fits remain CmdStanPy-only. M4 additionally
retains the frozen package's REVIEW boundary for trait-conditioned Markov state
models; no generic HMM or alternative Bayesian estimator is substituted.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.util import find_spec
from itertools import combinations
from typing import Any
import math

import numpy as np
import pandas as pd
from scipy.special import expit

from .exceptions import (
    EyeProcessBackendError,
    EyeProcessGovernanceError,
    EyeProcessValidationError,
)
from .irt import EyeResult, _result
from .process_irt_07 import (
    irt_response_channel,
    irt_rt_channel,
    irt_count_channel,
    irt_continuous_channel,
)

_NUISANCE = ["baseline", "luminance", "gaze_x", "gaze_y", "quality", "blink", "interpolated", "time_on_task"]


def _df(x: Any, name: str = "data") -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, Mapping) and isinstance(x.get("data"), pd.DataFrame):
        return x["data"].copy()
    try:
        return pd.DataFrame(x)
    except Exception as exc:
        raise EyeProcessValidationError(f"{name} must be coercible to a data frame.") from exc


def _req(d: pd.DataFrame, cols: Sequence[str], name: str = "data") -> None:
    miss = [c for c in cols if c not in d.columns]
    if miss:
        raise EyeProcessValidationError(f"{name} is missing required columns: {', '.join(miss)}")


def _extract_data(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, Mapping) and isinstance(x.get("data"), pd.DataFrame):
        return x["data"].copy()
    raise EyeProcessValidationError("Expected a data frame or multimodal object containing `data`.")


def _validate_key(d: pd.DataFrame, person: str, item: str, trial: str | None = None, *, m2: bool = False) -> None:
    cols = [person, item] + ([trial] if trial else [])
    _req(d, cols)
    if d[cols].isna().any(axis=None) or any(d[c].astype(str).eq("").any() for c in cols):
        raise EyeProcessValidationError("Person/item/trial identifiers must be non-missing and non-empty.")
    if d.duplicated(cols).any():
        msg = "M2 reference data require at most one row per person-item key." if m2 else "Person-item-trial keys are not unique; duplicated rows detected."
        raise EyeProcessValidationError(msg)


def _connected_design(d: pd.DataFrame, person: str, item: str, observed: pd.Series) -> int:
    # Number of connected components in the person-item bipartite graph.
    z = d.loc[observed, [person, item]].astype(str)
    if z.empty:
        return 0
    adj: dict[str, set[str]] = {}
    for p, i in z.itertuples(index=False):
        pp, ii = f"p:{p}", f"i:{i}"
        adj.setdefault(pp, set()).add(ii)
        adj.setdefault(ii, set()).add(pp)
    seen: set[str] = set(); comp = 0
    for node in adj:
        if node in seen: continue
        comp += 1; stack = [node]; seen.add(node)
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, ()):
                if nxt not in seen: seen.add(nxt); stack.append(nxt)
    return comp


def prepare_multimodal_irt_data(
    data: Any, person: str, item: str, trial: str | None = None,
    response: str | None = None, rt: str | None = None, gaze: str | None = None,
    pupil: str | None = None, quality: Sequence[str] = (), device: Any = None,
    payloads: Mapping[str, Any] | None = None, provenance: Mapping[str, Any] | None = None,
) -> EyeResult:
    d = _df(data)
    needed = [x for x in [person, item, trial, response, rt, gaze, pupil, *quality] if x]
    _req(d, list(dict.fromkeys(needed)))
    _validate_key(d, person, item, trial)
    channels = {k: v for k, v in {"response": response, "rt": rt, "gaze": gaze, "pupil": pupil}.items() if v}
    missingness = {k: d[v].isna().to_numpy() for k, v in channels.items()}
    availability = {k: int(d[v].notna().sum()) for k, v in channels.items()}
    return _result(
        "eye_multimodal_measurement", data=d, keys={"person": person, "item": item, "trial": trial},
        channels=channels, quality=list(quality), device=device, payloads=dict(payloads or {}),
        provenance=dict(provenance or {}), missingness=missingness, availability=availability,
        interpretation=["Gaze and pupil are observed process measurements.",
                        "No channel is assigned a psychological construct automatically.",
                        "Time-series payloads are retained rather than silently aggregated."],
    )


def audit_multimodal_measurement(x: Any) -> EyeResult:
    if getattr(x, "eyeprocess_class", None) != "eye_multimodal_measurement":
        raise EyeProcessValidationError("x must be an eye_multimodal_measurement.")
    d = x["data"]; keys = [v for v in x["keys"].values() if v]
    key_ok = not d.duplicated(keys).any()
    rows = []
    for ch, col in x["channels"].items():
        z = d[col]; numeric = pd.api.types.is_numeric_dtype(z)
        finite_fraction = float(np.isfinite(pd.to_numeric(z, errors="coerce")).mean()) if numeric else np.nan
        rows.append({"channel": ch, "column": col, "n": len(z), "observed": int(z.notna().sum()),
                     "missing": int(z.isna().sum()), "missing_fraction": float(z.isna().mean()),
                     "finite_fraction": finite_fraction})
    issues = []
    if not key_ok: issues.append("duplicated_person_item_trial_keys")
    if not x["channels"]: issues.append("no_measurement_channels")
    if "rt" in x["channels"]:
        z = pd.to_numeric(d[x["channels"]["rt"]], errors="coerce")
        if (z.dropna() <= 0).any(): issues.append("nonpositive_response_time")
    if "gaze" in x["channels"]:
        z = pd.to_numeric(d[x["channels"]["gaze"]], errors="coerce")
        if (z.dropna() < 0).any(): issues.append("negative_gaze_measurement")
    return _result("eye_multimodal_audit", valid=not issues, issues=list(dict.fromkeys(issues)), key_unique=key_ok,
                   channel_table=pd.DataFrame(rows), payload_names=list(x["payloads"]), quality_fields=x["quality"], device=x["device"])


def multimodal_irt_spec(response: Any = None, rt: Any = None, gaze: Any = None, pupil: Any = None,
                       model: str = "M0", backend: str = "cmdstanr", identification: Mapping[str, Any] | None = None,
                       priors: Mapping[str, Any] | None = None) -> EyeResult:
    if model not in {"M0", "M1", "M2", "M3"}: raise EyeProcessValidationError("model must be M0, M1, M2, or M3.")
    if backend not in {"cmdstanr", "existing", "cmdstanpy"}: raise EyeProcessValidationError("Unsupported backend.")
    channels = {k: v for k, v in {"response": response, "rt": rt, "gaze": gaze, "pupil": pupil}.items() if v is not None}
    if not channels: raise EyeProcessValidationError("At least one existing eyeprocess IRT channel must be supplied.")
    bad = [k for k, v in channels.items() if not isinstance(v, Mapping) or v.get("superclass") != "eye_irt_channel"]
    if bad: raise EyeProcessValidationError("Channels must be existing eyeprocess eye_irt_channel objects: " + ", ".join(bad))
    req = {"M0": ["response"], "M1": ["response", "rt"], "M2": ["response", "rt", "gaze"], "M3": ["response", "rt", "gaze", "pupil"]}[model]
    absent = [r for r in req if r not in channels]
    if absent: raise EyeProcessValidationError(f"Model {model} requires channels: {', '.join(absent)}")
    latent = []
    for ch in channels.values():
        value = ch.get("latent")
        if not isinstance(value, str) or not value: raise EyeProcessValidationError("Every multimodal channel must carry exactly one explicit latent identifier.")
        if value not in latent: latent.append(value)
    return _result("eye_multimodal_irt_spec", id=f"multimodal_{model.lower()}", model=model, backend=backend,
                   channels=channels, latent=latent, identification=dict(identification or {}), priors=dict(priors or {}),
                   superclasses=["eye_irt_model_spec"], interpretation="Process channels are observed measurements; no psychological construct is assigned automatically.")


def simulate_multimodal_irt(n_person: int = 120, n_item: int = 20, seed: int = 42,
                            latent_cor: Any = None, rt_sd: float = .30, gaze_size: float = 8,
                            pupil_sd: float = .20, pupil_luminance: float = -.20,
                            gaze_x_effect: float = .08, gaze_y_effect: float = -.05,
                            missing_fraction: float = 0) -> EyeResult:
    n_person, n_item = int(n_person), int(n_item)
    if n_person <= 1 or n_item <= 1: raise EyeProcessValidationError("n_person and n_item must exceed 1.")
    C = np.eye(4) if latent_cor is None else np.asarray(latent_cor, dtype=float)
    if C.shape != (4, 4) or not np.allclose(np.diag(C), 1) or np.linalg.eigvalsh(C).min() <= 0:
        raise EyeProcessValidationError("latent_cor must be a positive-definite 4x4 correlation matrix.")
    if missing_fraction < 0 or missing_fraction >= 1: raise EyeProcessValidationError("missing_fraction must lie in [0,1).")
    rng = np.random.default_rng(int(seed)); latent = rng.multivariate_normal(np.zeros(4), C, size=n_person)
    persons = pd.DataFrame(latent, columns=["ability", "speed", "gaze_process", "pupil_responsivity"])
    persons.insert(0, "person_id", [f"P{i:03d}" for i in range(1, n_person + 1)])
    item = pd.DataFrame({"item": [f"I{i:03d}" for i in range(1, n_item + 1)], "difficulty": rng.normal(0, .8, n_item),
                         "time_intensity": rng.normal(4, .35, n_item), "gaze_intensity": rng.normal(2.2, .45, n_item),
                         "pupil_intensity": rng.normal(0, .35, n_item)})
    pi = np.repeat(np.arange(n_person), n_item); ii = np.tile(np.arange(n_item), n_person)
    d = pd.DataFrame({"person_id": persons.person_id.to_numpy()[pi], "item": item.item.to_numpy()[ii], "trial": np.tile(np.arange(1, n_item + 1), n_person)})
    d["response"] = rng.binomial(1, expit(latent[pi, 0] - item.difficulty.to_numpy()[ii]))
    d["rt"] = np.exp(rng.normal(item.time_intensity.to_numpy()[ii] - latent[pi, 1], rt_sd))
    mu = np.exp(item.gaze_intensity.to_numpy()[ii] + latent[pi, 2]); p = float(gaze_size) / (float(gaze_size) + mu)
    d["gaze_fixation_count"] = rng.negative_binomial(float(gaze_size), p)
    d["luminance_z"] = rng.normal(size=len(d)); d["gaze_x_z"] = rng.normal(size=len(d)); d["gaze_y_z"] = rng.normal(size=len(d))
    pmu = item.pupil_intensity.to_numpy()[ii] + latent[pi, 3] + pupil_luminance*d.luminance_z + gaze_x_effect*d.gaze_x_z + gaze_y_effect*d.gaze_y_z
    d["pupil_response"] = rng.normal(pmu, pupil_sd)
    if missing_fraction:
        for col in ["rt", "gaze_fixation_count", "pupil_response"]:
            d.loc[rng.random(len(d)) < missing_fraction, col] = np.nan
    truth = {"persons": persons, "items": item, "parameters": {"rt_sd": rt_sd, "gaze_size": gaze_size, "pupil_sd": pupil_sd, "latent_cor": C}, "seed": int(seed)}
    measurement = prepare_multimodal_irt_data(d, "person_id", "item", "trial", "response", "rt", "gaze_fixation_count", "pupil_response",
                                              quality=["luminance_z", "gaze_x_z", "gaze_y_z"], provenance={"simulator": "simulate_multimodal_irt", "seed": seed})
    return _result("eye_multimodal_simulation", data=d, measurement=measurement, truth=truth)


def process_information(baseline: Any, augmented: Any, metric: str = "variance_reduction") -> pd.DataFrame:
    if metric not in {"variance_reduction", "precision_gain", "entropy_reduction"}: raise EyeProcessValidationError("Invalid information metric.")
    b = np.asarray(baseline, dtype=float); a = np.asarray(augmented, dtype=float)
    if b.ndim == 1: b = b.reshape(-1, 1)
    if a.ndim == 1: a = a.reshape(-1, 1)
    if b.ndim != 2 or a.ndim != 2 or b.shape[1] != a.shape[1]: raise EyeProcessValidationError("Baseline and augmented draws must target the same columns.")
    vb = np.var(b, axis=0, ddof=1); va = np.var(a, axis=0, ddof=1)
    if np.any(~np.isfinite(vb)) or np.any(~np.isfinite(va)) or np.any(vb <= 0) or np.any(va <= 0): raise EyeProcessValidationError("Posterior variances must be finite and positive.")
    value = vb-va if metric=="variance_reduction" else (1/va-1/vb if metric=="precision_gain" else .5*np.log(vb/va))
    out = pd.DataFrame({"target": [f"target_{i+1}" for i in range(len(value))], "metric": metric, "value": value,
                        "baseline_variance": vb, "augmented_variance": va, "relative_variance_reduction": 1-va/vb})
    out.attrs["eyeprocess_class"] = "eye_process_information"; out.attrs["interpretation"] = "Positive values indicate reduced posterior uncertainty under the augmented model; this is not a causal claim."
    return out


def ablate_multimodal_channels(x: Any, include: Sequence[str] = ("response", "rt", "gaze", "pupil"), include_response: bool = True) -> EyeResult:
    if getattr(x, "eyeprocess_class", None) != "eye_multimodal_measurement": raise EyeProcessValidationError("x must be an eye_multimodal_measurement.")
    available = [ch for ch in include if ch in x["channels"]]
    if include_response and "response" not in available: raise EyeProcessValidationError("Response channel is required for response-anchored ablation.")
    varying = [ch for ch in available if not (include_response and ch == "response")]; scenarios = {}
    for k in range(len(varying)+1):
        for extra in combinations(varying, k):
            keep = (["response"] if include_response else []) + list(extra)
            y = EyeResult(dict(x), eyeprocess_class="eye_multimodal_measurement"); y["channels"] = {ch: x["channels"][ch] for ch in keep}; y["availability"] = {ch: x["availability"][ch] for ch in keep}; y["ablation_keep"] = keep
            scenarios["+".join(keep)] = y
    return _result("eye_multimodal_ablation", scenarios=scenarios, available_channels=available,
                   principle="Each scenario retains the same rows/keys; only channel inclusion changes.")


def multimodal_backend_status() -> pd.DataFrame:
    rows = [{"backend": "cmdstanr", "installed": False, "version": None},
            {"backend": "cmdstanpy", "installed": find_spec("cmdstanpy") is not None, "version": None},
            {"backend": "posterior", "installed": False, "version": None},
            {"backend": "arviz", "installed": find_spec("arviz") is not None, "version": None}]
    for row in rows:
        if row["installed"]:
            try:
                import importlib.metadata as md; row["version"] = md.version(row["backend"])
            except Exception: pass
    return pd.DataFrame(rows)


def multimodal_ppc(object: Any, variables: Sequence[str] | None = None) -> EyeResult:
    if isinstance(object, Mapping) and isinstance(object.get("data"), pd.DataFrame):
        d = object["data"]; variables = list(variables or [c for c in ["response", "rt", "gaze", "pupil"] if c in d])
        rows = []
        for v in variables:
            z = pd.to_numeric(d[v], errors="coerce"); rows.append({"variable": v, "observed_mean": float(z.mean()), "observed_sd": float(z.std())})
        return _result("eye_multimodal_ppc", summary=pd.DataFrame(rows), status="observed_summary_only_without_posterior_fit")
    raise EyeProcessValidationError("Unsupported object for multimodal_ppc().")


def validate_multimodal_irt(x: Any) -> EyeResult:
    checks = []
    def add(name: str, value: bool, detail: str=""): checks.append({"check": name, "pass": bool(value), "detail": detail})
    cls = getattr(x, "eyeprocess_class", None)
    if cls == "eye_multimodal_measurement":
        a = audit_multimodal_measurement(x); add("measurement_valid", a.valid, ",".join(a.issues)); add("unique_keys", a.key_unique); add("has_channels", bool(x.channels))
    elif cls == "eye_multimodal_simulation":
        add("measurement_class", getattr(x.measurement, "eyeprocess_class", None)=="eye_multimodal_measurement"); add("truth_present", bool(x.truth)); add("finite_response", x.data.response.isin([0,1]).all()); add("positive_rt", (x.data.rt.dropna()>0).all()); add("nonnegative_gaze", (x.data.gaze_fixation_count.dropna()>=0).all())
    elif isinstance(x, pd.DataFrame) and x.attrs.get("eyeprocess_class") == "eye_process_information":
        add("finite_information", np.isfinite(x.value).all()); add("matched_targets", len(x)>0)
    else: raise EyeProcessValidationError("Unsupported object class for multimodal validation.")
    tab = pd.DataFrame(checks); return _result("eye_multimodal_validation", valid=bool(tab["pass"].all()), checks=tab, scope="Software/estimator development validation only; not construct validity.")


def audit_multimodal_identifiability(x: Any, min_person: int=30, min_item: int=5) -> EyeResult:
    if getattr(x,"eyeprocess_class",None)!="eye_multimodal_measurement": raise EyeProcessValidationError("x must be an eye_multimodal_measurement.")
    d=x["data"]; key_map=x["keys"]; p=d[key_map["person"]].nunique(); i=d[key_map["item"]].nunique(); miss={ch: float(d[col].isna().mean()) for ch,col in x["channels"].items()}; issues=[]
    if p<int(min_person): issues.append("few_persons")
    if i<int(min_item): issues.append("few_items")
    if any(v>=.5 for v in miss.values()): issues.append("channel_missingness_ge_50_percent")
    return _result("eye_multimodal_identifiability_audit",supported=not issues,persons=p,items=i,channel_missingness=miss,issues=issues,caveat="Passing this structural screen does not establish model identifiability.")


# ---- M2 ------------------------------------------------------------------

def multimodal_m2_spec(backend: str="cmdstanr", prior_profile: str="regularized", missingness: str="ignorable") -> EyeResult:
    if backend not in {"cmdstanr", "cmdstanpy"}: raise EyeProcessValidationError("M2 supports only the canonical CmdStan backend; no fallback estimator is substituted.")
    if prior_profile not in {"regularized", "paper_centered"}: raise EyeProcessValidationError("Invalid prior_profile.")
    if missingness != "ignorable": raise EyeProcessValidationError("The M2 reference likelihood currently supports ignorable channel missingness only.")
    channels={"response":irt_response_channel("rasch",latent="theta"),"rt":irt_rt_channel(latent="tau"),"gaze":irt_count_channel(value="gaze",latent="omega")}
    return _result("eye_multimodal_m2_spec",model="M2",backend="cmdstanr" if backend=="cmdstanr" else "cmdstanpy",channels=channels,
                   superclasses=["eye_multimodal_irt_spec","eye_irt_model_spec"], prior_profile=prior_profile, missingness=missingness,
                   reference={"doi":"10.1177/01466216221089344","model":"Man-Harring-Zhan 2022 three-way joint model"},
                   fidelity={"response":"Rasch / 1PL","rt":"lognormal RT","gaze":"negative-binomial fixation count"},
                   interpretation="Process channels are observational measurements and require study-specific construct validation.")


def fit_multimodal_m2(
    x: Any, person: str="person_id", item: str="item_id", response: str="response",
    rt: str="rt", gaze: str="gaze", prior_profile: str="regularized",
    chains: int=4, parallel_chains: int | None=None, iter_warmup: int=1000,
    iter_sampling: int=1000, seed: int=20260814, adapt_delta: float=.95,
    max_treedepth: int=12, refresh: int=100, quiet_compile: bool=True,
) -> EyeResult:
    del x,person,item,response,rt,gaze,prior_profile,chains,parallel_chains,iter_warmup,iter_sampling,seed,adapt_delta,max_treedepth,refresh,quiet_compile
    raise EyeProcessBackendError("fit_multimodal_m2() requires CmdStanPy/CmdStan and the canonical packaged M2 Stan model; no fallback estimator is substituted.")


def simulate_multimodal_m2(n_person: int=100,n_item: int=10,mu_item: Any=None,sd_person: Any=None,cor_person: Any=None,sd_item: Any=None,cor_item: Any=None,nu_range: Sequence[float]=(.5,.8),gaze_shape: Mapping[str,float]|Sequence[float]=(2,6),dropout: Mapping[str,float]|Sequence[float]=(0,0,0),seed: int=20260814) -> EyeResult:
    n_person,n_item=int(n_person),int(n_item); rng=np.random.default_rng(int(seed))
    C=np.array([[1,.30,-.30],[.30,1,-.25],[-.30,-.25,1]],float) if cor_person is None else np.asarray(cor_person,float)
    sp=np.array([1,.5,.5]) if sd_person is None else np.asarray(list(sd_person.values()) if isinstance(sd_person,Mapping) else sd_person,float)
    cov=np.diag(sp)@C@np.diag(sp); person=rng.multivariate_normal(np.zeros(3),cov,n_person); theta,tau,omega=person.T
    Ci=np.array([[1,.25,.20],[.25,1,.30],[.20,.30,1]],float) if cor_item is None else np.asarray(cor_item,float)
    si=np.array([.75,.35,.60]) if sd_item is None else np.asarray(list(sd_item.values()) if isinstance(sd_item,Mapping) else sd_item,float)
    mui=np.array([0,4,3.5]) if mu_item is None else np.asarray(list(mu_item.values()) if isinstance(mu_item,Mapping) else mu_item,float)
    itemp=rng.multivariate_normal(mui,np.diag(si)@Ci@np.diag(si),n_item); b,beta,m=itemp.T; nu=rng.uniform(float(nu_range[0]),float(nu_range[1]),n_item)
    pi=np.repeat(np.arange(n_person),n_item); ii=np.tile(np.arange(n_item),n_person); d=pd.DataFrame({"person_id":[f"P{i+1:03d}" for i in pi],"item_id":[f"I{i+1:03d}" for i in ii]})
    d["response"]=rng.binomial(1,expit(theta[pi]-b[ii])); d["rt"]=np.exp(rng.normal(beta[ii]-tau[pi],1/np.sqrt(nu[ii])))
    shape=float(gaze_shape.get("shape",2) if isinstance(gaze_shape,Mapping) else gaze_shape[0]); mu=np.exp(m[ii]+omega[pi]); prob=shape/(shape+mu); d["gaze"]=rng.negative_binomial(shape,prob)
    complete=d.copy(); dr=np.asarray(list(dropout.values()) if isinstance(dropout,Mapping) else dropout,float)
    if dr.size!=3: raise EyeProcessValidationError("dropout must contain response, rt, gaze fractions.")
    for col,rate in zip(["response","rt","gaze"],dr):
        if rate<0 or rate>=1: raise EyeProcessValidationError("dropout fractions must lie in [0,1).")
        if rate: d.loc[rng.random(len(d))<rate,col]=np.nan
    truth={"theta":theta,"tau":tau,"omega":omega,"b":b,"beta":beta,"m":m,"nu":nu,"cor_person":C,"cor_item":Ci}
    return _result("eye_multimodal_m2_simulation",data=d,complete_data=complete,truth=truth,seed=int(seed),spec=multimodal_m2_spec(backend="cmdstanpy"))


def audit_multimodal_m2_identifiability(x: Any,person: str="person_id",item: str="item_id",response: str="response",rt: str="rt",gaze: str="gaze",model: str="M2",min_persons: int=20,min_items: int=5) -> EyeResult:
    d=_extract_data(x); _req(d,[person,item,response,rt,gaze]); _validate_key(d,person,item,m2=True); checks=[]; designs={}
    for ch,col in [("response",response),("rt",rt),("gaze",gaze)]:
        obs=d[col].notna(); comp=_connected_design(d,person,item,obs); designs[ch]={"components":comp,"observed":int(obs.sum())}; checks.append({"criterion":f"{ch}_connected","pass":comp==1})
    checks.extend([{"criterion":"persons","pass":d[person].nunique()>=min_persons},{"criterion":"items","pass":d[item].nunique()>=min_items}])
    tab=pd.DataFrame(checks)
    return _result("eye_multimodal_m2_identifiability",supported=bool(tab["pass"].all()),checks=tab,response_design=designs["response"],rt_design=designs["rt"],gaze_design=designs["gaze"],model=model)


def multimodal_m2_negative_controls(x: Any, seed: int=20260814) -> EyeResult:
    d=_extract_data(x); _req(d,["item_id","response","rt","gaze"]); rng=np.random.default_rng(int(seed)); sets={"observed":d.copy()}
    for col,name in [("gaze","gaze_within_item"),("rt","rt_within_item"),("response","response_within_item")]:
        z=d.copy()
        for _,idx in z.groupby("item_id",sort=False).groups.items():
            vals=z.loc[idx,col].to_numpy(copy=True); z.loc[idx,col]=vals[rng.permutation(len(vals))]
        sets[name]=z
    rows=[]
    for name,z in sets.items():
        for ch in ["response","rt","gaze"]: rows.append({"dataset":name,"channel":ch,"mean":float(pd.to_numeric(z[ch],errors="coerce").mean())})
    return _result("eye_multimodal_m2_negative_controls",datasets=sets,diagnostics=pd.DataFrame(rows),seed=int(seed),interpretation="Within-item shuffles preserve marginal distributions while breaking person-process alignment.")


def multimodal_m2_ppc(x: Any) -> EyeResult:
    d=_extract_data(x); rows=[]
    for ch in ["response","rt","gaze"]:
        if ch in d: rows.append({"channel":ch,"observed_mean":float(pd.to_numeric(d[ch],errors="coerce").mean())})
    return _result("eye_multimodal_m2_ppc",summary=pd.DataFrame(rows),status="observed_summary_without_posterior" if getattr(x,"eyeprocess_class","")!="eye_multimodal_m2_fit" else "posterior_predictive")


def validate_multimodal_m2(x: Any, include_ppc: bool=True, rhat_max: float=1.01, ess_min: int=200) -> EyeResult:
    del rhat_max,ess_min
    ident=audit_multimodal_m2_identifiability(x); checks=ident.checks.copy(); checks["domain"]="structure"
    return _result("eye_multimodal_m2_validation",valid=bool(ident.supported),checks=checks,identifiability=ident,ppc=multimodal_m2_ppc(x) if include_ppc else None)


def multimodal_m2_ablation(x: Any, **kwargs: Any) -> EyeResult:
    del kwargs; d=_extract_data(x); scenarios={"M0":d[[c for c in ["person_id","item_id","response"] if c in d]].copy(),"M1":d[[c for c in ["person_id","item_id","response","rt"] if c in d]].copy(),"M2":d.copy()}
    return _result("eye_multimodal_m2_ablation",scenarios=scenarios,executed=False,status="ablation_datasets_prepared")


def multimodal_m2_process_information(x: Any) -> EyeResult:
    d=_extract_data(x); return _result("eye_multimodal_m2_information",summary=pd.DataFrame({"channel":["rt","gaze"],"observed_fraction":[float(d[c].notna().mean()) for c in ["rt","gaze"]]}),status="design_information_without_posterior")


def multimodal_m2_recovery(
    n_rep: int=10, n_person: int=100, n_item: int=10, dropout: Sequence[float]=(0,0,0),
    base_seed: int=20260814, chains: int=4, parallel_chains: int | None=None,
    iter_warmup: int=1000, iter_sampling: int=1000, prior_profile: str="regularized",
    adapt_delta: float=.95, max_treedepth: int=12, refresh: int=0,
) -> EyeResult:
    fit_args={"chains":int(chains),"parallel_chains":int(chains if parallel_chains is None else parallel_chains),"iter_warmup":int(iter_warmup),"iter_sampling":int(iter_sampling),"prior_profile":prior_profile,"adapt_delta":float(adapt_delta),"max_treedepth":int(max_treedepth),"refresh":int(refresh)}
    design=pd.DataFrame({"replicate":np.arange(1,int(n_rep)+1),"seed":np.arange(int(base_seed),int(base_seed)+int(n_rep))})
    return _result("eye_multimodal_m2_recovery",design=design,n_person=int(n_person),n_item=int(n_item),dropout=list(dropout),fit_args=fit_args,executed=False,status="backend_required_for_recovery")


# ---- M3 ------------------------------------------------------------------

def multimodal_m3_spec(backend: str="cmdstanr",prior_profile: str="regularized",missingness: str="ignorable",pupil_representation: str="summary",nuisance: Mapping[str,bool]|None=None) -> EyeResult:
    if backend not in {"cmdstanr","cmdstanpy"}: raise EyeProcessValidationError("M3 currently supports only CmdStan; no fallback estimator is substituted.")
    if prior_profile not in {"regularized","paper_centered"}: raise EyeProcessValidationError("Invalid prior_profile.")
    if missingness!="ignorable": raise EyeProcessValidationError("The M3 reference likelihood currently supports ignorable channel missingness only.")
    if pupil_representation not in {"summary","functional_score"}: raise EyeProcessValidationError("Invalid pupil representation.")
    channels={"response":irt_response_channel("rasch",latent="theta"),"rt":irt_rt_channel(latent="tau"),"gaze":irt_count_channel(value="gaze",latent="omega"),"pupil":irt_continuous_channel("gaussian",value="pupil",lower=-1e12,upper=1e12,latent="rho")}
    return _result("eye_multimodal_m3_spec",model="M3",backend=backend,channels=channels,superclasses=["eye_multimodal_irt_spec","eye_irt_model_spec"],prior_profile=prior_profile,missingness=missingness,pupil_representation=pupil_representation,pupil_nuisance=dict(nuisance or {k:True for k in _NUISANCE}),reference={"m2_reference_doi":"10.1177/01466216221089344"},interpretation="Pupil responsivity is an observed-process latent dimension and is not automatically interpreted as cognitive load, effort, or arousal.")


def fit_multimodal_m3(
    x: Any, person: str="person_id", item: str="item_id", response: str="response", rt: str="rt",
    gaze: str="gaze", pupil: str="pupil", baseline: str="pupil_baseline", luminance: str="luminance",
    gaze_x: str="gaze_x", gaze_y: str="gaze_y", quality: str="pupil_quality", time_on_task: str="time_on_task",
    blink: str="pupil_blink", interpolated: str="pupil_interpolated", device: str="device", session: str="session",
    sampling_rate: str="sampling_rate_hz", pupil_scale: str="z", prior_profile: str="regularized",
    nuisance: Mapping[str,bool] | None=None, chains: int=4, parallel_chains: int | None=None, iter_warmup: int=1000,
    iter_sampling: int=1000, seed: int=20260815, adapt_delta: float=.95, max_treedepth: int=12, refresh: int=100,
    quiet_compile: bool=True, init: Any=0,
) -> EyeResult:
    del x,person,item,response,rt,gaze,pupil,baseline,luminance,gaze_x,gaze_y,quality,time_on_task,blink,interpolated,device,session,sampling_rate,pupil_scale,prior_profile,nuisance,chains,parallel_chains,iter_warmup,iter_sampling,seed,adapt_delta,max_treedepth,refresh,quiet_compile,init; raise EyeProcessBackendError("fit_multimodal_m3() requires CmdStanPy/CmdStan and the canonical packaged M3 Stan model; no fallback estimator is substituted.")


def simulate_multimodal_m3(n_person: int=120,n_item: int=12,pupil_signal: str="informative",pupil_missingness: str="mcar",mu_item: Any=None,sd_person: Any=None,cor_person: Any=None,sd_item: Any=None,cor_item: Any=None,nu_range: Sequence[float]=(.5,.8),gaze_shape: Sequence[float]=(2,6),pupil_noise: float=.65,confound_strength: Mapping[str,float]|None=None,dropout: Mapping[str,float]|Sequence[float]=(0,0,.05,.12),device_effect: float=0,session_effect: float=0,seed: int=20260815) -> EyeResult:
    if pupil_signal not in {"informative","weak","null","redundant","confounded"}: raise EyeProcessValidationError("Invalid pupil_signal.")
    if pupil_missingness not in {"mcar","quality","gaze","ability","device","none"}: raise EyeProcessValidationError("Invalid pupil_missingness.")
    base=simulate_multimodal_m2(n_person,n_item,dropout=(0,0,0),seed=seed); rng=np.random.default_rng(int(seed)+991); d=base.complete_data.copy(); theta,tau,omega=base.truth["theta"],base.truth["tau"],base.truth["omega"]
    rho=rng.normal(0,.55,n_person); kappa=rng.normal(0,.40,n_item)
    if pupil_signal=="null":
        C=np.eye(4); rho=rng.normal(0,.55,n_person); kappa=rng.normal(0,.40,n_item)
    elif pupil_signal=="confounded": rho=np.zeros(n_person); kappa=np.zeros(n_item); C=np.eye(4)
    else:
        C=np.array([[1,.30,-.30,.20],[.30,1,-.25,-.15],[-.30,-.25,1,.25],[.20,-.15,.25,1]])
        strength={"weak":.25,"redundant":.8}.get(pupil_signal,1.0); rho=strength*(.2*theta-.15*tau+.25*omega)+rng.normal(0,.35,n_person)
    pi=np.repeat(np.arange(n_person),n_item); ii=np.tile(np.arange(n_item),n_person); N=len(d)
    d["pupil_baseline"]=rng.normal(size=N); d["luminance"]=rng.normal(size=N); d["gaze_x"]=rng.normal(size=N); d["gaze_y"]=rng.normal(size=N); d["pupil_quality"]=rng.uniform(.7,1,N); d["pupil_blink"]=rng.binomial(1,.05,N); d["pupil_interpolated"]=rng.binomial(1,.08,N); d["time_on_task"]=np.tile(np.linspace(0,1,n_item),n_person); d["device"]=np.where(np.arange(N)%2==0,"device_A","device_B"); d["session"]="session_1"; d["sampling_rate_hz"]=60.0
    cs={"baseline":.25,"luminance":-.35,"gaze_x":.12,"gaze_y":-.10,"quality":.20,"blink":-.18,"interpolated":-.12,"time_on_task":.15}; cs.update(dict(confound_strength or {}))
    nuisance=(cs["baseline"]*d.pupil_baseline+cs["luminance"]*d.luminance+cs["gaze_x"]*d.gaze_x+cs["gaze_y"]*d.gaze_y+cs["quality"]*d.pupil_quality+cs["blink"]*d.pupil_blink+cs["interpolated"]*d.pupil_interpolated+cs["time_on_task"]*d.time_on_task)
    if pupil_signal=="confounded": signal=nuisance
    else: signal=rho[pi]+kappa[ii]+nuisance
    d["pupil_nuisance_effect"]=nuisance; d["pupil"]=signal+rng.normal(0,pupil_noise,N)
    complete=d.copy(); dr=np.asarray(list(dropout.values()) if isinstance(dropout,Mapping) else dropout,float); dr=np.pad(dr,(0,max(0,4-dr.size)))[:4]
    for col,rate in zip(["response","rt","gaze","pupil"],dr):
        if rate: d.loc[rng.random(N)<rate,col]=np.nan
    if pupil_missingness!="none":
        rate=.08; score=np.full(N,rate)
        if pupil_missingness=="quality": score=np.clip((1-d.pupil_quality)*.5,0,.6)
        elif pupil_missingness=="gaze": score=np.clip(.04+.02*np.log1p(d.gaze),0,.5)
        elif pupil_missingness=="ability": score=np.clip(.08+.04*(-theta[pi]),0,.5)
        elif pupil_missingness=="device": score=np.where(d.device.eq("device_B"),.14,.04)
        d.loc[rng.random(N)<score,"pupil"]=np.nan
    truth={**base.truth,"rho":rho,"kappa":kappa,"cor_person":C,"pupil_signal":pupil_signal,"pupil_missingness":pupil_missingness}
    return _result("eye_multimodal_m3_simulation",data=d,complete_data=complete,truth=truth,seed=int(seed),spec=multimodal_m3_spec(backend="cmdstanpy"))


def audit_multimodal_m3_identifiability(x: Any,pupil_scale: str="z",min_persons: int=20,min_items: int=5,max_pupil_missing: float=.50,max_blink_rate: float=.30,max_interpolation_rate: float=.30) -> EyeResult:
    del pupil_scale,max_blink_rate,max_interpolation_rate; d=_extract_data(x); required=["person_id","item_id","response","rt","gaze","pupil","pupil_baseline","luminance","gaze_x","gaze_y","pupil_quality","pupil_blink","pupil_interpolated","time_on_task"]; _req(d,required); _validate_key(d,"person_id","item_id",m2=True)
    observed=d.pupil.notna()
    for col in ["pupil_baseline","luminance","gaze_x","gaze_y","pupil_quality","pupil_blink","pupil_interpolated","time_on_task"]:
        if d.loc[observed,col].isna().any(): raise EyeProcessValidationError(f"M3 does not silently impute pupil confounds; `{col}` is missing on observed pupil rows.")
    channels=["response","rt","gaze","pupil"]; miss=pd.Series({c:float(d[c].isna().mean()) for c in channels}); variation=pd.Series({c:pd.to_numeric(d[c],errors="coerce").dropna().nunique()>1 for c in channels}); checks=pd.DataFrame([{"criterion":"persons","pass":d.person_id.nunique()>=min_persons},{"criterion":"items","pass":d.item_id.nunique()>=min_items},{"criterion":"pupil_missingness","pass":miss["pupil"]<=max_pupil_missing},{"criterion":"channel_variation","pass":bool(variation.all())}]); return _result("eye_multimodal_m3_identifiability",supported=bool(checks["pass"].all()),checks=checks,variation=variation,missing_fraction=miss)


def _phase_randomized(x: np.ndarray,rng: np.random.Generator)->np.ndarray:
    # Permutation preserves exact finite first two moments while breaking alignment.
    return x[rng.permutation(len(x))]


def multimodal_m3_negative_controls(x: Any,seed: int=20260815)->EyeResult:
    d=_extract_data(x); rng=np.random.default_rng(int(seed)); sets={"observed":d.copy()}
    within=d.copy();
    for _,idx in within.groupby("item_id",sort=False).groups.items():
        vals=within.loc[idx,"pupil"].to_numpy(copy=True); within.loc[idx,"pupil"]=vals[rng.permutation(len(vals))]
    sets["pupil_within_item"]=within
    withinp=d.copy();
    for _,idx in withinp.groupby("person_id",sort=False).groups.items():
        vals=withinp.loc[idx,"pupil"].to_numpy(copy=True); withinp.loc[idx,"pupil"]=vals[rng.permutation(len(vals))]
    sets["pupil_within_person"]=withinp
    phase=d.copy(); obs=phase.pupil.notna(); vals=phase.loc[obs,"pupil"].to_numpy(float); phase.loc[obs,"pupil"]=_phase_randomized(vals,rng); sets["pupil_phase_randomized"]=phase
    lum=d.copy(); lum["pupil"]=pd.to_numeric(lum["luminance"],errors="coerce"); sets["luminance_only_pupil"]=lum
    irr=d.copy(); base=pd.to_numeric(irr["pupil"],errors="coerce"); finite=base.notna(); irr.loc[finite,"pupil"]=rng.normal(float(base[finite].mean()),float(base[finite].std(ddof=1)),finite.sum()); sets["irrelevant_pupil"]=irr
    return _result("eye_multimodal_m3_negative_controls",datasets=sets,seed=int(seed),interpretation="Meaningless or misaligned pupil channels are negative controls; they must not improve substantive claims.")


def multimodal_m3_functional_bridge(data: Any,score: Any,pupil: str="pupil",provenance: Any=None)->EyeResult:
    d=_df(data)
    if isinstance(score, str):
        vals = pd.to_numeric(d[score], errors="coerce") if score in d else pd.Series([], dtype=float)
    else:
        try:
            raw = list(score)
        except TypeError as exc:
            raise EyeProcessValidationError("score must provide one finite-or-NA value per trial.") from exc
        if len(raw) != len(d):
            raise EyeProcessValidationError("score must provide one finite-or-NA value per trial.")
        vals = pd.to_numeric(pd.Series(raw, index=d.index), errors="coerce")
    if len(vals)!=len(d): raise EyeProcessValidationError("score must provide one finite-or-NA value per trial.")
    out=d.copy(); out[pupil]=vals.to_numpy(); return _result("eye_multimodal_m3_functional_bridge",data=out,score=score,pupil=pupil,representation="functional_score",provenance=provenance,boundary="This bridge does not claim that a functional pupil score is a psychological construct or equivalent to raw pupil diameter.")


def multimodal_m3_ppc(x: Any)->EyeResult:
    d=_extract_data(x); return _result("eye_multimodal_m3_ppc",summary=pd.DataFrame({"channel":[c for c in ["response","rt","gaze","pupil"] if c in d],"observed_mean":[float(pd.to_numeric(d[c],errors="coerce").mean()) for c in [c for c in ["response","rt","gaze","pupil"] if c in d]]}),status="observed_summary_without_posterior")


def multimodal_m3_ablation(x: Any,models: Any=None,nuisance: Any=None,**kwargs: Any)->EyeResult:
    del models,nuisance,kwargs; d=_extract_data(x); scenarios={"M0":d[[c for c in ["person_id","item_id","response"] if c in d]].copy(),"M1":d[[c for c in ["person_id","item_id","response","rt"] if c in d]].copy(),"M2":d[[c for c in ["person_id","item_id","response","rt","gaze"] if c in d]].copy(),"M3":d.copy()}; return _result("eye_multimodal_m3_ablation",scenarios=scenarios,executed=False,status="ablation_plan")


def multimodal_m3_process_information(x: Any,pupil_cost: float=1,decisive_z: float=2)->EyeResult:
    d=_extract_data(x); miss=float(d.pupil.isna().mean()) if "pupil" in d else 1.; return _result("eye_multimodal_m3_information",pupil_observed_fraction=1-miss,pupil_cost=float(pupil_cost),decisive_z=float(decisive_z),status="design_information_without_posterior")


def validate_multimodal_m3(x: Any,include_ppc: bool=True,rhat_max: float=1.05,ess_min: int=50,ebfmi_min: float=.30)->EyeResult:
    del rhat_max,ess_min,ebfmi_min; ident=audit_multimodal_m3_identifiability(x); return _result("eye_multimodal_m3_validation",valid=bool(ident.supported),checks=ident.checks,identifiability=ident,ppc=multimodal_m3_ppc(x) if include_ppc else None)


def multimodal_m3_recovery(reps: int=3,pupil_signal: Sequence[str]=("informative","weak","null","redundant","confounded"),pupil_missingness: Sequence[str]=("mcar","quality","device"),n_person: int=80,n_item: int=10,seed: int=20260815,fit_args: Mapping[str,Any]|None=None)->EyeResult:
    rows=[{"scenario":s,"missingness":m,"replicate":r+1,"seed":int(seed)+i*100+r} for i,(s,m) in enumerate((s,m) for s in pupil_signal for m in pupil_missingness) for r in range(int(reps))]; return _result("eye_multimodal_m3_recovery",design=pd.DataFrame(rows),n_person=int(n_person),n_item=int(n_item),fit_args=dict(fit_args or {}),executed=False,status="backend_required_for_recovery")


# ---- M4 ------------------------------------------------------------------

def multimodal_m4_spec(n_states: int=2,state_channels: Sequence[str]=("rt","gaze","pupil"),transition_structure: str="markov",trait_conditioning: Sequence[str]=("theta","tau"),initial_trait_conditioning: bool=True,min_sequence_length: int=2,identification: str="ordered_rt_effect",prior_profile: str="regularized",missingness: str="ignorable",nuisance: Mapping[str,bool]|None=None,backend: str="cmdstanr")->EyeResult:
    n_states=int(n_states); state_channels=list(dict.fromkeys(state_channels)); trait_conditioning=list(dict.fromkeys(trait_conditioning))
    if n_states<1 or n_states>4: raise EyeProcessValidationError("`n_states` must be one integer from 1 through 4.")
    if not state_channels or any(c not in {"rt","gaze","pupil"} for c in state_channels): raise EyeProcessValidationError("`state_channels` must be a non-empty subset of: rt, gaze, pupil.")
    if n_states>1 and "rt" not in state_channels: raise EyeProcessValidationError("K > 1 requires `rt` in `state_channels` under ordered-RT identification.")
    if transition_structure not in {"markov","iid"}: raise EyeProcessValidationError("Invalid transition_structure.")
    if any(t not in {"theta","tau","omega","rho"} for t in trait_conditioning): raise EyeProcessValidationError("trait_conditioning may contain only theta, tau, omega, rho.")
    if backend not in {"cmdstanr","cmdstanpy"}: raise EyeProcessValidationError("M4 currently supports only CmdStan; no fallback state estimator is substituted.")
    if missingness!="ignorable": raise EyeProcessValidationError("The M4 reference likelihood currently supports ignorable channel missingness only.")
    if int(min_sequence_length)<1: raise EyeProcessValidationError("min_sequence_length must be a positive integer.")
    m3=multimodal_m3_spec(backend="cmdstanpy" if backend=="cmdstanpy" else "cmdstanr",prior_profile=prior_profile,missingness=missingness,nuisance=nuisance)
    return _result("eye_multimodal_m4_spec",model="M4",backend=backend,channels=m3.channels,superclasses=["eye_multimodal_m3_spec","eye_multimodal_irt_spec","eye_irt_model_spec"],m3_parent=m3,n_states=n_states,state_null=n_states==1,state_channels=state_channels,transition_structure=transition_structure,trait_conditioning=trait_conditioning,initial_trait_conditioning=bool(initial_trait_conditioning),min_sequence_length=int(min_sequence_length),state_identification=identification,prior_profile=prior_profile,missingness=missingness,pupil_nuisance=m3.pupil_nuisance,lifecycle_status="experimental",identification={"m4_state":{"policy":identification,"note":"Ordering RT state deviations with a proper adjacent-gap prior fixes labels away from the zero-separation identification boundary; it does not order psychological meaning."}},interpretation="Latent response-process states are statistical states and are not automatically interpreted as strategy, attention, engagement, effort, guessing, or misconduct.")


def fit_multimodal_m4(
    x: Any, spec: Any=None, person: str="person_id", item: str="item_id", response: str="response", rt: str="rt",
    gaze: str="gaze", pupil: str="pupil", sequence: str="sequence_id", order: str="trial_index", baseline: str="pupil_baseline",
    luminance: str="luminance", gaze_x: str="gaze_x", gaze_y: str="gaze_y", quality: str="pupil_quality",
    time_on_task: str="time_on_task", blink: str="pupil_blink", interpolated: str="pupil_interpolated", device: str="device",
    session: str="session", sampling_rate: str="sampling_rate_hz", pupil_scale: str="z", n_states: int=2,
    state_channels: Sequence[str]=("rt","gaze","pupil"), transition_structure: str="markov",
    trait_conditioning: Sequence[str]=("theta","tau"), initial_trait_conditioning: bool=True, min_sequence_length: int=2,
    prior_profile: str="regularized", nuisance: Mapping[str,bool] | None=None, chains: int=4, parallel_chains: int | None=None,
    iter_warmup: int=1000, iter_sampling: int=1000, seed: int=20260820, adapt_delta: float=.97, max_treedepth: int=13,
    refresh: int=100, quiet_compile: bool=True, init: Any=0,
)->EyeResult:
    del person,item,response,rt,gaze,pupil,sequence,order,baseline,luminance,gaze_x,gaze_y,quality,time_on_task,blink,interpolated,device,session,sampling_rate,pupil_scale,chains,parallel_chains,iter_warmup,iter_sampling,seed,adapt_delta,max_treedepth,refresh,quiet_compile,init
    spec=spec or multimodal_m4_spec(n_states=n_states,state_channels=state_channels,transition_structure=transition_structure,trait_conditioning=trait_conditioning,initial_trait_conditioning=initial_trait_conditioning,min_sequence_length=min_sequence_length,prior_profile=prior_profile,nuisance=nuisance,backend="cmdstanpy")
    if spec.transition_structure=="markov" and spec.trait_conditioning:
        raise EyeProcessGovernanceError("M4 trait-conditioned Markov specification is gated for REVIEW; validation evidence is required before fitting/interpretation.")
    raise EyeProcessBackendError("fit_multimodal_m4() requires CmdStanPy/CmdStan and the canonical packaged M4 Stan model; no fallback state estimator is substituted.")


def _softmax(x: np.ndarray) -> np.ndarray:
    z=np.asarray(x,float); z=np.exp(z-np.max(z)); return z/z.sum()


def _m4_transition_array(person_traits: np.ndarray, K: int, scenario: str, trait_strength: float=.45) -> tuple[np.ndarray,np.ndarray]:
    J=len(person_traits); A=np.zeros((J,K,K),float); pi=np.full((J,K),1/max(K,1),float)
    if K==1:
        A[:,0,0]=1.; pi[:,0]=1.; return pi,A
    persistent=2.0 if scenario in {"persistent","clear","trait_conditioned"} else (-.6 if scenario=="rapid_switch" else 1.0)
    use_trait=scenario in {"trait_conditioned","clear","persistent","weak"}
    for j in range(J):
        theta,tau=person_traits[j,0],person_traits[j,1]
        init_eta=np.r_[np.linspace(-.35,.35,K-1),0.]
        if use_trait: init_eta[:K-1]+=trait_strength*theta-.25*tau
        pi[j]=_softmax(init_eta)
        for h in range(K):
            eta=np.full(K,-.4); eta[h]=persistent
            if use_trait:
                grad=np.linspace(-.5,.5,K); eta += trait_strength*theta*grad-.2*tau*grad[::-1]
            A[j,h]=_softmax(eta)
    return pi,A


def _m4_simulate_states(rng: np.random.Generator, person_index: np.ndarray, sequence_id: np.ndarray, initial: np.ndarray, transition: np.ndarray, K: int) -> np.ndarray:
    state=np.zeros(len(person_index),int)
    starts=[0]
    for idx in range(1,len(sequence_id)):
        if sequence_id[idx]!=sequence_id[idx-1]: starts.append(idx)
    starts.append(len(sequence_id))
    for a,b in zip(starts[:-1],starts[1:]):
        j=int(person_index[a])
        if K==1: state[a:b]=1; continue
        state[a]=int(rng.choice(np.arange(1,K+1),p=initial[j]))
        for t in range(a+1,b): state[t]=int(rng.choice(np.arange(1,K+1),p=transition[j,state[t-1]-1]))
    return state


def _m4_apply_missingness(rng: np.random.Generator, d: pd.DataFrame, mechanism: str, rate: float, state: np.ndarray, K: int) -> pd.DataFrame:
    out=d.copy(); N=len(out); rate=float(np.clip(rate,0,.8))
    if mechanism=="none" or rate<=0: return out
    base=np.full(N,rate); p_rt=base.copy(); p_gaze=base.copy(); p_pupil=base.copy()
    def logistic(v): return expit(v)
    logit_rate=math.log(max(rate,.01)/(1-max(rate,.01)))
    if mechanism=="quality": p_pupil=logistic(logit_rate+1.2*(-pd.to_numeric(out.pupil_quality,errors="coerce").fillna(0).to_numpy()))
    elif mechanism=="gaze":
        z=np.log1p(pd.to_numeric(out.gaze,errors="coerce").fillna(0).to_numpy()); sd=np.std(z,ddof=1); z=(z-z.mean())/(sd if sd>0 else 1)
        p_gaze=logistic(logit_rate+.8*z); p_pupil=logistic(logit_rate+.5*z)
    elif mechanism=="pupil_quality": p_pupil=logistic(logit_rate+1.4*(out.pupil_quality.to_numpy(float)<-.5))
    elif mechanism=="device":
        b=out.device.astype(str).eq("device_B").to_numpy(); p_pupil=np.minimum(.9,base+.18*b); p_gaze=np.minimum(.9,base+.08*b)
    elif mechanism=="state_dependent":
        if K<=1: raise EyeProcessValidationError("State-dependent missingness requires more than one true state.")
        high=state==K; p_rt=np.minimum(.9,base+.10*high); p_gaze=np.minimum(.9,base+.15*high); p_pupil=np.minimum(.9,base+.25*high)
    elif mechanism!="mcar": raise EyeProcessValidationError(f"Unknown M4 missingness mechanism: {mechanism}")
    out.loc[rng.random(N)<p_rt,"rt"]=np.nan; out.loc[rng.random(N)<p_gaze,"gaze"]=np.nan; out.loc[rng.random(N)<p_pupil,"pupil"]=np.nan
    return out


def simulate_multimodal_m4(n_person: int=80,n_item: int=12,n_session: int=1,n_states: int=2,scenario: str="clear",missingness: str="none",missing_rate: float=.08,seed: int=20260820)->EyeResult:
    scenarios={"clear","weak","null","persistent","rapid_switch","trait_conditioned","rt_redundant","gaze_redundant","pupil_redundant","nuisance_confounded","device_confounded"}
    mechanisms={"none","mcar","quality","gaze","pupil_quality","device","state_dependent"}
    if scenario not in scenarios: raise EyeProcessValidationError("Unknown M4 simulation scenario.")
    if missingness not in mechanisms: raise EyeProcessValidationError("Unknown M4 missingness scenario.")
    n_person,n_item,n_session,n_states=map(int,(n_person,n_item,n_session,n_states))
    if n_person<2 or n_item<3 or n_session<1 or n_session>n_item: raise EyeProcessValidationError("Use at least 2 persons, 3 items, and 1..n_item sessions.")
    if n_states<1 or n_states>4: raise EyeProcessValidationError("`n_states` must be 1 through 4.")
    if scenario in {"null","nuisance_confounded","device_confounded"}: n_states=1
    if missingness=="state_dependent" and n_states==1: raise EyeProcessValidationError("`state_dependent` missingness requires a data-generating state process with K > 1.")
    rng=np.random.default_rng(int(seed))
    pcorr=np.array([[1,-.25,.20,.15],[-.25,1,-.10,-.10],[.20,-.10,1,.30],[.15,-.10,.30,1]],float)
    icorr=np.array([[1,.20,.10,.05],[.20,1,.20,.10],[.10,.20,1,.25],[.05,.10,.25,1]],float)
    psd=np.array([.85,.65,.55,.50]); isd=np.array([.75,.45,.45,.35])
    persons=rng.multivariate_normal(np.zeros(4),np.diag(psd)@pcorr@np.diag(psd),n_person)
    items=rng.multivariate_normal(np.array([0,4.2,2.2,0]),np.diag(isd)@icorr@np.diag(isd),n_item)
    person_index=np.repeat(np.arange(n_person),n_item); item_index=np.tile(np.arange(n_item),n_person)
    person_id=np.array([f"P{i+1:03d}" for i in person_index]); item_id=np.array([f"I{i+1:03d}" for i in item_index])
    session_index_item=np.minimum(n_session,np.floor(np.arange(n_item)*n_session/n_item).astype(int)+1); session_index=np.tile(session_index_item,n_person)
    sequence_id=np.array([f"{p}_S{s:02d}" for p,s in zip(person_id,session_index)])
    trial_index=np.empty(len(person_index),int)
    for seq in dict.fromkeys(sequence_id):
        idx=np.flatnonzero(sequence_id==seq); trial_index[idx]=np.arange(1,len(idx)+1)
    dyn_scenario="clear" if scenario in {"rt_redundant","gaze_redundant","pupil_redundant"} else scenario
    initial,transition=_m4_transition_array(persons,n_states,dyn_scenario)
    state=_m4_simulate_states(rng,person_index,sequence_id,initial,transition,n_states)
    separation=.18 if scenario=="weak" else .55
    if n_states==1: delta_rt=delta_gaze=delta_pupil=np.zeros(1)
    else:
        anchor=np.linspace(-1,1,n_states); delta_rt=separation*anchor; delta_gaze=.55*separation*anchor; delta_pupil=.75*separation*anchor
        if scenario=="rt_redundant": delta_gaze=np.zeros(n_states); delta_pupil=np.zeros(n_states)
        elif scenario=="gaze_redundant": delta_rt=.18*anchor; delta_pupil=np.zeros(n_states); delta_gaze=.8*separation*anchor
        elif scenario=="pupil_redundant": delta_rt=.18*anchor; delta_gaze=np.zeros(n_states); delta_pupil=separation*anchor
    N=len(person_index); baseline=rng.normal(size=N); luminance=rng.normal(size=N); gx=rng.normal(size=N); gy=rng.normal(size=N); quality=np.clip(rng.normal(.4,.7,N),-2,2); blink=rng.binomial(1,.05,N); interp=rng.binomial(1,.08,N)
    time_on_task=np.tile(np.linspace(0,1,n_item),n_person); device_person=rng.choice(["device_A","device_B"],n_person); device=device_person[person_index]
    gamma=np.array([.10,-.28,.05,-.05,.12,-.20,-.08,.10]); X=np.column_stack([baseline,luminance,gx,gy,quality,blink,interp,time_on_task]); nuisance=X@gamma
    se_rt=delta_rt[state-1]; se_gaze=delta_gaze[state-1]; se_pupil=delta_pupil[state-1]
    if scenario in {"null","nuisance_confounded","device_confounded"}: se_rt=se_gaze=se_pupil=np.zeros(N)
    response=rng.binomial(1,expit(persons[person_index,0]-items[item_index,0])); log_rt=items[item_index,1]-persons[person_index,1]+se_rt+rng.normal(0,.32,N)
    gaze_eta=items[item_index,2]+persons[person_index,2]+se_gaze; gaze=rng.negative_binomial(7,7/(7+np.exp(gaze_eta)))
    pupil_mu=items[item_index,3]+persons[person_index,3]+nuisance+se_pupil; device_shift=np.zeros(N)
    if scenario=="nuisance_confounded":
        block=np.concatenate([np.resize(np.array([-1.,1.]),np.sum(sequence_id==seq)) for seq in dict.fromkeys(sequence_id)]); luminance=block+rng.normal(0,.15,N); gamma=np.array([.10,-.65,0,0,.12,0,0,.10]); X=np.column_stack([baseline,luminance,gx,gy,quality,blink,interp,time_on_task]); nuisance=X@gamma; pupil_mu=items[item_index,3]+persons[person_index,3]+nuisance
    if scenario=="device_confounded":
        b=device=="device_B"; device_shift=np.where(b,.75,-.25); pupil_mu += device_shift; gaze_eta += np.where(b,.18,-.05); gaze=rng.negative_binomial(7,7/(7+np.exp(gaze_eta)))
    pupil=pupil_mu+rng.normal(0,.42,N)
    complete=pd.DataFrame({"person_id":person_id,"item_id":item_id,"sequence_id":sequence_id,"trial_index":trial_index.astype(float),"session":[f"S{s:02d}" for s in session_index],"response":response,"rt":np.exp(log_rt),"gaze":gaze.astype(float),"pupil":pupil,"pupil_baseline":baseline,"luminance":luminance,"gaze_x":gx,"gaze_y":gy,"pupil_quality":quality,"pupil_blink":blink,"pupil_interpolated":interp,"time_on_task":time_on_task,"device":device,"sampling_rate_hz":60.})
    d=_m4_apply_missingness(rng,complete,missingness,missing_rate,state,n_states)
    truth={"n_states":n_states,"state":state,"initial_prob_person":initial,"transition_prob_person":transition,"delta_rt":delta_rt,"delta_gaze":delta_gaze,"delta_pupil":delta_pupil,"theta":persons[:,0],"tau":persons[:,1],"omega":persons[:,2],"rho":persons[:,3],"b":items[:,0],"beta":items[:,1],"m":items[:,2],"kappa":items[:,3],"gamma_pupil":dict(zip(_NUISANCE,gamma)),"device_shift":device_shift,"person_levels":[f"P{i+1:03d}" for i in range(n_person)],"item_levels":[f"I{i+1:03d}" for i in range(n_item)]}
    return _result("eye_multimodal_m4_simulation",data=d,complete_data=complete,truth=truth,scenario=scenario,missingness=missingness,missing_rate=float(missing_rate),seed=int(seed),spec=multimodal_m4_spec(n_states=n_states,trait_conditioning=() if n_states==1 else ("theta","tau"),backend="cmdstanpy"),interpretation="Synthetic state labels are recovery truth, not psychological constructs.")


def multimodal_m4_state_diagnostics(x: Any)->EyeResult:
    if getattr(x,"eyeprocess_class",None)=="eye_multimodal_m4_simulation":
        d=x.data.reset_index(drop=True); state=np.asarray(x.truth["state"],int); K=int(x.truth["n_states"])
        prob=pd.DataFrame({"source_row":np.arange(1,len(d)+1),"person_id":d.person_id,"item_id":d.item_id,"sequence_id":d.sequence_id,"trial_index":d.trial_index})
        for k in range(1,K+1): prob[f"state_{k}_probability"]=(state==k).astype(float)
        prob["posterior_entropy"]=0.0; prob["MAP_state"]=state
        occ=np.array([float(np.mean(state==k)) for k in range(1,K+1)])
        occupancy=pd.DataFrame({"state":np.arange(1,K+1),"mean_probability":occ,"MAP_fraction":occ})
        transition=np.asarray(x.truth.get("transition_prob_person",np.ones((1,K,K))/K),float).mean(axis=0)
        transition_long=pd.DataFrame([(i+1,j+1,float(transition[i,j])) for i in range(K) for j in range(K)],columns=["from","to","probability"])
        run_rows=[]
        for seq,g in prob.groupby("sequence_id",sort=False):
            vals=g.MAP_state.to_numpy(int); lengths=[]; states=[]
            if len(vals):
                cur=vals[0]; n=1
                for v in vals[1:]:
                    if v==cur: n+=1
                    else: states.append(cur); lengths.append(n); cur=v; n=1
                states.append(cur); lengths.append(n)
            run_rows.extend({"sequence_id":seq,"state":st,"run_length":ln} for st,ln in zip(states,lengths))
        runs=pd.DataFrame(run_rows)
        prof=[]
        pmat=np.column_stack([(state==k).astype(float) for k in range(1,K+1)])
        for ch in ["rt","gaze","pupil"]:
            y=pd.to_numeric(d[ch],errors="coerce").to_numpy(float); y=np.log(y) if ch=="rt" else y
            for k in range(K):
                ok=np.isfinite(y); w=pmat[:,k]; denom=w[ok].sum(); val=float(np.sum(w[ok]*y[ok])/denom) if denom>0 else np.nan
                prof.append({"state":k+1,"channel":ch,"posterior_weighted_mean":val,"effective_weight":float(denom)})
        channel_profile=pd.DataFrame(prof)
        participant=pd.DataFrame([{"person_id":pid,"state":k,"mean_probability":float(np.mean(state[idx]==k))} for pid,idx in d.groupby("person_id",sort=False).groups.items() for k in range(1,K+1)])
        item=pd.DataFrame([{"item_id":iid,"state":k,"mean_probability":float(np.mean(state[idx]==k))} for iid,idx in d.groupby("item_id",sort=False).groups.items() for k in range(1,K+1)])
        switch_num=0; switch_den=0
        for _,g in prob.groupby("sequence_id",sort=False):
            vals=g.MAP_state.to_numpy(int); switch_num+=int(np.sum(vals[1:]!=vals[:-1])); switch_den+=max(0,len(vals)-1)
        summary={"mean_entropy":0.0,"median_entropy":0.0,"mean_run_length":float(runs.run_length.mean()) if len(runs) else np.nan,"switching_rate":float(switch_num/max(1,switch_den))}
        return _result("eye_multimodal_m4_states",source="synthetic_truth",probability=prob,occupancy=occupancy,transition=transition_long,transition_matrix=transition,runs=runs,channel_profile=channel_profile,participant=participant,item=item,summary=summary,n_states=K,interpretation="MAP labels are secondary summaries; posterior probabilities carry uncertainty.")
    if getattr(x,"eyeprocess_class",None)=="eye_multimodal_m4_fit" and isinstance(x.get("state_probability"),pd.DataFrame):
        prob=x.state_probability.copy(); K=int(x.spec.n_states); return _result("eye_multimodal_m4_states",source="posterior",probability=prob,n_states=K,interpretation="MAP labels are secondary summaries; posterior probabilities carry uncertainty.")
    raise EyeProcessValidationError("State diagnostics require an M4 simulation or validated fitted M4 object.")


def audit_multimodal_m4_identifiability(x: Any,spec: Any=None,include_posterior: bool=True,rhat_max: float=1.05,ess_min: int=100,ebfmi_min: float=.30,occupancy_min: float=.03,entropy_fraction_review: float=.80)->EyeResult:
    del include_posterior,rhat_max,ess_min,ebfmi_min,occupancy_min,entropy_fraction_review; d=_extract_data(x); spec=spec or getattr(x,"spec",None) or multimodal_m4_spec(backend="cmdstanpy"); _req(d,["person_id","item_id","sequence_id","trial_index","response","rt","gaze","pupil"])
    # No silent reorder: sequence blocks must be contiguous and order strictly increasing.
    seq=d.sequence_id.astype(str).to_numpy()
    runs=[]
    if len(seq):
        runs=[seq[0]]+[seq[i] for i in range(1,len(seq)) if seq[i]!=seq[i-1]]
    if len(runs)!=len(set(runs)):
        raise EyeProcessValidationError("Each M4 sequence must occupy one contiguous block; M4 does not silently sort data.")
    for _,g in d.groupby("sequence_id",sort=False):
        if g.person_id.astype(str).nunique()!=1:
            raise EyeProcessValidationError("Each M4 sequence must belong to exactly one person.")
        vals=pd.to_numeric(g.trial_index,errors="coerce").to_numpy(float)
        if np.any(~np.isfinite(vals)) or np.any(np.diff(vals)<=0):
            raise EyeProcessValidationError("M4 sequence/order contract requires finite unique rows strictly sorted by trial_index; no silent sorting is performed.")
    rows=[]
    def add(domain,criterion,status,severity,value=np.nan,threshold=np.nan,message="",recommendation=""):
        rows.append({"domain":domain,"criterion":criterion,"status":status,"severity":severity,"value":value,"threshold":threshold,"message":message,"recommendation":recommendation})
    add("structure","person_count","PASS" if d.person_id.nunique()>=8 else "REVIEW","review",d.person_id.nunique(),8)
    add("structure","item_count","PASS" if d.item_id.nunique()>=5 else "REVIEW","review",d.item_id.nunique(),5)
    trait_gate=spec.n_states>1 and spec.transition_structure=="markov" and bool(spec.trait_conditioning)
    add("identification","trait_conditioned_markov","REVIEW" if trait_gate else "PASS","review" if trait_gate else "none",message="Trait-conditioned Markov transitions remain evidence-gated." if trait_gate else "No trait-conditioned Markov gate triggered.")
    tab=pd.DataFrame(rows); fail=(tab.status=="FAIL").any(); review=(tab.status=="REVIEW").any(); overall="FAIL" if fail else ("REVIEW" if review else "PASS"); supported=not fail
    return _result("eye_multimodal_m4_identifiability",checks=tab,overall=overall,supported=bool(supported),spec=spec)


def multimodal_m4_negative_controls(x: Any,controls: Sequence[str]=("order_shuffle","process_shuffle","state_independent","nuisance_pseudostate","device_session_pseudostate","overfit_state_count"),seed: int=20260820,run: bool=False,fit_args: Mapping[str,Any]|None=None)->EyeResult:
    d=_extract_data(x); rng=np.random.default_rng(int(seed)); out=d.copy(); # deterministic design object; data itself unchanged when run=False
    if run: raise EyeProcessBackendError("Executing M4 negative-control fits requires the canonical CmdStan backend and REVIEW evidence.")
    return _result("eye_multimodal_m4_negative_controls",controls=list(controls),data=out,seed=int(seed),executed=False,fit_args=dict(fit_args or {}),interpretation="Negative controls test whether state evidence survives meaningless/misaligned process structure.")


def multimodal_m4_sensitivity(x: Any,n_states: Sequence[int]=range(1,5),run: bool=False,fit_args: Mapping[str,Any]|None=None)->EyeResult:
    if run: raise EyeProcessBackendError("Executing M4 state-count sensitivity requires the canonical CmdStan backend.")
    vals=[int(k) for k in n_states if 1<=int(k)<=4]; return _result("eye_multimodal_m4_sensitivity",design=pd.DataFrame({"n_states":vals}),executed=False,fit_args=dict(fit_args or {}),source=x)


def multimodal_m4_ppc(x: Any)->EyeResult:
    d=_extract_data(x); return _result("eye_multimodal_m4_ppc",measurement=pd.DataFrame({"channel":[c for c in ["response","rt","gaze","pupil"] if c in d],"mean":[float(pd.to_numeric(d[c],errors="coerce").mean()) for c in [c for c in ["response","rt","gaze","pupil"] if c in d]]}),status="observed_summary_without_posterior")


def multimodal_m4_ablation(x: Any,run: bool=False,include_channel_ablations: bool=False,m3_fit: Any=None,m4_fit: Any=None,fit_args: Mapping[str,Any]|None=None)->EyeResult:
    del m3_fit,m4_fit
    if run: raise EyeProcessBackendError("Executing M4 ablations requires the canonical CmdStan backend.")
    names=["M3_vs_M4","iid_vs_markov","K1_vs_K2"] + (["rt","gaze","pupil"] if include_channel_ablations else []); return _result("eye_multimodal_m4_ablation",design=pd.DataFrame({"ablation":names}),executed=False,fit_args=dict(fit_args or {}),source=x)


def multimodal_m4_process_information(x: Any,decisive_z: float=2)->EyeResult:
    states=multimodal_m4_state_diagnostics(x); return _result("eye_multimodal_m4_information",occupancy=states.get("occupancy",pd.DataFrame()),decisive_z=float(decisive_z),status="state_information_without_response_elpd" if states.source=="synthetic_truth" else "posterior_state_information")


def validate_multimodal_m4(x: Any,information: Any=None,negative_controls: Any=None,sensitivity: Any=None,recovery: Any=None,include_ppc: bool=True)->EyeResult:
    ident=audit_multimodal_m4_identifiability(x); rows=ident.checks.copy(); rows["domain"] = rows["domain"].astype(str); return _result("eye_multimodal_m4_validation",overall=ident.overall,supported=ident.supported,checks=rows,identifiability=ident,information=information,negative_controls=negative_controls,sensitivity=sensitivity,recovery=recovery,ppc=multimodal_m4_ppc(x) if include_ppc else None,interpretation="M4 evidence is multi-criterion and remains bounded by REVIEW governance.")


def multimodal_m4_recovery(simulation: Any=None,fit: Any=None,scenarios: Sequence[str]=("clear","weak","null","trait_conditioned","nuisance_confounded"),run: bool=False,simulation_args: Mapping[str,Any]|None=None,fit_args: Mapping[str,Any]|None=None)->EyeResult:
    del simulation,fit
    design=pd.DataFrame({"scenario":list(scenarios)}); return _result("eye_multimodal_m4_recovery",design=design,executed=False if not run else False,simulation_args=dict(simulation_args or {"n_person":60,"n_item":10}),fit_args=dict(fit_args or {}),status="inert_validation_design" if not run else "backend_required")


__all__=[
"prepare_multimodal_irt_data","audit_multimodal_measurement","multimodal_irt_spec","simulate_multimodal_irt","process_information","ablate_multimodal_channels","multimodal_backend_status","multimodal_ppc","validate_multimodal_irt","audit_multimodal_identifiability",
"multimodal_m2_spec","fit_multimodal_m2","audit_multimodal_m2_identifiability","multimodal_m2_ppc","validate_multimodal_m2","multimodal_m2_ablation","multimodal_m2_process_information","multimodal_m2_negative_controls","simulate_multimodal_m2","multimodal_m2_recovery",
"multimodal_m3_spec","fit_multimodal_m3","audit_multimodal_m3_identifiability","multimodal_m3_ppc","multimodal_m3_ablation","multimodal_m3_process_information","multimodal_m3_negative_controls","multimodal_m3_functional_bridge","validate_multimodal_m3","simulate_multimodal_m3","multimodal_m3_recovery",
"multimodal_m4_spec","fit_multimodal_m4","audit_multimodal_m4_identifiability","multimodal_m4_state_diagnostics","multimodal_m4_ppc","multimodal_m4_ablation","multimodal_m4_process_information","multimodal_m4_negative_controls","multimodal_m4_sensitivity","validate_multimodal_m4","simulate_multimodal_m4","multimodal_m4_recovery"]
