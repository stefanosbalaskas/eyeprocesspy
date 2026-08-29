"""Advanced process-IRT integrations from frozen eyeprocess 0.7-era sources.

Dependency-light procedures are source translations. Where the R reference
requires an R-only optional engine, eyeprocesspy either preserves an explicit
external-engine gate or provides a clearly labelled Python reference estimator.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.spatial.distance import pdist
from scipy.stats import rankdata, spearmanr

from .exceptions import EyeProcessBackendError, EyeProcessModelError, EyeProcessValidationError
from .irt import EyeResult, _as_df, _req_cols, _result, _tag
from .process_irt_07 import (
    _REGISTRY,
    _dummy_design,
    _logistic_fit,
    _ols_fit,
    _register_builtins,
)

__all__ = [
    "fit_process_hmm_irt", "process_state_occupancy", "process_state_transition_summary",
    "fit_cognitive_diagnosis_process", "fit_latent_class_process_irt",
    "fit_crossclassified_process_irt", "fit_latent_space_irt", "process_residual_map",
    "validate_latent_space_process_similarity", "equate_irt_scales", "process_person_fit",
    "process_dif_nuisance_surrogate", "audit_process_adjusted_dif", "process_ngram_features",
    "process_sequence_embedding", "fit_response_process_embedding_irt", "fit_gpirt",
    "compare_parametric_nonparametric_irf", "audit_irf_shape", "fit_dynamic_gpirt",
    "fit_continuous_time_irt", "latent_trait_trajectory", "predict_theta_at_time",
    "fit_flow_mirt", "fit_variational_irt", "process_item_information",
    "expected_process_information", "select_next_item_process", "simulate_process_cat",
    "validate_latent_space_process_similarity",
]

_EPS = np.finfo(float).eps


def _df(data: Any, cols: Sequence[str] = (), name: str = "data") -> pd.DataFrame:
    d = _as_df(data, name)
    _req_cols(d, [c for c in cols if c], name)
    return d


def _zscore(X: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, float)
    center = np.nanmean(X, axis=0)
    scale = np.nanstd(X, axis=0, ddof=1)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    Z = (X - center) / scale
    Z[~np.isfinite(Z)] = 0.0
    return Z, center, scale


def _hmm_fb(x: np.ndarray, pi: np.ndarray, A: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    T, K = x.shape[0], len(pi)
    logB = np.empty((T, K), float)
    for k in range(K):
        s = np.maximum(sd[k], 1e-6)
        logB[:, k] = -0.5 * np.sum(((x - mu[k]) / s) ** 2 + 2 * np.log(s) + math.log(2 * math.pi), axis=1)
    la = np.full((T, K), -np.inf)
    la[0] = np.log(np.maximum(pi, 1e-300)) + logB[0]
    for t in range(1, T):
        for k in range(K):
            la[t, k] = logB[t, k] + logsumexp(la[t - 1] + np.log(np.maximum(A[:, k], 1e-300)))
    ll = float(logsumexp(la[-1]))
    lb = np.zeros((T, K), float)
    for t in range(T - 2, -1, -1):
        for j in range(K):
            lb[t, j] = logsumexp(np.log(np.maximum(A[j], 1e-300)) + logB[t + 1] + lb[t + 1])
    gamma = np.exp(la + lb - ll)
    xi = np.zeros((max(T - 1, 0), K, K), float)
    for t in range(T - 1):
        z = la[t, :, None] + np.log(np.maximum(A, 1e-300)) + (logB[t + 1] + lb[t + 1])[None, :] - ll
        xi[t] = np.exp(z)
    return ll, gamma, xi


def fit_process_hmm_irt(data: Any, sequence_id: str = "trial_id", order: str = "timestamp",
                        process_features: Sequence[str] = ("x", "y"), response: str = "response",
                        person: str = "participant_id", item: str = "item_id", n_states: int = 3,
                        max_iter: int = 100, tol: float = 1e-5, seed: int = 1) -> EyeResult:
    n_states = int(n_states)
    if n_states < 2:
        raise EyeProcessValidationError("n_states must be at least 2.")
    feats = list(process_features)
    d = _df(data, list(dict.fromkeys([sequence_id, order, *feats, response, person, item]))).copy()
    d = d.dropna(subset=[sequence_id, order, *feats]).reset_index(drop=True)
    if len(d) < n_states:
        raise EyeProcessValidationError("Too few complete rows for the requested number of states.")
    X, center, scale = _zscore(d[feats].to_numpy(float))
    rng = np.random.default_rng(seed)
    # scipy kmeans2 is deterministic with an explicit generator.
    centers, labels = kmeans2(X, n_states, iter=25, minit="++", seed=rng)
    mu = np.asarray(centers, float)
    global_sd = np.nanstd(X, axis=0, ddof=1)
    global_sd[~np.isfinite(global_sd) | (global_sd < 1e-4)] = 1.0
    sds = np.repeat(global_sd[None, :], n_states, axis=0)
    pi = np.repeat(1 / n_states, n_states)
    A = np.ones((n_states, n_states), float)
    seqs = {k: np.asarray(v, int) for k, v in d.groupby(sequence_id, sort=False).groups.items()}
    for ids in seqs.values():
        ids = ids[np.argsort(d.loc[ids, order].to_numpy())]
        cl = labels[ids]
        for a, b in zip(cl[:-1], cl[1:]):
            A[a, b] += 1
    A /= A.sum(axis=1, keepdims=True)
    gamma_all = np.zeros((len(d), n_states), float)
    history: list[float] = []
    for _ in range(int(max_iter)):
        pi_num = np.zeros(n_states); A_num = np.zeros_like(A)
        mu_num = np.zeros_like(mu); second_num = np.zeros_like(mu); mu_den = np.zeros(n_states)
        ll_total = 0.0
        for ids0 in seqs.values():
            ids = ids0[np.argsort(d.loc[ids0, order].to_numpy())]
            ll, gam, xi = _hmm_fb(X[ids], pi, A, mu, sds)
            ll_total += ll; gamma_all[ids] = gam; pi_num += gam[0]
            if len(ids) > 1:
                A_num += xi.sum(axis=0)
            for k in range(n_states):
                w = gam[:, k]
                mu_num[k] += np.sum(X[ids] * w[:, None], axis=0)
                second_num[k] += np.sum((X[ids] ** 2) * w[:, None], axis=0)
                mu_den[k] += w.sum()
        pi = (pi_num + 1e-6) / np.sum(pi_num + 1e-6)
        A = (A_num + 1e-6); A /= A.sum(axis=1, keepdims=True)
        mu = mu_num / np.maximum(mu_den[:, None], 1e-8)
        var = second_num / np.maximum(mu_den[:, None], 1e-8) - mu ** 2
        sds = np.sqrt(np.maximum(var, 1e-4))
        history.append(float(ll_total))
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            break
    state = np.argmax(gamma_all, axis=1) + 1
    d[".process_state"] = state
    rows = []
    for sid, ids in seqs.items():
        row = {sequence_id: sid}
        p = gamma_all[ids].mean(axis=0)
        row.update({f"state_{k+1}_occupancy": float(p[k]) for k in range(n_states)})
        rows.append(row)
    occupancy = pd.DataFrame(rows)
    meta = d.drop_duplicates(sequence_id)[[sequence_id, response, person, item]]
    summary_data = meta.merge(occupancy, on=sequence_id, how="left", sort=False)
    response_fit = None
    y = pd.to_numeric(summary_data[response], errors="coerce")
    if y.dropna().isin([0, 1]).all() and y.notna().sum() >= 2:
        occ_cols = [c for c in occupancy if c.endswith("_occupancy")][:-1]
        Xr, names = _dummy_design(summary_data.dropna(subset=[response]), continuous=occ_cols, categorical=[person, item])
        response_fit = _logistic_fit(Xr, summary_data.dropna(subset=[response])[response].to_numpy(float))
        response_fit["design_names"] = names
    return _result("eye_process_hmm_irt", pi=pi, transition=A, means=mu, sds=sds,
                   posterior_state=gamma_all, state=state, row_data=d, occupancy=occupancy,
                   summary_data=summary_data, response_model=response_fit,
                   logLik=history[-1] if history else math.nan, logLik_history=np.asarray(history),
                   scaling={"center": center, "scale": scale}, process_features=feats,
                   n_states=n_states, status="experimental-two-stage",
                   note="States are statistical process states and must not be assigned psychological labels without independent validation.")


def process_state_occupancy(object: Any) -> pd.DataFrame:
    if getattr(object, "eyeprocess_class", None) != "eye_process_hmm_irt":
        raise EyeProcessValidationError("object must be an eye_process_hmm_irt.")
    return object.occupancy.copy()


def process_state_transition_summary(object: Any) -> pd.DataFrame:
    if getattr(object, "eyeprocess_class", None) != "eye_process_hmm_irt":
        raise EyeProcessValidationError("object must be an eye_process_hmm_irt.")
    A = np.asarray(object.transition, float)
    rows = [{"from_state": i + 1, "to_state": j + 1, "probability": float(A[i, j])}
            for i in range(A.shape[0]) for j in range(A.shape[1])]
    return pd.DataFrame(rows)


def fit_cognitive_diagnosis_process(response_matrix: Any, q_matrix: Any, process_data: Any = None,
                                    process_features: Sequence[str] | None = None, person_id: str | None = None,
                                    engine: str = "GDINA", external_engine: Callable[..., Any] | None = None,
                                    **kwargs: Any) -> EyeResult:
    X = np.asarray(response_matrix, float); Q = np.asarray(q_matrix, float)
    if X.ndim != 2 or Q.ndim != 2 or X.shape[1] != Q.shape[0]:
        raise EyeProcessValidationError("Response-matrix columns must match Q-matrix rows.")
    if engine == "GDINA":
        raise EyeProcessBackendError("The exact R GDINA engine is not a Python core dependency; supply external_engine for a validated cognitive-diagnosis backend.")
    if engine != "external" or not callable(external_engine):
        raise EyeProcessBackendError("Supply external_engine as a validated function.")
    response_fit = external_engine(response_matrix=X, q_matrix=Q, **kwargs)
    process_summary = None
    if process_data is not None and process_features:
        if person_id is None:
            raise EyeProcessValidationError("person_id is required when process_data is supplied.")
        d = _df(process_data, [person_id, *process_features])
        agg = d.groupby(person_id, sort=False)[list(process_features)].mean().reset_index()
        Z, _, _ = _zscore(agg[list(process_features)].to_numpy(float))
        if Z.shape[1] > 1:
            u, s, _ = np.linalg.svd(Z, full_matrices=False); pc = u[:, 0] * s[0]
        else:
            pc = Z[:, 0]
        process_summary = pd.DataFrame({"person_id": agg[person_id].to_numpy(), "process_surrogate": pc})
    return _result("eye_cognitive_diagnosis_process", response_model=response_fit,
                   process_summary=process_summary, process_mastery_correlation=None,
                   q_matrix=Q, status="experimental-integration")


def fit_latent_class_process_irt(data: Any, response: str = "response", process_features: Sequence[str] = (),
                                 person: str = "participant_id", item: str = "item_id", n_classes: int = 2,
                                 seed: int = 1) -> EyeResult:
    feats = list(process_features)
    if not feats:
        raise EyeProcessValidationError("process_features must not be empty.")
    d = _df(data, [response, *feats, person, item]).copy()
    Z, center, scale = _zscore(d[feats].to_numpy(float))
    centers, labels = kmeans2(Z, int(n_classes), iter=25, minit="++", seed=np.random.default_rng(seed))
    d[".process_class"] = labels + 1
    ok = d[response].notna()
    response_fit = None
    if ok.any() and pd.to_numeric(d.loc[ok, response], errors="coerce").isin([0, 1]).all():
        Xr, names = _dummy_design(d.loc[ok], categorical=[".process_class", person, item])
        response_fit = _logistic_fit(Xr, pd.to_numeric(d.loc[ok, response]).to_numpy(float)); response_fit["design_names"] = names
    return _result("eye_latent_class_process_irt", response_model=response_fit,
                   class_=pd.Categorical(d[".process_class"]), centers=np.asarray(centers), data=d,
                   process_features=feats, scaling={"center": center, "scale": scale},
                   status="experimental-two-stage")


def fit_crossclassified_process_irt(data: Any, outcome: str, person: str = "participant_id",
                                    item: str = "item_id", context: str | None = None,
                                    family: str = "gaussian", fixed: Sequence[str] | str | None = None) -> EyeResult:
    if family not in {"gaussian", "binomial", "poisson", "negative_binomial"}:
        raise EyeProcessValidationError("family must be gaussian, binomial, poisson, or negative_binomial.")
    fixeds = [] if fixed is None else ([fixed] if isinstance(fixed, str) else list(fixed))
    cats = [person, item] + ([context] if context else [])
    d = _df(data, [outcome, *cats, *fixeds]).dropna(subset=[outcome]).copy()
    continuous = [c for c in fixeds if pd.api.types.is_numeric_dtype(d[c])]
    categorical = cats + [c for c in fixeds if c not in continuous]
    X, names = _dummy_design(d, continuous=continuous, categorical=categorical)
    y = pd.to_numeric(d[outcome], errors="coerce").to_numpy(float)
    if family == "gaussian":
        fit = _ols_fit(X, y)
    elif family == "binomial":
        fit = _logistic_fit(X, y)
    else:
        # Auditable log-count reference in lieu of lme4 GLMM/negative-binomial random effects.
        fit = _ols_fit(X, np.log1p(np.maximum(y, 0)))
    fit["design_names"] = names
    return _result("eye_crossclassified_process_irt", model=fit, family=family, person=person,
                   item=item, context=context, status="python-reference-estimator",
                   algorithmic_parity=False)


def fit_latent_space_irt(response_matrix: Any, dimensions: int = 2, penalty: float | None = None,
                         constraint: float | None = None, starts: Any = None, tol: float = 1e-3,
                         silent: bool = True) -> EyeResult:
    raise EyeProcessBackendError("Latent-space IRT parity requires the R LSMjml engine or a separately validated Python backend; no silent approximation is used.")


def process_residual_map(object: Any, entity: str = "both") -> pd.DataFrame:
    if getattr(object, "eyeprocess_class", None) != "eye_latent_space_irt":
        raise EyeProcessValidationError("object must be an eye_latent_space_irt.")
    if entity not in {"both", "person", "item"}:
        raise EyeProcessValidationError("entity must be both, person, or item.")
    p = pd.DataFrame(np.asarray(object.person_coordinates)); p["entity_id"] = list(getattr(object, "person_ids", range(1, len(p)+1))); p["entity_type"] = "person"
    i = pd.DataFrame(np.asarray(object.item_coordinates)); i["entity_id"] = list(getattr(object, "item_ids", range(1, len(i)+1))); i["entity_type"] = "item"
    if entity == "person": return p
    if entity == "item": return i
    return pd.concat([p, i], ignore_index=True)


def validate_latent_space_process_similarity(object: Any, process_matrix: Any, entity: str = "person") -> EyeResult:
    if getattr(object, "eyeprocess_class", None) != "eye_latent_space_irt":
        raise EyeProcessValidationError("object must be an eye_latent_space_irt.")
    coord = np.asarray(object.person_coordinates if entity == "person" else object.item_coordinates, float)
    P = np.asarray(process_matrix, float)
    if P.ndim != 2 or len(P) != len(coord):
        raise EyeProcessValidationError("process_matrix rows must align with selected latent-space entities.")
    C, _, _ = _zscore(coord); Z, _, _ = _zscore(P)
    dl, dp = pdist(C), pdist(Z)
    rho = float(spearmanr(dl, dp, nan_policy="omit").statistic) if dl.size else math.nan
    return _result("eye_latent_space_process_validation", entity=entity,
                   spearman_distance_correlation=rho, latent_distance=dl, process_distance=dp,
                   interpretation="Positive distance association supports convergent structure but is not proof of a shared construct.")


def _icc(theta: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return expit(theta[:, None] * a[None, :] - (a * b)[None, :])


def equate_irt_scales(reference: Any, new: Any, method: str = "stocking-lord",
                      theta_grid: Sequence[float] | None = None) -> EyeResult:
    ref, nw = _as_df(reference, "reference"), _as_df(new, "new")
    _req_cols(ref, ["a", "b"], "reference"); _req_cols(nw, ["a", "b"], "new")
    if len(ref) != len(nw): raise EyeProcessValidationError("Reference and new anchor sets must contain the same number of items.")
    if method not in {"stocking-lord", "haebara", "mean-sigma", "mean-mean"}:
        raise EyeProcessValidationError("Unsupported linking method.")
    ra, rb = ref.a.to_numpy(float), ref.b.to_numpy(float); na, nb = nw.a.to_numpy(float), nw.b.to_numpy(float)
    if method == "mean-sigma":
        A = np.std(rb, ddof=1) / np.std(nb, ddof=1); B = np.mean(rb) - A * np.mean(nb)
    elif method == "mean-mean":
        A = np.mean(na) / np.mean(ra); B = np.mean(rb) - A * np.mean(nb)
    else:
        th = np.linspace(-4, 4, 81) if theta_grid is None else np.asarray(theta_grid, float)
        pref = _icc(th, ra, rb)
        def obj(par: np.ndarray) -> float:
            A0, B0 = math.exp(par[0]), par[1]
            pnew = _icc(th, na / A0, A0 * nb + B0)
            return float(np.sum((pref.sum(1)-pnew.sum(1))**2) if method=="stocking-lord" else np.sum((pref-pnew)**2))
        opt = minimize(obj, np.array([0.0, 0.0]), method="BFGS")
        A, B = math.exp(float(opt.x[0])), float(opt.x[1])
    transformed = nw.copy(); transformed["a"] = na / A; transformed["b"] = A * nb + B
    return _result("eye_irt_equating", A=float(A), B=float(B), method=method, transformed=transformed,
                   reference=ref, new=nw, equation="theta_reference = A * theta_new + B")


def process_person_fit(object: Any, data: Any = None, person: str | None = None,
                       response_weight: float = 1, rt_weight: float = 1, process_weight: float = 1) -> pd.DataFrame:
    if getattr(object, "eyeprocess_class", None) != "eye_joint_gaze_rt_irt":
        raise EyeProcessValidationError("Currently supports eye_joint_gaze_rt_irt objects.")
    if data is None: raise EyeProcessValidationError("Supply the data used for the fitted reference model.")
    cols = object.columns; person = person or cols["person"]
    d = _df(data, list(cols.values())).dropna(subset=list(cols.values())).copy()
    y = pd.to_numeric(d[cols["response"]]).to_numpy(float); rt = pd.to_numeric(d[cols["rt"]]).to_numpy(float); g = pd.to_numeric(d[cols["gaze"]]).to_numpy(float)
    # Reconstruct the same fixed-effect designs used by the Python reference fit.
    X, _ = _dummy_design(d, categorical=[cols["person"], cols["item"]])
    p = np.asarray(object.response_model.fitted, float)
    if len(p) != len(d): p = expit(X @ np.asarray(object.response_model.coefficients, float))
    r_resp = (y-p)/np.sqrt(np.maximum(p*(1-p), 1e-6))
    rt_pred = np.asarray(object.rt_model.fitted, float); rr=np.log(rt)-rt_pred; s=np.std(rr,ddof=1); r_rt=(rr-np.mean(rr))/(s if np.isfinite(s) and s>0 else 1)
    gaze_pred = np.expm1(np.asarray(object.gaze_model.fitted, float)); r_gaze=(g-gaze_pred)/np.sqrt(np.maximum(gaze_pred,1e-6))
    den=response_weight+rt_weight+process_weight
    row_score=np.sqrt((response_weight*r_resp**2+rt_weight*r_rt**2+process_weight*r_gaze**2)/den)
    work=pd.DataFrame({"participant_id":d[person].astype(str),"response_r":r_resp,"rt_r":r_rt,"process_r":r_gaze,"score":row_score})
    out=work.groupby("participant_id",sort=False).agg(n=("score","size"),response_rms=("response_r",lambda x:float(np.sqrt(np.mean(x*x)))),rt_rms=("rt_r",lambda x:float(np.sqrt(np.mean(x*x)))),process_rms=("process_r",lambda x:float(np.sqrt(np.mean(x*x)))),combined_rms=("score",lambda x:float(np.sqrt(np.mean(x*x))))).reset_index()
    out["empirical_percentile"] = rankdata(out.combined_rms, method="average")/len(out)
    return _tag(out,"eye_process_person_fit")


def process_dif_nuisance_surrogate(data: Any, process_features: Sequence[str], person: str = "participant_id", aggregate: bool = True) -> pd.DataFrame:
    feats=list(process_features); d=_df(data,[person,*feats])
    z=d.groupby(person,sort=False)[feats].mean().reset_index() if aggregate else d[[person,*feats]].copy()
    Z,_,_=_zscore(z[feats].to_numpy(float)); pc=Z[:,0] if Z.shape[1]==1 else np.linalg.svd(Z,full_matrices=False)[0][:,0]*np.linalg.svd(Z,full_matrices=False)[1][0]
    return pd.DataFrame({"person_id":z[person].to_numpy(),"process_nuisance":pc})


def audit_process_adjusted_dif(data: Any, response: str = "response", ability: str | None = None,
                               group: str | None = None, item: str = "item_id", process_features: Sequence[str] = (),
                               person: str = "participant_id") -> EyeResult:
    if not ability or not group: raise EyeProcessValidationError("ability and group must name columns.")
    d=_df(data,[response,ability,group,item,person,*process_features]).copy()
    sur=process_dif_nuisance_surrogate(d,process_features,person=person,aggregate=True).rename(columns={"person_id":person})
    d=d.merge(sur,on=person,how="left",sort=False)
    # Fixed-effect Python diagnostic matching the R nuisance-control intent.
    X0,n0=_dummy_design(d,continuous=[ability],categorical=[group,item]); X1,n1=_dummy_design(d,continuous=[ability,"process_nuisance"],categorical=[group,item])
    y=pd.to_numeric(d[response],errors="coerce").to_numpy(float); m0=_logistic_fit(X0,y); m1=_logistic_fit(X1,y)
    # Report coefficients that are common after process adjustment; interaction-specific R coefficients are backend-dependent.
    c0=dict(zip(n0,np.asarray(m0.coefficients,float))); c1=dict(zip(n1,np.asarray(m1.coefficients,float))); names=sorted(set(c0)&set(c1))
    tab=pd.DataFrame({"term":names,"unadjusted":[c0[n] for n in names],"adjusted":[c1[n] for n in names]}); tab["absolute_reduction"]=np.abs(tab.unadjusted)-np.abs(tab.adjusted)
    return _result("eye_process_adjusted_dif",unadjusted_model=m0,adjusted_model=m1,coefficients=tab,surrogate=sur,status="python-reference-estimator",algorithmic_parity=False,note="Process adjustment is a diagnostic nuisance-control analysis; causal explanations of DIF require study-specific evidence.")


def _seqs(sequence: Any, separator: str) -> list[list[str]]:
    if isinstance(sequence,str): return [sequence.split(separator)]
    if isinstance(sequence,(list,tuple)) and sequence and isinstance(sequence[0],(list,tuple,np.ndarray,pd.Series)): return [[str(x) for x in s] for s in sequence]
    return [[str(x) for x in sequence]]


def process_ngram_features(sequence: Any, n: Sequence[int] = (1,2,3), separator: str = ">") -> np.ndarray:
    seqs=_seqs(sequence,separator); orders=sorted({int(k) for k in n if int(k)>0}); counts=[]; vocab=set()
    for s in seqs:
        c=Counter()
        for k in orders:
            for i in range(max(0,len(s)-k+1)): c[separator.join(s[i:i+k])]+=1
        counts.append(c); vocab.update(c)
    vocab=sorted(vocab); M=np.zeros((len(seqs),len(vocab)),float)
    for i,c in enumerate(counts):
        for j,g in enumerate(vocab): M[i,j]=c.get(g,0)
    # preserve language-neutral metadata through ndarray attributes is impossible; return subclass not needed for parity tests.
    return M


def process_sequence_embedding(sequence: Any, n: Sequence[int]=(1,2,3), dimensions: int=5) -> np.ndarray:
    X=process_ngram_features(sequence,n=n)
    if X.shape[1]==0: raise EyeProcessValidationError("No n-grams could be constructed.")
    tf=X/np.maximum(X.sum(axis=1,keepdims=True),1); idf=np.log((len(X)+1)/(np.sum(X>0,axis=0)+1))+1; Z=tf*idf
    u,s,_=np.linalg.svd(Z,full_matrices=False); k=min(int(dimensions),len(s)); return u[:,:k]*s[:k]


def fit_response_process_embedding_irt(data: Any, sequences: Any, response: str="response", person: str="participant_id", item: str="item_id", dimensions: int=5, n: Sequence[int]=(1,2,3)) -> EyeResult:
    d=_df(data,[response,person,item]).copy(); emb=process_sequence_embedding(sequences,n=n,dimensions=dimensions)
    if len(emb)!=len(d): raise EyeProcessValidationError("sequences must align row-for-row with data.")
    dd=d.copy(); emb_cols=[f"process_embedding_{i+1}" for i in range(emb.shape[1])]
    for i,c in enumerate(emb_cols): dd[c]=emb[:,i]
    X,names=_dummy_design(dd,continuous=emb_cols,categorical=[person,item]); fit=_logistic_fit(X,pd.to_numeric(dd[response]).to_numpy(float)); fit["design_names"]=names
    return _result("eye_response_process_embedding_irt",model=fit,embedding=emb,data=dd,status="experimental-feature-integration",algorithmic_parity=False)


def _poly_design(theta: np.ndarray, degree: int=3) -> np.ndarray:
    return np.column_stack([np.ones(len(theta))]+[theta**k for k in range(1,degree+1)])


def fit_gpirt(response_matrix: Any, engine: str="spline_reference", external_engine: Callable[...,Any]|None=None, spline_df: int=5, **kwargs: Any) -> EyeResult:
    X=np.asarray(response_matrix,float)
    if X.ndim!=2: raise EyeProcessValidationError("response_matrix must be two-dimensional.")
    if engine=="external":
        if not callable(external_engine): raise EyeProcessBackendError("Supply a validated GPIRT fitter through external_engine.")
        return _result("eye_gpirt",model=external_engine(response_matrix=X,**kwargs),engine="external",exact_gpirt=True,status="experimental-external")
    if engine!="spline_reference": raise EyeProcessValidationError("engine must be spline_reference or external.")
    p=np.nanmean(X,axis=1); nn=np.sum(np.isfinite(X),axis=1); adj=(p*nn+.5)/(nn+1); theta=np.log(np.clip(adj,1e-5,1-1e-5)/(1-np.clip(adj,1e-5,1-1e-5)))
    models=[]
    for j in range(X.shape[1]):
        ok=np.isfinite(X[:,j]); D=_poly_design(theta[ok],degree=max(2,min(5,int(spline_df)-1))); fit=_logistic_fit(D,X[ok,j]); fit["theta_degree"]=D.shape[1]-1; models.append(fit)
    names=[f"item_{i+1}" for i in range(X.shape[1])]
    return _result("eye_gpirt",response_matrix=X,models=models,item_names=names,theta_proxy=theta,engine="spline_reference",exact_gpirt=False,status="experimental-model-criticism",algorithmic_parity=False,note="Polynomial-flexible IRFs are a Python shape-audit surrogate; not a Gaussian-process IRT estimator.")


def compare_parametric_nonparametric_irf(response_matrix: Any, gpirt_object: Any=None, theta_grid: Sequence[float]|None=None) -> pd.DataFrame:
    X=np.asarray(response_matrix,float); obj=fit_gpirt(X) if gpirt_object is None else gpirt_object
    if getattr(obj,"eyeprocess_class",None)!="eye_gpirt" or obj.engine!="spline_reference": raise EyeProcessValidationError("This comparison requires a spline-reference eye_gpirt object.")
    grid=np.linspace(-4,4,101) if theta_grid is None else np.asarray(theta_grid,float); rows=[]
    for j in range(X.shape[1]):
        ok=np.isfinite(X[:,j]); th=obj.theta_proxy[ok]; y=X[ok,j]; lin=_logistic_fit(np.column_stack([np.ones(len(th)),th]),y)
        pp=expit(np.column_stack([np.ones(len(grid)),grid]) @ np.asarray(lin.coefficients,float)); deg=int(obj.models[j].theta_degree); pf=expit(_poly_design(grid,deg) @ np.asarray(obj.models[j].coefficients,float))
        rows.extend({"item":obj.item_names[j],"theta":float(t),"parametric":float(a),"flexible":float(b),"absolute_difference":float(abs(a-b))} for t,a,b in zip(grid,pp,pf))
    return _tag(pd.DataFrame(rows),"eye_irf_comparison")


def audit_irf_shape(comparison: Any, mean_absolute_threshold: float=.05, max_absolute_threshold: float=.15) -> pd.DataFrame:
    d=_as_df(comparison,"comparison"); _req_cols(d,["item","absolute_difference"],"comparison")
    out=d.groupby("item",sort=False).absolute_difference.agg(mean_absolute_difference="mean",max_absolute_difference="max").reset_index(); out["flag"]=(out.mean_absolute_difference>mean_absolute_threshold)|(out.max_absolute_difference>max_absolute_threshold); return out


def _external_gate(name: str, message: str, data_kw: str, data: Any, external_engine: Callable[...,Any]|None, kwargs: dict[str,Any], cls: str) -> EyeResult:
    if not callable(external_engine): raise EyeProcessBackendError(message)
    return _result(cls,model=external_engine(**{data_kw:data},**kwargs),engine="external",status="experimental-gated")


def fit_dynamic_gpirt(data: Any, external_engine: Callable[...,Any]|None=None, **kwargs: Any) -> EyeResult:
    return _external_gate("fit_dynamic_gpirt","Dynamic GPIRT is experimental; supply a validated external_engine.","data",data,external_engine,kwargs,"eye_dynamic_gpirt")

def fit_continuous_time_irt(data: Any, external_engine: Callable[...,Any]|None=None, **kwargs: Any) -> EyeResult:
    return _external_gate("fit_continuous_time_irt","Continuous-time IRT requires a validated external_engine.","data",data,external_engine,kwargs,"eye_continuous_time_irt")

def fit_flow_mirt(response_matrix: Any, external_engine: Callable[...,Any]|None=None, **kwargs: Any) -> EyeResult:
    return _external_gate("fit_flow_mirt","Flow-MIRT remains experimental; supply a validated external_engine.","response_matrix",response_matrix,external_engine,kwargs,"eye_flow_mirt")

def fit_variational_irt(response_matrix: Any, external_engine: Callable[...,Any]|None=None, **kwargs: Any) -> EyeResult:
    return _external_gate("fit_variational_irt","Supply a validated variational IRT external_engine; no silent approximation is used.","response_matrix",response_matrix,external_engine,kwargs,"eye_variational_irt")


def latent_trait_trajectory(time: Any, theta: Any, spar: float|None=None) -> EyeResult:
    t=np.asarray(time,float); th=np.asarray(theta,float); ok=np.isfinite(t)&np.isfinite(th)
    if ok.sum()<4: raise EyeProcessValidationError("At least four finite time/theta pairs are required.")
    ord_=np.argsort(t[ok]); tx=t[ok][ord_]; ty=th[ok][ord_]
    # scipy's smoothing spline is not numerically identical to R smooth.spline; mark it.
    s=None if spar is None else float(spar)*len(tx)*np.var(ty)
    fit=UnivariateSpline(tx,ty,s=s)
    return _result("eye_latent_trait_trajectory",model=fit,time=tx,theta=ty,status="python-reference-estimator",algorithmic_parity=False)


def predict_theta_at_time(object: Any, time: Any) -> pd.DataFrame:
    if getattr(object,"eyeprocess_class",None)!="eye_latent_trait_trajectory": raise EyeProcessValidationError("object must come from latent_trait_trajectory().")
    t=np.asarray(time,float); return pd.DataFrame({"x":t,"y":object.model(t)})


def process_item_information(theta: Any, a: Any, b: Any, process_information: Any=0, rt_information: Any=0,
                             weights: Mapping[str,float]|Sequence[float] = {"response":1,"rt":0,"process":0},
                             expected_time: Any=0, burden_weight: float=0) -> EyeResult:
    th=np.atleast_1d(np.asarray(theta,float)); aa=np.atleast_1d(np.asarray(a,float)); bb=np.atleast_1d(np.asarray(b,float))
    if len(aa)!=len(bb): raise EyeProcessValidationError("a and b must have equal length.")
    p=expit(th[:,None]*aa[None,:]-(aa*bb)[None,:]); resp=p*(1-p)*(aa[None,:]**2)
    if isinstance(weights,Mapping): w={"response":float(weights.get("response",0)),"rt":float(weights.get("rt",0)),"process":float(weights.get("process",0))}
    else:
        z=list(weights); w={"response":z[0] if len(z)>0 else 0,"rt":z[1] if len(z)>1 else 0,"process":z[2] if len(z)>2 else 0}
    proc=np.resize(np.asarray(process_information,float),len(aa)); rti=np.resize(np.asarray(rt_information,float),len(aa)); et=np.resize(np.asarray(expected_time,float),len(aa))
    utility=w["response"]*resp + (w["rt"]*rti+w["process"]*proc-burden_weight*et)[None,:]
    return _result("eye_process_item_information",theta=th,response_information=resp,utility=utility)


def expected_process_information(info: Any, theta_weights: Any=None) -> np.ndarray:
    if getattr(info,"eyeprocess_class",None)!="eye_process_item_information": raise EyeProcessValidationError("info must come from process_item_information().")
    w=np.repeat(1/len(info.theta),len(info.theta)) if theta_weights is None else np.asarray(theta_weights,float); w=w/w.sum(); return w @ np.asarray(info.utility,float)


def select_next_item_process(theta: float, item_bank: Any, used: Sequence[str]=(), weights: Mapping[str,float]|Sequence[float]={"response":1,"rt":0,"process":0}, burden_weight: float=0) -> EyeResult:
    b=_as_df(item_bank,"item_bank"); _req_cols(b,["item_id","a","b"],"item_bank"); avail=b[~b.item_id.astype(str).isin([str(x) for x in used])].copy()
    if avail.empty: raise EyeProcessValidationError("No unused items remain.")
    pi=avail["process_information"].to_numpy(float) if "process_information" in avail else 0; ri=avail["rt_information"].to_numpy(float) if "rt_information" in avail else 0; et=avail["expected_time"].to_numpy(float) if "expected_time" in avail else 0
    info=process_item_information([theta],avail.a,avail.b,pi,ri,weights,et,burden_weight); u=np.asarray(info.utility)[0]; j=int(np.nanargmax(u))
    return _result("eye_process_item_selection",item_id=avail.iloc[j].item_id,utility=float(u[j]),row=avail.iloc[[j]].copy(),all_utilities=pd.DataFrame({"item_id":avail.item_id.to_numpy(),"utility":u}))


def simulate_process_cat(item_bank: Any, true_theta: float=0, n_items: int=10, weights: Mapping[str,float]|Sequence[float]={"response":1,"rt":0,"process":0}, burden_weight: float=0, seed: int=1) -> pd.DataFrame:
    bank=_as_df(item_bank,"item_bank"); rng=np.random.default_rng(seed); theta_hat=0.0; used=[]; rows=[]; responses=[]
    for step in range(1,min(int(n_items),len(bank))+1):
        sel=select_next_item_process(theta_hat,bank,used,weights,burden_weight); it=sel.row.iloc[0]; p=float(expit(float(it.a)*(true_theta-float(it.b)))); y=int(rng.binomial(1,p)); used.append(str(it.item_id)); responses.append(y)
        grid=np.linspace(-4,4,321); sub=bank.set_index(bank.item_id.astype(str)).loc[used]; aa=sub.a.to_numpy(float); bb=sub.b.to_numpy(float); yy=np.asarray(responses,float); probs=expit(grid[:,None]*aa[None,:]-(aa*bb)[None,:]); ll=np.sum(yy*np.log(np.clip(probs,_EPS,1))+(1-yy)*np.log(np.clip(1-probs,_EPS,1)),axis=1); theta_hat=float(grid[int(np.argmax(ll))])
        rows.append({"step":step,"item_id":it.item_id,"response":y,"selection_utility":sel.utility,"utility":sel.utility,"theta_hat":theta_hat,"true_theta":true_theta})
    return _tag(pd.DataFrame(rows),"eye_process_cat_simulation")


def _upgrade_registry() -> None:
    _register_builtins()
    mapping={"process_hmm":fit_process_hmm_irt,"latent_space_process":fit_latent_space_irt,"gpirt_shape_audit":fit_gpirt,"flow_mirt":fit_flow_mirt}
    for key,fn in mapping.items():
        if key in _REGISTRY: _REGISTRY[key]["fit_fun"] = fn

_upgrade_registry()
