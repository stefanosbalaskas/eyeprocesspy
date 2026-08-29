from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eyeprocesspy as ep


def _trial_data(n_persons=30,n_items=5,seed=11):
    rng=np.random.default_rng(seed); rows=[]
    for p in range(n_persons):
        group="A" if p<n_persons/2 else "B"; ability=rng.normal()
        for j in range(n_items):
            process=500+25*j+35*(group=="B")+rng.normal(0,45)
            prob=1/(1+np.exp(-(ability-.25*j+.18*(group=="B")+.0007*process)))
            rows.append({"person_id":p+1,"item_id":f"I{j+1}","group":group,"ability":ability,"response":rng.binomial(1,prob),"dwell_ms":process,"pupil":3+.05*j+.08*(group=="B")+rng.normal(0,.1),"time":p%5+1})
    return pd.DataFrame(rows)


def test_device_linking_estimates_applies_and_plots():
    rng=np.random.default_rng(1); base=pd.MultiIndex.from_product([range(1,31),range(1,4)],names=["person_id","item_id"]).to_frame(index=False); ref=base.copy(); ref["device"]="reference"; ref["metric"]=rng.normal(10,2,len(ref)); cand=ref.copy(); cand["device"]="candidate"; cand["metric"]=1+1.1*ref.metric+rng.normal(0,.3,len(ref)); data=pd.concat([ref,cand],ignore_index=True)
    fit=ep.fit_device_linking(data,"metric","reference"); assert fit.eyeprocess_class=="eye_device_linking"
    linked=ep.apply_device_linking(cand,fit); assert "metric_linked" in linked
    equivalence=ep.audit_device_equivalence(fit,equivalence_margin=2); assert equivalence.eyeprocess_class=="eye_device_equivalence"
    assert len(ep.estimate_device_specific_error(fit))>=1
    for fun,obj in [(ep.plot_device_agreement,fit),(ep.plot_device_bias_by_magnitude,fit),(ep.plot_device_transfer_curve,fit),(ep.plot_cross_vendor_metric_matrix,fit),(ep.plot_device_equivalence_intervals,equivalence)]:
        ax=fun(obj); assert ax.figure is not None; plt.close(ax.figure)
    eq=ep.fit_device_linking(data,"metric","reference",method="equipercentile"); out=ep.apply_device_linking(cand,eq); assert np.isfinite(out.metric_linked).all()
    ax=ep.plot_device_bias_by_magnitude(eq); assert ax.figure is not None; plt.close(ax.figure)


def test_pareto_item_bank_optimization_selects_requested_items_and_plots():
    rng=np.random.default_rng(1); items=pd.DataFrame({"item_id":[f"I{i}" for i in range(1,21)],"information":rng.random(20),"burden":rng.random(20),"fairness":rng.random(20),"exposure":rng.random(20)})
    spec=ep.item_objective_spec("information","burden","fairness","exposure"); pareto=ep.item_pareto_front(items,spec); assert pareto.eyeprocess_class=="eye_item_pareto" and pareto.table.pareto_front.any()
    opt=ep.optimize_item_bank(pareto,8,spec); assert len(opt.selected)==8
    stability=ep.audit_bank_decision_stability(opt,draws=20); assert len(stability.summary)==20
    for fun,obj in [(ep.plot_item_pareto,pareto),(ep.plot_objective_tradeoffs,pareto),(ep.plot_bank_information_coverage,opt),(ep.plot_decision_stability,stability),(ep.plot_selected_bank_profile,opt)]:
        ax=fun(obj); assert ax.figure is not None; plt.close(ax.figure)


def test_process_dif_drift_decomposition_transportability_and_plots():
    data=_trial_data(); dif=ep.fit_process_dif(data,"response","dwell_ms","group","item_id"); assert dif.eyeprocess_class=="eye_process_dif" and len(dif.summary)==5
    drift=ep.monitor_dif_drift(data,"time","group",["dwell_ms","pupil"],"item_id"); assert drift.eyeprocess_class=="eye_dif_drift" and {"slope","n_time_points"}.issubset(drift.slopes.columns); assert np.all(drift.slopes.slope.isna()|np.isfinite(drift.slopes.slope))
    dec=ep.decompose_dif_evidence(dif); assert "evidence_pattern" in dec.table
    trans_df=dif.summary.copy(); trans_df["device"]=["A","B","A","B","A"]
    transport=ep.audit_fairness_transportability(trans_df); assert transport.eyeprocess_class=="eye_fairness_transportability"
    for fun,obj in [(ep.plot_process_dif_forest,dif),(ep.plot_group_icc_process_overlay,dif),(ep.plot_item_group_process_curves,dif),(ep.plot_dif_drift_heatmap,drift),(ep.plot_fairness_transport_matrix,transport)]:
        ax=fun(obj); assert ax.figure is not None; plt.close(ax.figure)


def test_conditional_process_centiles_deviation_transportability_and_plots():
    rng=np.random.default_rng(1); age=rng.uniform(10,18,120); ref=pd.DataFrame({"age":age,"dwell_ms":np.exp(6+.03*age+rng.normal(0,.1,120))})
    model=ep.fit_process_norms(ref,"dwell_ms","age"); assert model.eyeprocess_class=="eye_process_norms"
    pred=ep.predict_process_centiles(model,pd.DataFrame({"age":[12,15,18]})); assert len(pred)==3
    scores=ep.score_process_deviation(model,ref.iloc[:10],"centile"); assert np.all((scores.deviation_score>=0)&(scores.deviation_score<=100))
    audit=ep.audit_norm_transportability(model,ref.iloc[:40]); assert audit.eyeprocess_class=="eye_norm_transportability"
    for fun in [ep.plot_process_centiles,ep.plot_normative_fan,ep.plot_person_normative_profile,ep.plot_item_normative_deviation]:
        ax=fun(model); assert ax.figure is not None; plt.close(ax.figure)
