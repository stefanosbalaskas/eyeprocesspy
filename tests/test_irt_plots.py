from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eyeprocesspy as ep
from eyeprocesspy.irt import EyeResult


def _items():
    return pd.DataFrame({"item_id":["I1","I2","I3","I4"],"a":[.8,1,1.2,1.4],"b":[-1,-.2,.4,1],"c":0.,"d":1.})


def test_all_irt_plot_counterparts_return_axes_with_data_or_explicit_empty_state():
    th=np.linspace(-3,3,31); items=_items(); info=ep.eyeprocess_irt_test_information(th,items); tcc=ep.eyeprocess_irt_test_characteristic_curve(th,items)
    spec=ep.eyeprocess_irt_model_spec("2pl"); ident=ep.eyeprocess_irt_identification_audit(spec,constraints={"theta_mean_fixed":True,"theta_sd_fixed":True})
    sparse=ep.eyeprocess_irt_sparse_design_audit(pd.DataFrame({"person":[1,1,2,2,3],"item":["I1","I2","I1","I3","I2"],"response":[1,0,1,np.nan,1]}),person="person",item="item",response="response",min_person_items=1,min_item_persons=1)
    rng=np.random.default_rng(1); p=pd.DataFrame(rng.uniform(.2,.8,(20,4)),columns=items.item_id); y=pd.DataFrame(rng.binomial(1,p),columns=items.item_id)
    q3=ep.eyeprocess_irt_q3(y,p); itemfit=ep.eyeprocess_irt_infit_outfit(y,p,by="item"); personfit=ep.eyeprocess_irt_infit_outfit(y,p,by="person")
    dashboard=EyeResult({"components":{"item_fit":itemfit,"person_fit":personfit,"q3":q3,"parameter_audit":{},"identification":ident}},eyeprocess_class="eye_irt_fit_dashboard")
    score_unc=EyeResult({"mean_se":.3,"median_se":.28,"p95_se":.5},eyeprocess_class="eye_irt_score_uncertainty")
    trace=ep.eyeprocess_irt_adaptive_trace(["I1","I2"],[0,.1],[.1,.2],[.5,.4],[1,1.5],[1,0])
    link_stability=EyeResult({"table":pd.DataFrame({"set":["A","B"],"A":[1.,1.1],"B":[0.,.1]})},eyeprocess_class="eye_irt_link_stability")
    dif=ep.eyeprocess_irt_dif_effect_curve(items.iloc[[0]],items.iloc[[1]]); dtf=ep.eyeprocess_irt_dtf_curve(items,items.assign(b=items.b+.1))
    profile=ep.eyeprocess_process_item_profile(pd.DataFrame({"item":["I1","I1","I2","I2","I3","I3","I4","I4"],"rt":[1,1.1,1.2,1.3,1.4,1.5,1.6,1.7]}),"item",["rt"])
    alignment=ep.eyeprocess_irt_process_alignment(items,profile)
    recovery=EyeResult({"estimates":pd.DataFrame({"a_truth":[1,1.2],"a_estimate":[.9,1.25],"b_truth":[-.2,.4],"b_estimate":[-.1,.35]})},eyeprocess_class="eye_irt_recovery_result")
    sbc=ep.eyeprocess_irt_sbc_summary(np.arange(20)%20,n_draws=19,bins=10)
    qaudit=ep.eyeprocess_cdm_qmatrix_audit(np.array([[1,0],[0,1],[1,1],[1,0]]))
    bankcov=ep.eyeprocess_irt_bank_coverage(items,target_information=.2); targeting=ep.eyeprocess_irt_targeting_gap(np.linspace(-2,2,80),items)
    miss=ep.eyeprocess_irt_missing_by_design_audit(np.array([[1,0],[np.nan,1.]]),np.array([[1,1],[0,1]]))
    ps=EyeResult({"table":pd.DataFrame({"estimate":[-.1,0,.2]})},eyeprocess_class="eye_irt_prior_sensitivity")

    calls=[
      (ep.plot_eye_irt_information_profile,(info,),{}),
      (ep.plot_eye_irt_test_characteristic_curve,(tcc,),{}),
      (ep.plot_eye_irt_identification_audit,(ident,),{}),
      (ep.plot_eye_irt_sparse_design_audit,(sparse,),{}),
      (ep.plot_eye_irt_q3_matrix,(q3,),{}),
      (ep.plot_eye_irt_item_fit,(itemfit,),{}),
      (ep.plot_eye_irt_person_fit,(personfit,),{}),
      (ep.plot_eye_irt_fit_dashboard,(dashboard,),{}),
      (ep.plot_eye_irt_score_uncertainty,(score_unc,),{}),
      (ep.plot_eye_irt_adaptive_trace,(trace,),{}),
      (ep.plot_eye_irt_link_stability,(link_stability,),{}),
      (ep.plot_eye_irt_dif_curve,(dif,),{}),
      (ep.plot_eye_irt_dtf_curve,(dtf,),{}),
      (ep.plot_eye_irt_process_alignment,(alignment,),{}),
      (ep.plot_eye_irt_recovery_result,(recovery,),{}),
      (ep.plot_eye_irt_sbc_evidence,(sbc,),{}),
      (ep.plot_eye_cdm_qmatrix_audit,(qaudit,),{}),
      (ep.plot_eye_irt_bank_coverage,(bankcov,),{}),
      (ep.plot_eye_irt_targeting_gap,(targeting,),{}),
      (ep.plot_eye_irt_missing_design_audit,(miss,),{}),
      (ep.plot_eye_irt_prior_sensitivity,(ps,),{}),
    ]
    for fun,args,kw in calls:
        ax=fun(*args,**kw)
        assert ax is not None and ax.figure is not None
        assert len(ax.lines)+len(ax.collections)+len(ax.patches)+len(ax.images) > 0 or ax.get_title()
        plt.close(ax.figure)
