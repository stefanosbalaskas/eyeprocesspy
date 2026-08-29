"""Matplotlib counterparts for eyeprocess 0.8 process-governance/window/pupil plots."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from .exceptions import EyeProcessValidationError
from .process_governance_08 import pupil_response_kernel

__all__=[
 "plot_eye_biometric_preflight","plot_eye_process_anomaly_audit","plot_eye_presentation_accessibility",
 "plot_eye_process_drift_audit","plot_eye_process_window_sensitivity","plot_eye_pupil_frequency_features",
 "plot_eye_pupil_frequency_stability","plot_eye_pupil_deconvolution","plot_eye_pupil_confound_model",
 "plot_eye_aoi_trajectory","plot_eye_aoi_growth_curve","plot_eye_signal_filter_audit","plot_eye_process_windows",
 "plot_eye_pupil_fatigue_drift","plot_eye_presentation_fairness_comparison",
 "plot_pupil_spectrum","plot_pupil_band_power","plot_pupil_activity_windows","plot_pupil_activity_sensitivity",
 "plot_process_window_sensitivity",
]

def _ax(ax=None):
    import matplotlib.pyplot as plt
    return plt.subplots()[1] if ax is None else ax

def _cls(x,name):
    if getattr(x,"eyeprocess_class",None)!=name: raise EyeProcessValidationError(f"x must be {name}.")

def plot_eye_biometric_preflight(x:Any,type:str="heatmap",ax=None):
    _cls(x,"eye_biometric_preflight");ax=_ax(ax);d=x.table.copy()
    if type=="decision_counts":
        p=d.preflight_decision.value_counts();ax.bar(p.index.astype(str),p.values);ax.tick_params(axis='x',rotation=45);plotdata=p.rename_axis('decision').reset_index(name='count')
    elif type=="heatmap":
        flags=x.flag_columns;z=d.sort_values('preflight_flag_count',ascending=False);M=z[flags].astype(int).to_numpy();ax.imshow(M,aspect='auto');ax.set_xticks(range(len(flags)),[f.removesuffix('_flag') for f in flags],rotation=90);plotdata=z
    else: raise EyeProcessValidationError("type must be heatmap or decision_counts.")
    ax.set_title("Biometric pre-flight");ax.eyeprocess_plot_data=plotdata;return ax

def plot_eye_process_anomaly_audit(x:Any,ax=None):
    _cls(x,"eye_process_anomaly_audit");ax=_ax(ax);d=x.table.copy();ax.vlines(np.arange(len(d)),0,d.mahalanobis_process_distance);ax.axhline(x.threshold,ls='--');ax.set(ylabel='Mahalanobis process distance',title='Multivariate process/data-quality review distance');ax.eyeprocess_plot_data=d;return ax

def plot_eye_presentation_accessibility(x:Any,ax=None):
    _cls(x,"eye_presentation_accessibility");ax=_ax(ax);d=x.table.copy();ax.hist(d.presentation_sensitivity_index.dropna());ax.axvline(x.threshold,ls='--');ax.set(title='Presentation/accessibility sensitivity audit');ax.eyeprocess_plot_data=d;return ax

def plot_eye_process_drift_audit(x:Any,type:str="trajectory",metric:str|None=None,item:Any=None,ax=None):
    _cls(x,"eye_process_drift_audit");ax=_ax(ax);metric=metric or x.metrics[0]
    if metric not in x.metrics: raise EyeProcessValidationError(f"Unknown metric: {metric}")
    if type=="trajectory":
        tr=x.trajectories.copy();item=item if item is not None else tr[x.item].iloc[0];d=tr.loc[tr[x.item].isin(np.atleast_1d(item))].sort_values('batch_order');ax.plot(d.batch_order,d[metric],marker='o')
    elif type=="delta":
        d=x.table.copy();dc=f"{metric}_delta";o=np.argsort(pd.to_numeric(d[dc],errors='coerce'));ax.scatter(d[dc].iloc[o],np.arange(len(d)));ax.axvline(0,ls=':')
    elif type=="control":
        d=x.trajectories.groupby('batch_order',as_index=False)[metric].mean().rename(columns={metric:'mean_metric'});ax.plot(d.batch_order,d.mean_metric,marker='o');mu=d.mean_metric.mean();sd=d.mean_metric.std();ax.axhline(mu,ls='--');ax.axhline(mu-2*sd,ls=':');ax.axhline(mu+2*sd,ls=':')
    elif type=="heatmap":
        d=x.table.copy();cols=[c for c in d if c.endswith('_delta')];M=d[cols].to_numpy(float);sd=np.nanstd(M,axis=0,ddof=1);Z=np.divide(M,sd,out=np.zeros_like(M),where=np.isfinite(sd)&(sd>0));ax.imshow(Z,aspect='auto');ax.set_xticks(range(len(cols)),[c.removesuffix('_delta') for c in cols],rotation=90)
    else: raise EyeProcessValidationError("type must be trajectory, delta, heatmap, or control.")
    ax.set_title('Process drift audit');ax.eyeprocess_plot_data=d;return ax

def plot_eye_process_window_sensitivity(x:Any,ax=None):
    _cls(x,"eye_process_window_sensitivity");ax=_ax(ax);d=x.table.copy()
    for step,z in d.groupby('step_ms'):
        ax.plot(z.width_ms,z.mean_value,marker='o',label=f"step {step} ms")
    if d.step_ms.nunique()>1:
        ax.legend()
    ax.set(xlabel='Window width (ms)',ylabel=f"Mean {x.metric}",title='Process-window sensitivity')
    ax.eyeprocess_plot_data=d
    return ax

def plot_eye_pupil_frequency_features(x:Any,type:str="features",ax=None):
    _cls(x,"eye_pupil_frequency_features");ax=_ax(ax);d=x.features.copy()
    if type=="power_relationship": ax.scatter(d.pupil_low_frequency_power,d.pupil_high_frequency_power);ax.set(xlabel='Low-band power',ylabel='High-band power')
    elif type=="features":
        vars=[v for v in ['pupil_frequency_contrast','pupil_velocity_activity','pupil_ripa_proxy'] if v in d];ax.bar(vars,[pd.to_numeric(d[v],errors='coerce').mean() for v in vars]);ax.tick_params(axis='x',rotation=35)
    else: raise EyeProcessValidationError("type must be features or power_relationship.")
    ax.set_title('Pupil activity/frequency features');ax.eyeprocess_plot_data=d;return ax

def plot_eye_pupil_frequency_stability(x:Any,feature:str='pupil_frequency_contrast',ax=None):
    _cls(x,"eye_pupil_frequency_stability");ax=_ax(ax);d=x.table.copy();a=d.groupby('window_ms',as_index=False)[feature].mean().rename(columns={feature:'value'});ax.plot(a.window_ms,a.value,marker='o');ax.set(title='Pupil frequency-feature stability');ax.eyeprocess_plot_data=a;return ax

def plot_eye_pupil_deconvolution(x:Any,type:str='observed_fitted',ax=None):
    _cls(x,"eye_pupil_deconvolution");ax=_ax(ax)
    if type=='kernels':
        t=np.linspace(-500,3000,300);d=pd.DataFrame({'time':t,'kernel':pupil_response_kernel(t,x.tmax_ms,x.shape)});ax.plot(d.time,d.kernel)
    elif type=='effects':
        d=x.effects.copy();cols=[c for c in d if c.startswith('beta__')];ax.bar([c.removeprefix('beta__') for c in cols],[d[c].mean() for c in cols])
    else:
        d=x.fitted.copy()
        if type=='residuals':ax.scatter(d.time,d.residual,s=8);ax.axhline(0,ls='--')
        elif type=='observed_fitted':ax.plot(d.time,d.observed);ax.plot(d.time,d.fitted,ls='--')
        else:raise EyeProcessValidationError("unknown deconvolution plot type")
    ax.set_title('Pupil deconvolution');ax.eyeprocess_plot_data=d;return ax

def plot_eye_pupil_confound_model(x:Any,type:str='raw_adjusted',ax=None):
    _cls(x,"eye_pupil_confound_model");ax=_ax(ax);d=x.data.copy()
    if type=='luminance':ax.scatter(d['.luminance'],d['.pupil'])
    elif type=='trial_order':ax.scatter(d['.trial'],d.pupil_confound_adjusted)
    elif type=='raw_adjusted':ax.scatter(d['.pupil'],d.pupil_confound_adjusted);lo=min(d['.pupil'].min(),d.pupil_confound_adjusted.min());hi=max(d['.pupil'].max(),d.pupil_confound_adjusted.max());ax.plot([lo,hi],[lo,hi],ls='--')
    elif type=='theta_luminance_surface':
        if pd.to_numeric(d['.theta'],errors='coerce').std()==0: ax.text(.5,.5,'Theta unavailable',ha='center')
        else: ax.scatter(d['.luminance'],d['.theta'],c=d.pupil_confound_adjusted)
    else: raise EyeProcessValidationError("unknown pupil confound plot type")
    ax.set_title('Pupil confound model');ax.eyeprocess_plot_data=d;return ax

def plot_eye_aoi_trajectory(x:Any,type:str='coefficients',ax=None):
    _cls(x,"eye_aoi_trajectory");ax=_ax(ax);d=x.features.copy();cols=[c for c in d if '_gca_degree' in c]
    if type=='coefficients':ax.scatter([d[c].mean() for c in cols],np.arange(len(cols)));ax.axvline(0,ls=':');ax.set_yticks(range(len(cols)),cols)
    elif type=='profiles':ax.plot(np.arange(len(cols)),d[cols].to_numpy().T)
    else:raise EyeProcessValidationError("type must be coefficients or profiles.")
    ax.set_title('AOI trajectory');ax.eyeprocess_plot_data=d;return ax

def plot_eye_aoi_growth_curve(x:Any,ax=None):
    _cls(x,"eye_aoi_growth_curve");from .process_governance_08 import predict_aoi_trajectory;ax=_ax(ax);p=predict_aoi_trajectory(x);ax.scatter(x.time,x.outcome,s=12);ax.plot(p.time,p.predicted);ax.set_title('AOI growth curve');ax.eyeprocess_plot_data=p;return ax

def plot_eye_signal_filter_audit(x:Any,ax=None):
    _cls(x,"eye_signal_filter_audit");ax=_ax(ax);d=x.data.copy();ax.plot(d.sample_index,d.raw);ax.plot(d.sample_index,d.filtered,ls='--');ax.set_title(f'Raw and filtered eye signal -- {x.method}');ax.eyeprocess_plot_data=d;return ax

def plot_eye_process_windows(x:Any,feature:str='pupil_mean',group:str|None=None,ax=None):
    _cls(x,"eye_process_windows");ax=_ax(ax);d=x.data.copy()
    if group is None or group not in d:
        a=d.groupby('window_mid',as_index=False)[feature].mean().rename(columns={feature:'value'});ax.plot(a.window_mid,a.value,marker='o')
    else:
        a=d.groupby([group,'window_mid'],as_index=False)[feature].mean().rename(columns={feature:'value'});
        for g,z in a.groupby(group):ax.plot(z.window_mid,z.value,label=str(g));ax.legend()
    ax.set(title='Windowed process trajectory',ylabel=feature);ax.eyeprocess_plot_data=a;return ax

def plot_eye_pupil_fatigue_drift(x:Any,ax=None):
    _cls(x,"eye_pupil_fatigue_drift");ax=_ax(ax);d=x.data.copy();ax.scatter(d.trial_order,d.pupil,s=10);b=np.polyfit(d.trial_order,d.pupil,1);xx=np.array([d.trial_order.min(),d.trial_order.max()]);ax.plot(xx,np.polyval(b,xx),ls='--');ax.set_title(f'Within-person trial-order sensitivity -- {x.engine}');ax.eyeprocess_plot_data=d;return ax

def plot_eye_presentation_fairness_comparison(x:Any,ax=None):
    _cls(x,"eye_presentation_fairness_comparison");ax=_ax(ax);d=x.summary.copy();ax.bar(d.variant.astype(str),d['mean']);ax.set_title('Presentation-variant outcome comparison');ax.eyeprocess_plot_data=d;return ax

def plot_pupil_spectrum(signal:Any,sampling_rate_hz:float,max_hz:float|None=None,**kwargs:Any):
    ax=kwargs.pop("ax",None)
    from .process_governance_08 import _interp
    if sampling_rate_hz<=0:raise EyeProcessValidationError('sampling_rate_hz must be positive.')
    max_hz=sampling_rate_hz/2 if max_hz is None else max_hz;y=_interp(signal)
    if len(y)<8 or not np.all(np.isfinite(y)):raise EyeProcessValidationError('At least eight samples with sufficient finite signal values are required.')
    y=y-y.mean();n=len(y);f=np.fft.fft(y);power=np.abs(f)**2/n;freq=np.arange(n)*sampling_rate_hz/n;keep=(freq<=max_hz)&(freq<=sampling_rate_hz/2);d=pd.DataFrame({'frequency_hz':freq[keep],'power':power[keep]});ax=_ax(ax);ax.plot(d.frequency_hz,d.power);ax.set_title('Pupil signal spectrum');ax.eyeprocess_plot_data=d;return ax

def plot_pupil_band_power(x:Any,**kwargs:Any):
    ax=kwargs.pop("ax",None)
    _cls(x,'eye_pupil_frequency_features');d=x.features;vals=pd.DataFrame({'band':['low_band','high_band'],'power':[d.pupil_low_frequency_power.mean(),d.pupil_high_frequency_power.mean()]});ax=_ax(ax);ax.bar(vals.band,vals.power);ax.set_title('Pupil frequency-band power');ax.eyeprocess_plot_data=vals;return ax

def plot_pupil_activity_windows(x:Any,feature:str='pupil_frequency_contrast',**kwargs:Any):
    ax=kwargs.pop('ax',None)
    d=x.features.copy() if getattr(x,'eyeprocess_class',None)=='eye_pupil_frequency_features' else x.table.copy() if getattr(x,'eyeprocess_class',None)=='eye_pupil_frequency_stability' else None
    if d is None:raise EyeProcessValidationError('x must be a pupil frequency feature/stability object.')
    ax=_ax(ax);ax.plot(np.arange(1,len(d)+1),pd.to_numeric(d[feature],errors='coerce'),marker='o');ax.set_title('Pupil activity across windows/groups');ax.eyeprocess_plot_data=d;return ax

def plot_pupil_activity_sensitivity(x:Any,feature:str='pupil_frequency_contrast',**kwargs:Any):
    ax=kwargs.pop('ax',None)
    return plot_eye_pupil_frequency_stability(x,feature,ax)
def plot_process_window_sensitivity(x:Any,**kwargs:Any):
    ax=kwargs.pop('ax',None)
    return plot_eye_process_window_sensitivity(x,ax)
