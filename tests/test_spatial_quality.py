# ruff: noqa: E701, E702
import math

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep


def test_perfect_fixation_and_constant_offset_distinguish_accuracy_precision():
    t=np.arange(6)*10
    perfect=pd.DataFrame({"timestamp_ms":t,"gaze_x":0.,"gaze_y":0.,"target_x":0.,"target_y":0.})
    a=ep.compute_gaze_accuracy(perfect); p=ep.compute_rms_s2s(perfect,time="timestamp_ms"); b=ep.compute_bcea(perfect)
    assert a.accuracy_mean.iloc[0]==pytest.approx(0)
    assert p.precision_rms_s2s.iloc[0]==pytest.approx(0)
    assert b.bcea.iloc[0]==pytest.approx(0)
    off=perfect.copy(); off["gaze_x"]=1
    assert ep.compute_gaze_accuracy(off).accuracy_mean.iloc[0]==pytest.approx(1)
    assert ep.compute_rms_s2s(off,time="timestamp_ms").precision_rms_s2s.iloc[0]==pytest.approx(0)


def test_rms_known_value_and_missing_gap_not_bridged():
    d=pd.DataFrame({"gaze_x":[0,3,3],"gaze_y":[0,4,8],"timestamp_ms":[0,10,20]})
    assert ep.compute_rms_s2s(d,time="timestamp_ms").precision_rms_s2s.iloc[0]==pytest.approx(math.sqrt((25+16)/2))
    m=pd.DataFrame({"gaze_x":[0,np.nan,10],"gaze_y":[0,np.nan,0],"timestamp_ms":[0,10,20]})
    assert math.isnan(ep.compute_rms_s2s(m,time="timestamp_ms").precision_rms_s2s.iloc[0])


def test_bcea_probability_and_correlated_noise_formula():
    rng=np.random.default_rng(1); x=rng.normal(size=500); y=.8*x+rng.normal(scale=.3,size=500); d=pd.DataFrame({"gaze_x":x,"gaze_y":y})
    out=ep.compute_bcea(d,probability=.68).iloc[0]
    sx=np.std(x,ddof=0); sy=np.std(y,ddof=0); rho=np.corrcoef(x,y)[0,1]; expected=2*math.pi*(-math.log(.32))*sx*sy*math.sqrt(1-rho*rho)
    assert out.bcea==pytest.approx(expected)
    assert out.bcea_probability==pytest.approx(.68)
    with pytest.raises(ValueError): ep.compute_bcea(d,probability=1)


def test_sampling_effective_rate_jitter_duplicates_and_nonmonotonic_localized():
    d=pd.DataFrame({"grp":["a"]*4+["b"]*4,"timestamp_ms":[0,10,20,30,0,10,10,5],"gaze_x":0.,"gaze_y":0.,"target_x":0.,"target_y":0.})
    e=ep.estimate_effective_sampling_rate(d,by="grp"); assert e.loc[e.grp=="a","effective_sampling_hz"].iloc[0]==pytest.approx(100)
    v=ep.validate_gaze_quality_inputs(d,time="timestamp_ms",target_x="target_x",target_y="target_y",by="grp")
    assert len(v["group_issues"])==1 and v["group_issues"][0]["grp"]=="b"
    assert set(v["group_issues"][0]["issues"])=={"duplicate_timestamps","non_monotonic_timestamps"}


def test_validity_loss_and_missing_reasons():
    d=pd.DataFrame({"timestamp_ms":[0,10,20,30],"gaze_x":[0,np.nan,1,1],"gaze_y":[0,np.nan,1,1],"valid":[1,1,0,1],"missing_reason":[None,"blink",None,None]})
    q=ep.compute_gaze_data_loss(d,valid="valid",missing_reason="missing_reason").iloc[0]
    assert q.valid_sample_fraction==pytest.approx(.5); assert q.data_loss_fraction==pytest.approx(.5); assert q.missing_run_count==1
    assert q["missing_reason_blink_fraction"]==pytest.approx(.5)


def test_pixel_degree_conversion_is_explicit():
    geom={"screen_width_px":1920,"screen_height_px":1080,"screen_width_cm":53.0,"screen_height_cm":29.8,"viewing_distance_cm":60.0}
    d=pd.DataFrame({"gaze_x":[960,1060],"gaze_y":[540,540],"target_x":[960,960],"target_y":[540,540]})
    native=ep.compute_gaze_accuracy(d,unit="pixels")
    angular=ep.compute_gaze_accuracy(d,unit="pixels",output_unit="degrees",geometry=geom)
    assert native.unit.iloc[0]=="px" and angular.unit.iloc[0]=="deg" and angular.accuracy_mean.iloc[0]>0
    with pytest.raises(ValueError): ep.compute_gaze_accuracy(d,unit="pixels",output_unit="degrees")


def test_mixed_units_fail_and_mixed_targets_flag_review_only():
    d=pd.DataFrame({"timestamp_ms":[0,10,20,30],"gaze_x":[0,0,1,1],"gaze_y":[0,0,1,1],"target_x":[0,0,1,1],"target_y":[0,0,1,1],"coordinate_unit":["degrees"]*4})
    ep.validate_gaze_quality_inputs(d,time="timestamp_ms",target_x="target_x",target_y="target_y",unit_column="coordinate_unit")
    bad=d.copy(); bad.loc[3,"coordinate_unit"]="pixels"
    with pytest.raises(ValueError,match="mixed coordinate units"): ep.validate_gaze_quality_inputs(bad,time="timestamp_ms",target_x="target_x",target_y="target_y",unit_column="coordinate_unit")
    with pytest.raises(ValueError,match="mixed coordinate units"): ep.create_gaze_quality_report(bad)
    r=ep.create_gaze_quality_report(d); assert bool(r.review_required.iloc[0]); assert "mixed_accuracy_targets" in r.quality_flags.iloc[0]


def test_user_thresholds_flag_but_never_exclude():
    d=pd.DataFrame({"grp":["a"]*4,"timestamp_ms":[0,10,20,30],"gaze_x":[1,1,1,1],"gaze_y":[0,0,0,0],"target_x":[0]*4,"target_y":[0]*4})
    r=ep.create_gaze_quality_report(d,by="grp",thresholds={"accuracy_mean":{"max":.5}})
    assert r.accuracy_unit.iloc[0]=="deg"
    assert r.precision_rms_s2s_unit.iloc[0]=="deg"
    assert r.precision_sd_unit.iloc[0]=="deg"
    assert r.bcea_unit.iloc[0]=="deg^2"
    assert len(r)==1 and bool(r.review_required.iloc[0]) and "accuracy_mean>max" in r.quality_flags.iloc[0]
    assert r.attrs["gaze_quality_provenance"]["automatic_exclusion"] is False
    with pytest.raises(ValueError,match="must contain only"):
        ep.create_gaze_quality_report(d,by="grp",thresholds={"accuracy_mean":{"upper":.5}})
    with pytest.raises(ValueError,match="finite numeric"):
        ep.create_gaze_quality_report(d,by="grp",thresholds={"accuracy_mean":np.inf})


def test_synthetic_six_profiles_show_accuracy_precision_noninterchangeability():
    d=ep.simulate_gaze_quality_calibration(samples_per_target=8)
    assert d.profile.nunique()==6 and d.target_id.nunique()==9
    r=ep.create_gaze_quality_report(d,by=["profile","target_id"],valid="valid",missing_reason="missing_reason",nominal_sampling_hz=60)
    s=r.groupby("profile").mean(numeric_only=True)
    assert s.loc["poor_accuracy_good_precision","accuracy_mean"]>s.loc["good_accuracy_good_precision","accuracy_mean"]
    assert s.loc["good_accuracy_poor_precision","precision_rms_s2s"]>s.loc["good_accuracy_good_precision","precision_rms_s2s"]
    assert s.loc["missingness","data_loss_fraction"]>0
    assert abs(s.loc["irregular_sampling","effective_sampling_hz"]-60)>1


def test_plot_and_reporting_surfaces():
    import matplotlib
    matplotlib.use("Agg")
    d=ep.simulate_gaze_quality_calibration(samples_per_target=5)
    r=ep.create_gaze_quality_report(d,by=["profile","target_id"],valid="valid",missing_reason="missing_reason")
    assert hasattr(ep.plot_gaze_accuracy(r),"eyeprocess_plot_data")
    assert hasattr(ep.plot_gaze_precision(r),"eyeprocess_plot_data")
    assert hasattr(ep.plot_bcea(r),"eyeprocess_plot_data")
    assert hasattr(ep.plot_sampling_intervals(d[d.profile=="good_accuracy_good_precision"]),"eyeprocess_plot_data")
    fig=ep.plot_gaze_quality_dashboard(r); assert hasattr(fig,"eyeprocess_plot_data")
    txt=ep.report_gaze_quality(r); assert "automatic exclusion" in txt


def test_provenance_fingerprint_deterministic_and_changes_with_source():
    d=ep.simulate_gaze_quality_calibration(samples_per_target=4)
    a=ep.create_gaze_quality_report(d,by=["profile","target_id"]); b=ep.create_gaze_quality_report(d,by=["profile","target_id"])
    assert a.attrs["gaze_quality_provenance"]["source_fingerprint"]==b.attrs["gaze_quality_provenance"]["source_fingerprint"]
    d2=d.copy(); d2.loc[0,"gaze_x"]+=.1; c=ep.create_gaze_quality_report(d2,by=["profile","target_id"])
    assert a.attrs["gaze_quality_provenance"]["source_fingerprint"]!=c.attrs["gaze_quality_provenance"]["source_fingerprint"]
    av=ep.create_gaze_quality_report(d,by=["profile","target_id"],valid="valid")
    d3=d.copy(); d3.loc[0,"valid"]=0; e=ep.create_gaze_quality_report(d3,by=["profile","target_id"],valid="valid")
    assert av.attrs["gaze_quality_provenance"]["source_fingerprint"]!=e.attrs["gaze_quality_provenance"]["source_fingerprint"]


def test_validation_and_conversion_error_paths_and_non_dataframe_input():
    d={"gaze_x":[0,1],"gaze_y":[0,1],"timestamp_ms":[0,10]}
    assert ep.validate_gaze_quality_inputs(d,time="timestamp_ms")["n_rows"]==2
    with pytest.raises(ValueError,match="missing required columns"): ep.compute_gaze_accuracy(pd.DataFrame(d))
    with pytest.raises(ValueError,match="supplied together"): ep.validate_gaze_quality_inputs(pd.DataFrame(d),target_x="gaze_x")
    with pytest.raises(ValueError,match="unit must"): ep.validate_gaze_quality_inputs(pd.DataFrame(d),unit="cm")
    with pytest.raises(ValueError,match="time_unit"): ep.validate_gaze_quality_inputs(pd.DataFrame(d),time="timestamp_ms",time_unit="minutes")
    tagged=pd.DataFrame({**d,"u":["pixels","pixels"]})
    with pytest.raises(ValueError,match="conflicts"): ep.validate_gaze_quality_inputs(tagged,time="timestamp_ms",unit="degrees",unit_column="u")
    with pytest.raises(ValueError,match="unit and output_unit"): ep.compute_gaze_sd_precision(pd.DataFrame(d),unit="degrees",output_unit="cm")
    geom={"screen_width_px":1000,"screen_height_px":500,"screen_width_cm":50,"screen_height_cm":25,"viewing_distance_cm":60}
    nd=pd.DataFrame({"gaze_x":[.5,.6],"gaze_y":[.5,.6],"target_x":[.5,.5],"target_y":[.5,.5]})
    assert ep.compute_gaze_accuracy(nd,unit="normalized",output_unit="pixels",geometry=geom).unit.iloc[0]=="px"
    assert ep.compute_gaze_accuracy(nd,unit="normalized",output_unit="degrees",geometry=geom).unit.iloc[0]=="deg"
    dd=pd.DataFrame({"gaze_x":[0.,1.],"gaze_y":[0.,1.],"target_x":[0.,0.],"target_y":[0.,0.]})
    assert ep.compute_gaze_accuracy(dd,unit="degrees",output_unit="normalized",geometry=geom).unit.iloc[0]=="normalized"
    bad_geom=dict(geom); bad_geom["screen_width_px"]=0
    with pytest.raises(ValueError,match="finite positive"): ep.compute_gaze_accuracy(nd,unit="normalized",output_unit="degrees",geometry=bad_geom)


def test_precision_edge_paths_dimensions_and_gap_rules():
    d=pd.DataFrame({"timestamp_ms":[0,10,40],"gaze_x":[0,3,6],"gaze_y":[0,4,8]})
    assert ep.compute_rms_s2s(d,time="timestamp_ms",dimension="horizontal").precision_rms_s2s.iloc[0]==pytest.approx(3)
    assert ep.compute_rms_s2s(d,time="timestamp_ms",dimension="vertical").precision_rms_s2s.iloc[0]==pytest.approx(4)
    assert ep.compute_rms_s2s(d,time="timestamp_ms",max_gap_ms=15).n_steps.iloc[0]==1
    with pytest.raises(ValueError,match="dimension"): ep.compute_rms_s2s(d,dimension="radialish")
    with pytest.raises(ValueError,match="positive value"): ep.compute_rms_s2s(d,time="timestamp_ms",max_gap_ms=0)
    with pytest.raises(ValueError,match="time is required"): ep.compute_rms_s2s(d.drop(columns="timestamp_ms"),max_gap_ms=20)
    with pytest.raises(ValueError,match="time_unit"): ep.compute_rms_s2s(d,time="timestamp_ms",time_unit="minute")
    one=pd.DataFrame({"gaze_x":[1.],"gaze_y":[2.]})
    assert math.isnan(ep.compute_rms_s2s(one).precision_rms_s2s.iloc[0])
    assert ep.compute_gaze_sd_precision(one).precision_sd.iloc[0]==pytest.approx(0)
    assert math.isnan(ep.compute_bcea(one).bcea.iloc[0])
    one_report=ep.create_gaze_quality_report(pd.DataFrame({"timestamp_ms":[0],"gaze_x":[1.],"gaze_y":[2.]}),target_x=None,target_y=None)
    assert "insufficient_bcea_samples" in one_report.quality_flags.iloc[0]
    assert "insufficient_rms_pairs" in one_report.quality_flags.iloc[0]
    line=pd.DataFrame({"gaze_x":[1.,1.,1.],"gaze_y":[0.,1.,2.]})
    assert ep.compute_bcea(line).correlation_xy.iloc[0]==pytest.approx(0)
    bundle=ep.compute_gaze_precision(d,time="timestamp_ms",probability=.95)
    assert set(bundle)=={"rms_s2s","sd","bcea"} and bundle["bcea"].bcea_probability.iloc[0]==pytest.approx(.95)


def test_sampling_edge_paths_and_dropped_interval_diagnostics():
    d=pd.DataFrame({"timestamp_ms":[0,10,20,50]})
    i=ep.estimate_sampling_interval(d); j=ep.estimate_sampling_jitter(d); e=ep.estimate_effective_sampling_rate(d,nominal_sampling_hz=100)
    assert i.max_interval_ms.iloc[0]==pytest.approx(30); assert j.sampling_jitter_ms.iloc[0]>0
    assert e.long_interval_count.iloc[0]==1; assert e.dropped_interval_count.iloc[0]==2
    single=pd.DataFrame({"timestamp_ms":[0]})
    assert math.isnan(ep.estimate_sampling_interval(single).median_interval_ms.iloc[0])
    assert math.isnan(ep.estimate_sampling_jitter(single).sampling_jitter_ms.iloc[0])
    assert math.isnan(ep.estimate_effective_sampling_rate(single).effective_sampling_hz.iloc[0])
    assert ep.estimate_effective_sampling_rate(single,nominal_sampling_hz=100).effective_sampling_hz.iloc[0]==pytest.approx(100)
    for fn in (ep.estimate_sampling_interval,ep.estimate_sampling_jitter,ep.estimate_effective_sampling_rate):
        with pytest.raises(ValueError,match="time_unit"): fn(d,time_unit="minute")
    with pytest.raises(ValueError,match="nominal_sampling_hz"): ep.estimate_effective_sampling_rate(d,nominal_sampling_hz=0)
    with pytest.raises(ValueError,match="dropped_interval_factor"): ep.estimate_effective_sampling_rate(d,dropped_interval_factor=1)


def test_effective_frequency_uses_valid_sample_count_when_gaze_is_supplied():
    d=pd.DataFrame({"timestamp_ms":[0,10,20,30,40],"gaze_x":[0,0,np.nan,0,0],"gaze_y":[0,0,np.nan,0,0],"valid":[1,1,1,1,1]})
    stream=ep.estimate_effective_sampling_rate(d)
    valid=ep.estimate_effective_sampling_rate(d,x="gaze_x",y="gaze_y",valid="valid")
    assert stream.observed_sample_count.iloc[0]==5 and stream.effective_sample_count.iloc[0]==5
    assert stream.trial_duration_s.iloc[0]==pytest.approx(.05) and stream.effective_sampling_hz.iloc[0]==pytest.approx(100)
    assert valid.effective_sample_count.iloc[0]==4 and valid.effective_sampling_hz.iloc[0]==pytest.approx(80)
    assert valid.effective_count_rule.iloc[0]=="valid gaze samples with finite timestamps"
    with pytest.raises(ValueError,match="both be supplied"): ep.estimate_effective_sampling_rate(d,x="gaze_x")
    with pytest.raises(ValueError,match="required when valid"): ep.estimate_effective_sampling_rate(d,valid="valid")


def test_valid_fraction_bool_no_valid_empty_and_data_loss_no_time():
    d=pd.DataFrame({"gaze_x":[0,np.nan,1],"gaze_y":[0,np.nan,1],"valid":pd.Series([True,pd.NA,False],dtype="boolean")})
    q=ep.compute_valid_sample_fraction(d,valid="valid").iloc[0]
    assert q.valid_sample_fraction==pytest.approx(1/3) and q.invalid_sample_fraction==pytest.approx(1/3)
    q2=ep.compute_valid_sample_fraction(d.drop(columns="valid")).iloc[0]
    assert q2.valid_sample_fraction==pytest.approx(2/3)
    loss=ep.compute_gaze_data_loss(d,valid="valid",time=None).iloc[0]
    assert math.isnan(loss.longest_missing_run_ms)
    with pytest.raises(ValueError,match="time_unit"): ep.compute_gaze_data_loss(pd.DataFrame({"timestamp_ms":[0],"gaze_x":[0],"gaze_y":[0]}),time_unit="minute")


def test_summaries_threshold_variants_no_targets_comparisons_and_provenance():
    d=pd.DataFrame({"session_id":[1,1,2,2],"condition":["a","a","b","b"],"timestamp_ms":[0,10,0,10],"gaze_x":[0,0,1,1],"gaze_y":[0,0,1,1],"target_x":[0,0,0,0],"target_y":[0,0,0,0]})
    assert len(ep.summarise_spatial_quality(d,by="session_id",time="timestamp_ms"))==2
    assert len(ep.summarise_sampling_quality(d,by="session_id",time="timestamp_ms"))==2
    with pytest.raises(ValueError,match="threshold metrics"):
        ep.create_gaze_quality_report(d,by="session_id",thresholds={"missing":1})
    r=ep.create_gaze_quality_report(d,by="session_id",thresholds={"accuracy_mean":0.5,"valid_sample_fraction":{"min":1.1}})
    assert r.review_required.all()
    no_targets=d.drop(columns=["target_x","target_y"])
    r2=ep.create_gaze_quality_report(no_targets,by="session_id",target_x=None,target_y=None,quality_rules={"named":"rule"})
    assert "accuracy_mean" not in r2 and r2.attrs["gaze_quality_provenance"]["quality_rules"]=={"named":"rule"}
    assert len(ep.compare_gaze_quality_sessions(d,target_x="target_x",target_y="target_y"))==2
    assert len(ep.compare_gaze_quality_conditions(d,target_x="target_x",target_y="target_y"))==2
    with pytest.raises(ValueError,match="missing required column"): ep.compare_gaze_quality_sessions(d,session="not_here")


def test_plot_passed_axis_dashboard_missing_metrics_and_empty_report():
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    r=pd.DataFrame({"accuracy_mean":[1.],"precision_rms_s2s":[.1],"bcea":[.2]})
    assert ep.plot_gaze_accuracy(r,ax=ax) is ax
    fig2=ep.plot_gaze_quality_dashboard(pd.DataFrame({"accuracy_mean":[1.]})); assert hasattr(fig2,"eyeprocess_plot_data")
    assert ep.report_gaze_quality(pd.DataFrame())=="No gaze-quality rows were available."
    assert "Review required" in ep.report_gaze_quality(pd.DataFrame({"accuracy_mean":[np.nan]}))
    with pytest.raises(ValueError,match="digits"): ep.report_gaze_quality(r,digits=-1)
    with pytest.raises(ValueError,match="time_unit"): ep.plot_sampling_intervals(pd.DataFrame({"timestamp_ms":[0,10]}),time_unit="minute")


def test_simulator_small_sample_guard_and_reason_branches():
    with pytest.raises(ValueError,match="at least 4"): ep.simulate_gaze_quality_calibration(samples_per_target=3)
    with pytest.raises(ValueError,match="integer"): ep.simulate_gaze_quality_calibration(samples_per_target=4.5)
    with pytest.raises(ValueError,match="finite positive"): ep.simulate_gaze_quality_calibration(nominal_sampling_hz=0)
    d=ep.simulate_gaze_quality_calibration(samples_per_target=15)
    miss=d[d.profile=="missingness"]
    assert {"blink","tracker_invalidity"}.issubset(set(miss.missing_reason.dropna()))


def test_internal_coordinate_contract_and_empty_validation_group_path():
    import eyeprocesspy.spatial_quality as sq
    d=pd.DataFrame({"gaze_x":[0.],"gaze_y":[0.],"target_x":[0.]})
    with pytest.raises(ValueError,match="supplied together"):
        sq._prepare_coordinates(d,x="gaze_x",y="gaze_y",target_x="target_x",target_y=None,unit="degrees",output_unit=None,geometry=None)
    empty=pd.DataFrame({"gaze_x":pd.Series(dtype=float),"gaze_y":pd.Series(dtype=float),"timestamp_ms":pd.Series(dtype=float)})
    assert ep.validate_gaze_quality_inputs(empty,time="timestamp_ms")["valid"] is True


def test_validation_without_time_branch():
    d=pd.DataFrame({"gaze_x":[0.],"gaze_y":[0.]})
    out=ep.validate_gaze_quality_inputs(d)
    assert out["time_unit"] is None and out["group_issues"]==[]


def test_cross_language_fixture_matches_frozen_expected_values():
    from pathlib import Path
    root=Path(__file__).parent/"fixtures"
    d=pd.read_csv(root/"spatial_quality_input.csv")
    expected=pd.read_csv(root/"spatial_quality_expected.csv")
    actual=ep.create_gaze_quality_report(d,by="fixture_group",valid="valid",nominal_sampling_hz=100,bcea_probability=.68)
    for col in expected.columns:
        if col=="fixture_group":
            assert actual[col].tolist()==expected[col].tolist()
        else:
            np.testing.assert_allclose(pd.to_numeric(actual[col]),pd.to_numeric(expected[col]),rtol=1e-12,atol=1e-12,equal_nan=True)


def test_missing_group_ids_are_preserved_and_flagged():
    d=pd.DataFrame({
        "trial_id":["T1","T1",np.nan,np.nan],
        "timestamp_ms":[0,10,0,0],
        "gaze_x":[0,0,1,1],
        "gaze_y":[0,0,1,1],
        "target_x":[0,0,1,1],
        "target_y":[0,0,1,1],
    })
    q=ep.create_gaze_quality_report(d,by="trial_id")
    assert len(q)==2
    missing=q[q.trial_id.isna()]
    assert len(missing)==1
    assert "duplicate_timestamps" in missing.quality_flags.iloc[0]
