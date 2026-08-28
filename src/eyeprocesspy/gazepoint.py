from __future__ import annotations

from pathlib import Path
import re
from typing import Any
import numpy as np
import pandas as pd

from .adapters import combine_eye_datasets
from .dataset import add_provenance, is_eye_dataset, new_eye_dataset, validate_eye_dataset
from .importers import (
    _as_character_id,
    _first_existing,
    _map_column,
    _read_delimited,
    _safe_logical,
    _safe_numeric,
    read_eye_generic,
)
from .mapping import eye_mapping
from .schema import empty_eye_table, new_coordinate_space, standardize_eye_table
from .timebase import estimate_sampling_rate

_GP_COLUMNS = {
    "time": ["TIME", "TIMETICK", "TIME_TICK", "timestamp", "TIME_SECS"],
    "participant": ["USER", "USER_ID", "PARTICIPANT", "participant_id"],
    "recording": ["RECORDING_ID", "recording_id", "SESSION_ID"],
    "x": ["BPOGX", "FPOGX", "CX", "gaze_x"],
    "y": ["BPOGY", "FPOGY", "CY", "gaze_y"],
    "valid": ["BPOGV", "FPOGV", "gaze_valid"],
    "left_x": ["LPOGX", "left_x"], "left_y": ["LPOGY", "left_y"],
    "right_x": ["RPOGX", "right_x"], "right_y": ["RPOGY", "right_y"],
    "left_valid": ["LPOGV", "left_valid"], "right_valid": ["RPOGV", "right_valid"],
    "pupil_left": ["LPMM", "LPD", "LPS", "pupil_left"],
    "pupil_right": ["RPMM", "RPD", "RPS", "pupil_right"],
    "pupil_left_valid": ["LPMMV", "LPV", "pupil_left_valid"],
    "pupil_right_valid": ["RPMMV", "RPV", "pupil_right_valid"],
    "fixation_id": ["FPOGID", "FIXATION_ID", "fixation_id"],
    "stimulus": ["MEDIA_ID", "MEDIA_NAME", "stimulus_id"],
    "trial": ["TRIAL_ID", "TRIAL_INDEX", "trial_id"],
    "marker": ["USER_DATA", "MARKER", "EVENT", "event_name"],
}


def _gp_pick(names, key):
    return _first_existing(list(map(str, names)), _GP_COLUMNS[key])


def _gp_id_token(x):
    if x is None or (isinstance(x,float) and np.isnan(x)): return "unknown"
    s=str(x).strip() or "unknown"
    return re.sub(r"_+","_",re.sub(r"[^A-Za-z0-9]+","_",s))


def _gp_filename_identity(path, participant_id=None, recording_id=None, session_id="S001"):
    base=Path(path).stem
    stem=re.sub(r"(?i)(?:[_ -](?:all[_ -]?gaze|fixations?|user(?:[_ -]?fix)?))$","",base).strip()
    m=re.match(r"(?i)^user[ _-]*([0-9]+)$",stem)
    inferred=f"User {m.group(1)}" if m else stem
    if not inferred or re.match(r"(?i)^data[_ -]?summary",inferred): inferred="P001"
    participant=str(participant_id if participant_id is not None else inferred)
    session=str(session_id)
    recording=str(recording_id if recording_id is not None else f"rec_{_gp_id_token(participant)}_{_gp_id_token(session)}")
    return {"participant_id":participant,"recording_id":recording,"session_id":session,"source_stem":stem}


def _gp_time_info(data: pd.DataFrame):
    nms=list(map(str,data.columns))
    media=next((n for n in nms if re.match(r"(?i)^TIME(?:\(|$)",n)),None)
    tick=next((n for n in nms if re.match(r"(?i)^TIMETICK(?:\(|$)",n)),None)
    freq=np.nan
    if tick:
        m=re.search(r"(?i)f\s*=\s*([0-9.]+)",tick)
        if m:
            try: freq=float(m.group(1))
            except ValueError: pass
    start=pd.NA
    if media:
        m=re.match(r"^TIME\((.*)\)$",media)
        if m: start=m.group(1)
    return {"media_time_col":media,"tick_col":tick,"tick_frequency":freq,"recording_start":start}


def _gp_tick_origin(data,info):
    tick_col=info["tick_col"]; freq=info["tick_frequency"]
    if not tick_col or not np.isfinite(freq): return np.nan
    tick=_safe_numeric(data[tick_col]); media=_safe_numeric(data[info["media_time_col"]]) if info["media_time_col"] else pd.Series(np.nan,index=data.index)
    keep=np.flatnonzero(np.isfinite(tick.to_numpy(dtype=float)))
    if len(keep)==0: return np.nan
    i=keep[0]
    return float(tick.iloc[i]-media.iloc[i]*freq) if np.isfinite(media.iloc[i]) else float(tick.iloc[i])


def _gp_biometric_mapping(data):
    nms=list(map(str,data.columns))
    candidates={
        "gsr_raw":["GSR","GSR_RAW"],
        "eda":["GSR_US","EDA","SKIN_CONDUCTANCE"],
        "skin_conductance_level":["GSR_US_TONIC","GSR_SCL","SCL"],
        "skin_conductance_response":["GSR_US_PHASIC","GSR_SCR","SCR"],
        "heart_rate":["HR","HEART_RATE","HEARTRATE","Heart Rate"],
        "heart_rate_period":["HRP"],
        "interbeat_interval":["IBI","INTERBEAT_INTERVAL","RR_INTERVAL"],
        "engagement_dial":["DIAL","ENGAGEMENT","ENGAGEMENT_DIAL"],
    }
    return {channel:col for channel,cands in candidates.items() if (col:=_first_existing(nms,cands)) is not None}


def _gp_mapping(data):
    nms=list(map(str,data.columns)); info=_gp_time_info(data)
    timestamp=info["tick_col"] or info["media_time_col"] or _gp_pick(nms,"time")
    return eye_mapping(
        participant=_gp_pick(nms,"participant"),recording=_gp_pick(nms,"recording"),timestamp=timestamp,
        x=_gp_pick(nms,"x"),y=_gp_pick(nms,"y"),left_x=_gp_pick(nms,"left_x"),left_y=_gp_pick(nms,"left_y"),
        right_x=_gp_pick(nms,"right_x"),right_y=_gp_pick(nms,"right_y"),gaze_valid=_gp_pick(nms,"valid"),
        left_valid=_gp_pick(nms,"left_valid"),right_valid=_gp_pick(nms,"right_valid"),pupil_left=_gp_pick(nms,"pupil_left"),
        pupil_right=_gp_pick(nms,"pupil_right"),pupil_left_valid=_gp_pick(nms,"pupil_left_valid"),pupil_right_valid=_gp_pick(nms,"pupil_right_valid"),
        fixation_id=_gp_pick(nms,"fixation_id"),blink_id=_first_existing(nms,["BKID","BLINK_ID"]),trial=_gp_pick(nms,"trial"),
        stimulus=_gp_pick(nms,"stimulus"),event_name=_gp_pick(nms,"marker"),biometric_channels=_gp_biometric_mapping(data)
    )


def _gp_is_summary_report(path):
    p=Path(path)
    if not p.is_file(): return False
    if re.match(r"(?i)^Data[_ -]?Summary[_ -]?export.*\.csv$",p.name): return True
    try: lines=p.read_text(encoding="utf-8",errors="replace").splitlines()[:20]
    except OSError: return False
    return any(line.strip()=="AOI Summary" for line in lines) and any(line.startswith("Gazepoint Analysis") for line in lines)


def is_gazepoint_export(path, inspect_rows=20):
    """Return Gazepoint format confidence using the frozen 0.11.1 detector."""
    p=Path(path)
    if p.is_dir():
        files=[q for q in p.iterdir() if q.suffix.lower() in {".csv",".txt",".tsv"}]
        return max((is_gazepoint_export(q,inspect_rows=inspect_rows) for q in files),default=0.0)
    if p.suffix.lower() not in {".csv",".txt",".tsv"}: return 0.0
    if _gp_is_summary_report(p): return 0.99
    filename_signature=bool(re.search(r"(?i)(?:_all[_ -]?gaze|_fixations?|[-_]user(?:[-_]fix)?)\.csv$",p.name))
    try: header=list(_read_delimited(p,nrows=min(2,int(inspect_rows))).columns)
    except Exception: return .75 if filename_signature else 0.0
    known={str(v).upper() for values in _GP_COLUMNS.values() for v in values}
    gp_hits=sum(str(h).upper() in known for h in header)
    sig=any(s in {str(h).upper() for h in header} for s in ["FPOGX","BPOGX","LPOGX","RPOGX","FPOGV","BPOGV","TIMETICK(F=10000000)"])
    score=min(1.0,gp_hits/8.0)
    if sig: score=max(score,.85)
    if filename_signature or "currentaoistatistics" in p.name.lower(): score=max(score,.95)
    return float(score)


def gp_identify_export_type(path):
    p=Path(path)
    if p.is_dir(): return "folder"
    if _gp_is_summary_report(p) or "currentaoistatistics" in p.name.lower(): return "aoi_statistics"
    if re.search(r"(?i)(?:_fixations?|[-_]user[-_]?fix)\.csv$",p.name): return "fixations"
    try: d=_read_delimited(p,nrows=3)
    except Exception: return "unknown"
    nms={str(n).upper() for n in d.columns}
    if {"FPOGS","FPOGD"}.issubset(nms) and "FPOGID" in d and not d["FPOGID"].duplicated().any() and not re.search(r"(?i)_all[_ -]?gaze",p.name): return "fixations"
    has_gaze=bool(nms & {"BPOGX","FPOGX","LPOGX","RPOGX"})
    has_bio=any(re.match(r"^(HR|GSR|EDA|DIAL|IBI)",n) for n in nms)
    if has_gaze and has_bio: return "combined_biometrics"
    if has_gaze: return "gaze"
    return "unknown"


def gp_profile_export(path):
    p=Path(path); files=[q for q in p.iterdir() if q.is_file()] if p.is_dir() else [p]
    rows=[]
    for f in files:
        try: d=_read_delimited(f,nrows=10); columns="|".join(map(str,d.columns)); ncols=d.shape[1]
        except Exception: columns=pd.NA; ncols=pd.NA
        rows.append({"file":str(f.resolve()),"export_type":gp_identify_export_type(f),"columns":columns,"n_columns":ncols,"size_bytes":f.stat().st_size if f.exists() else np.nan,"confidence":is_gazepoint_export(f)})
    out=pd.DataFrame(rows); out.attrs["path_type"]=gp_identify_export_type(path); return out


def gp_list_export_fields(path):
    p=Path(path)
    if p.is_dir():
        vals=[]
        for f in p.iterdir():
            if f.suffix.lower() in {".csv",".txt",".tsv"}:
                vals.extend(gp_list_export_fields(f))
        return list(dict.fromkeys(vals))
    return list(map(str,_read_delimited(p,nrows=1).columns))


def gp_validate_export(path):
    profile=gp_profile_export(path); rows=[]
    if profile.empty: rows.append({"severity":"error","code":"no_files","file":str(path),"message":"No readable files found."})
    for row in profile.itertuples(index=False):
        if float(row.confidence)<.5: rows.append({"severity":"warning","code":"low_format_confidence","file":row.file,"message":"File does not strongly match known Gazepoint fields."})
    return pd.DataFrame(rows,columns=["severity","code","file","message"])


def _gp_apply_timebase(out,data,info,origin_tick=None):
    if not info["tick_col"] or not np.isfinite(info["tick_frequency"]): return out
    origin=_gp_tick_origin(data,info) if origin_tick is None else origin_tick
    if not np.isfinite(origin): return out
    freq=float(info["tick_frequency"])
    for table in ["gaze_samples","eye_samples","biometrics","events"]:
        d=out[table].copy()
        if not d.empty and "timestamp_native" in d:
            d["timestamp_seconds"]=(pd.to_numeric(d.timestamp_native,errors="coerce")-origin)/freq
            out[table]=d
    if not out["recordings"].empty:
        out["recordings"].loc[:,"recording_start"]=info["recording_start"]
        rate=pd.to_numeric(out["recordings"].nominal_sampling_rate,errors="coerce")
        out["recordings"].loc[~np.isfinite(rate),"nominal_sampling_rate"]=60
    if not out["streams"].empty:
        out["streams"].loc[:,"timestamp_unit"]="ticks"; out["streams"].loc[:,"source_clock"]="Gazepoint TIMETICK"
        for idx,row in out["streams"].iterrows():
            if row.stream_type=="gaze_combined":
                z=out["gaze_samples"].loc[out["gaze_samples"].recording_id==row.recording_id,"timestamp_seconds"]
                out["streams"].loc[idx,"observed_rate_hz"]=estimate_sampling_rate(z)
    out.vendor_metadata["gazepoint_timebase"]={"native_clock":info["tick_col"],"tick_frequency":freq,"origin_tick":origin,"recording_start":info["recording_start"],"media_relative_clock":info["media_time_col"],"normalized_clock":"seconds since recording start"}
    return out


def _gp_apply_biometric_validity(out,data):
    if out["biometrics"].empty: return out
    info=_gp_time_info(data)
    validity={"gsr_raw":"GSRV","eda":"GSRV","skin_conductance_level":"GSRV","skin_conductance_response":"GSRV","heart_rate":"HRV","heart_rate_period":"HRV","interbeat_interval":"HRV","engagement_dial":"DIALV"}
    units={"gsr_raw":"vendor_raw","eda":"microsiemens","skin_conductance_level":"microsiemens","skin_conductance_response":"microsiemens","heart_rate":"beats_per_minute","heart_rate_period":"vendor_units","interbeat_interval":"seconds","engagement_dial":"proportion"}
    tick=_safe_numeric(data[info["tick_col"]]) if info["tick_col"] else None
    for channel in pd.unique(out["biometrics"].channel):
        idx=out["biometrics"].channel.eq(channel)
        vcol=validity.get(channel)
        if tick is not None and vcol in data.columns:
            lookup=dict(zip(tick,_safe_logical(data[vcol])))
            out["biometrics"].loc[idx,"valid"]=[lookup.get(v,pd.NA) for v in out["biometrics"].loc[idx,"timestamp_native"]]
        if channel in units: out["biometrics"].loc[idx,"unit"]=units[channel]
    if not out["streams"].empty:
        for channel,unit in units.items(): out["streams"].loc[out["streams"].stream_type.eq(channel),"value_unit"]=unit
    return out


def gp_parse_user_events(x):
    if not is_eye_dataset(x): raise TypeError("Expected an `eye_dataset` object.")
    raw=x.raw.get("gazepoint") if isinstance(x.raw,dict) else None
    if raw is None and isinstance(x.raw,dict): raw=x.raw.get("generic")
    if not isinstance(raw,pd.DataFrame): return x
    col=_first_existing(list(raw.columns),["USER_DATA","MARKER","EVENT"]); time_col=_gp_pick(raw.columns,"time")
    if not col or not time_col: return x
    values=raw[col].astype("string"); keep=values.notna()&values.ne("")
    if not keep.any(): return x
    rec_id=str(x["recordings"].recording_id.iloc[0]) if len(x["recordings"])==1 else None
    if rec_id is None: return x
    rows=[]
    for j,i in enumerate(np.flatnonzero(keep.to_numpy()),1):
        t=float(_safe_numeric(raw[time_col]).iloc[i]) if np.isfinite(_safe_numeric(raw[time_col]).iloc[i]) else np.nan
        rows.append({"event_id":f"{rec_id}_gp_user_{j:07d}","recording_id":rec_id,"timestamp_native":t,"timestamp_seconds":t,"event_type":"user_data","event_name":values.iloc[i],"event_value":values.iloc[i],"duration":np.nan,"source":"Gazepoint USER_DATA","native_record":values.iloc[i],"trial_id":pd.NA,"stimulus_id":pd.NA})
    ev=pd.DataFrame(rows)
    existing=set(zip(x["events"].recording_id.astype(str),x["events"].timestamp_seconds.astype(str),x["events"].event_name.astype(str))) if not x["events"].empty else set()
    ev=ev[[ (str(r.recording_id),str(r.timestamp_seconds),str(r.event_name)) not in existing for r in ev.itertuples(index=False) ]]
    if not ev.empty: x["events"]=standardize_eye_table(pd.concat([x["events"],ev],ignore_index=True,sort=False),"events")
    return x


def gp_parse_media_events(x):
    if not is_eye_dataset(x): raise TypeError("Expected an `eye_dataset` object.")
    if x["gaze_samples"].empty or x["gaze_samples"].stimulus_id.isna().all(): return x
    d=x["gaze_samples"].sort_values(["recording_id","timestamp_seconds"],kind="stable").reset_index(drop=True)
    changed=d.groupby("recording_id",sort=False).stimulus_id.transform(lambda z:z.ne(z.shift(1))) & d.stimulus_id.notna()
    rows=[]
    for j,row in enumerate(d.loc[changed].itertuples(index=False),1):
        rows.append({"event_id":f"{row.recording_id}_media_{j:07d}","recording_id":row.recording_id,"timestamp_native":row.timestamp_native,"timestamp_seconds":row.timestamp_seconds,"event_type":"media_change","event_name":"MEDIA_START","event_value":row.stimulus_id,"duration":np.nan,"source":"Gazepoint MEDIA_ID","native_record":pd.NA,"trial_id":row.trial_id,"stimulus_id":row.stimulus_id})
    if rows: x["events"]=standardize_eye_table(pd.concat([x["events"],pd.DataFrame(rows)],ignore_index=True,sort=False).drop_duplicates(subset=["recording_id","timestamp_seconds","event_type","event_value"],keep="first"),"events")
    return x


def read_gazepoint(path,participant_id=None,recording_id=None,session_id="S001",nominal_sampling_rate=60,screen_width=np.nan,screen_height=np.nan,keep_raw=True,quiet=False,**kwargs):
    """Import Gazepoint Analysis 7.x samples with frozen 0.11.1 semantics."""
    p=Path(path)
    if p.is_dir(): return read_gazepoint_folder(p,participant_id=participant_id,recording_id=recording_id,session_id=session_id,keep_raw=keep_raw,quiet=quiet,**kwargs)
    typ=gp_identify_export_type(p)
    if typ=="fixations": return read_gazepoint_fixations(p,participant_id=participant_id,recording_id=recording_id,session_id=session_id,keep_raw=keep_raw,quiet=quiet,**kwargs)
    data=_read_delimited(p,**kwargs); identity=_gp_filename_identity(p,participant_id,recording_id,session_id); info=_gp_time_info(data); mapping=_gp_mapping(data)
    if not all(k in mapping for k in ("timestamp","x","y")): raise ValueError("Gazepoint sample export is missing identifiable time and gaze-coordinate columns.")
    pupil_unit="millimetres" if any(c in data.columns for c in ["LPMM","RPMM","LPMMV","RPMMV"]) else "vendor_units"
    out=read_eye_generic(data,mapping=mapping,time_unit="ticks" if info["tick_col"] else "seconds",coordinate_space="display_normalized_top_left",screen_width=screen_width,screen_height=screen_height,pupil_unit=pupil_unit,vendor="Gazepoint",recording_id=identity["recording_id"],participant_id=identity["participant_id"],session_id=identity["session_id"],nominal_sampling_rate=nominal_sampling_rate,keep_raw=keep_raw,quiet=True)
    out=_gp_apply_timebase(out,data,info); out=_gp_apply_biometric_validity(out,data)
    out["recordings"].loc[:,"device_model"]="Gazepoint"; out["recordings"].loc[:,"software_name"]="Gazepoint Analysis"
    out.vendor_metadata["gazepoint"]={"export_type":typ,"source_columns":list(map(str,data.columns)),"source_file":str(p.resolve()),"source_identity":identity,"coordinate_convention":"normalized top-left; out-of-range values retained","biometric_channels":_gp_biometric_mapping(data)}
    if keep_raw:
        if not isinstance(out.raw,dict): out.raw={}
        out.raw["gazepoint"]=data.copy()
    out=gp_parse_user_events(out); out=gp_parse_media_events(out)
    out=add_provenance(out,"import_gazepoint_v7","dataset",f"Gazepoint export type: {typ}",source_files=str(p))
    out.validation=validate_eye_dataset(out)
    return out


def read_gazepoint_gaze(*args, **kwargs):
    """Gaze-only public alias retained from R ``read_gazepoint_gaze``."""
    return read_gazepoint(*args, **kwargs)


def _gp_aoi_id(media_id,aoi):
    vals=[]
    for m,a in zip(pd.Series(media_id),pd.Series(aoi)):
        if pd.isna(a) or not str(a).strip(): vals.append(pd.NA); continue
        numeric=re.sub(r"(?i)^AOI\s*","",str(a).strip())
        vals.append(f"media_{_gp_id_token(m)}_aoi_{_gp_id_token(numeric)}")
    return pd.Series(vals,dtype="string")


def read_gazepoint_fixations(path,participant_id=None,recording_id=None,session_id="S001",origin_tick=None,keep_raw=True,quiet=False,**kwargs):
    data=_read_delimited(path,**kwargs); nms=list(data.columns); identity=_gp_filename_identity(path,participant_id,recording_id,session_id); info=_gp_time_info(data)
    pick=lambda *xs:_first_existing(nms,list(xs))
    duration=_safe_numeric(_map_column(data,{"duration":pick("FPOGD","FIXATION_DURATION","duration")},"duration"))
    finite=duration[np.isfinite(duration)]
    duration_ms=duration*1000 if len(finite)>0 and bool((finite<20).all()) else duration
    tick=_safe_numeric(data[info["tick_col"]]) if info["tick_col"] else pd.Series(np.nan,index=data.index)
    origin=_gp_tick_origin(data,info) if origin_tick is None or not np.isfinite(origin_tick) else origin_tick
    if np.isfinite(origin) and np.isfinite(info["tick_frequency"]):
        end=(tick-origin)/float(info["tick_frequency"]); start=end-duration_ms/1000
    else:
        start=_safe_numeric(_map_column(data,{"start":pick("FPOGS","FIXATION_START","start_time")},"start")); end=start+duration_ms/1000
    fix=_map_column(data,{"fixation_id":pick("FPOGID","FIXATION_ID","fixation_id")},"fixation_id").astype("string")
    missing=fix.isna()|fix.eq(""); fix=fix.copy(); fix.loc[missing]=[f"{i:07d}" for i in range(1,int(missing.sum())+1)]
    stimulus=_as_character_id(_map_column(data,{"stimulus":pick("MEDIA_ID","MEDIA_NAME","stimulus_id")},"stimulus"))
    media_name=_map_column(data,{"stimulus":pick("MEDIA_NAME")},"stimulus").astype("string")
    raw_aoi=_map_column(data,{"aoi":pick("AOI","AOI_ID","aoi_id")},"aoi").astype("string")
    episode_id=pd.Series([f"{identity['recording_id']}_fix_{_gp_id_token(s)}_{_gp_id_token(f)}" for s,f in zip(stimulus,fix)])
    if episode_id.duplicated().any(): episode_id=episode_id+pd.Series([f"_row{i:05d}" for i in range(1,len(episode_id)+1)])
    episodes=pd.DataFrame({"episode_id":episode_id,"recording_id":identity["recording_id"],"episode_type":"fixation","eye":"combined","start_time":start,"end_time":end,"duration_ms":duration_ms,"start_x":np.nan,"start_y":np.nan,"end_x":np.nan,"end_y":np.nan,"centroid_x":_safe_numeric(_map_column(data,{"x":pick("FPOGX","FIXATION_X","x")},"x")),"centroid_y":_safe_numeric(_map_column(data,{"y":pick("FPOGY","FIXATION_Y","y")},"y")),"amplitude":np.nan,"peak_velocity":np.nan,"dispersion":np.nan,"coordinate_space_id":"coord_display_normalized_top_left","source_algorithm":"Gazepoint Analysis","source_parameters":pd.NA,"derived_by":"vendor","trial_id":pd.NA,"stimulus_id":stimulus,"aoi_id":_gp_aoi_id(stimulus,raw_aoi),"timestamp_native":tick,"source_fixation_id":fix,"source_media_time_start":_safe_numeric(_map_column(data,{"start":pick("FPOGS")},"start")),"source_media_time_end":_safe_numeric(data[info["media_time_col"]]) if info["media_time_col"] else np.nan,"source_media_name":media_name,"source_aoi_label":raw_aoi})
    recordings=pd.DataFrame([{"recording_id":identity["recording_id"],"participant_id":identity["participant_id"],"session_id":identity["session_id"],"vendor":"Gazepoint","vendor_family":"Gazepoint","device_model":"Gazepoint","firmware_version":pd.NA,"software_name":"Gazepoint Analysis","software_version":pd.NA,"experiment_type":pd.NA,"nominal_sampling_rate":60,"screen_width_px":np.nan,"screen_height_px":np.nan,"recording_start":info["recording_start"],"source_timezone":pd.NA,"source_file_set":str(Path(path).resolve())}])
    out=new_eye_dataset(recordings=recordings,episodes=episodes,coordinate_spaces=new_coordinate_space("coord_display_normalized_top_left","display_normalized_top_left"),raw={"gazepoint_fixations":data.copy()} if keep_raw else {},vendor_metadata={"gazepoint_fixations":{"source_columns":list(map(str,data.columns)),"source_identity":identity,"timebase":{"tick_col":info["tick_col"],"tick_frequency":info["tick_frequency"],"origin_tick":origin}}},validate=False)
    out=add_provenance(out,"import_gazepoint_fixations_v7","episodes",f"{len(episodes)} fixations",source_files=str(path)); out.validation=validate_eye_dataset(out); return out


def gp_pair_exports(path):
    p=Path(path)
    if not p.is_dir(): raise FileNotFoundError(f"Directory does not exist: {path}")
    files=[f for f in p.iterdir() if f.is_file() and f.suffix.lower() in {".csv",".txt",".tsv"}]
    rows=[]
    for f in files:
        typ=gp_identify_export_type(f); base=f.stem
        group=re.sub(r"(?i)(?:[_ -](?:all[_ -]?gaze|fixations?|user(?:[_ -]?fix)?))$","",base)
        if typ=="aoi_statistics": group="gazepoint_data_summary"
        ident=_gp_filename_identity(f)
        rows.append({"group":group,"participant_id":ident["participant_id"],"file":str(f.resolve()),"export_type":typ})
    return pd.DataFrame(rows,columns=["group","participant_id","file","export_type"])


def gp_match_recordings(path): return gp_pair_exports(path)

def gp_match_biometrics(path):
    pairs=gp_pair_exports(path); return pairs[pairs.export_type.isin(["combined_biometrics","gaze"])].reset_index(drop=True)


def gp_audit_file_pairs(path):
    pairs=gp_pair_exports(path); rows=[]
    nonsummary=pairs[pairs.export_type.ne("aoi_statistics")]
    for group,d in nonsummary.groupby("group",sort=False):
        has_gaze=d.export_type.isin(["gaze","combined_biometrics"]).any()
        rows.append({"group":group,"participant_id":d.participant_id.iloc[0],"has_gaze":bool(has_gaze),"has_fixations":bool(d.export_type.eq("fixations").any()),"has_biometrics":bool(d.export_type.eq("combined_biometrics").any()),"n_files":len(d),"status":"usable" if has_gaze else "incomplete"})
    out=pd.DataFrame(rows); out.attrs["summary_reports"]=int(pairs.export_type.eq("aoi_statistics").sum()) if not pairs.empty else 0; return out


def read_gazepoint_folder(path,include=("gaze","fixations","events","biometrics","aoi"),participant_id=None,recording_id=None,session_id="S001",keep_raw=True,recursive=False,quiet=False,**kwargs):
    p=Path(path)
    if not p.is_dir(): raise FileNotFoundError(f"Directory does not exist: {path}")
    files=[f for f in (p.rglob('*') if recursive else p.iterdir()) if f.is_file() and f.suffix.lower() in {".csv",".txt",".tsv"}]
    if not files: raise ValueError("No delimited Gazepoint exports found.")
    profile=gp_pair_exports(p); objs=[]; user=profile[profile.export_type.isin(["gaze","combined_biometrics","fixations"])]
    groups=list(pd.unique(user.group))
    if recording_id is not None and len(groups)>1: raise ValueError("recording_id can only be supplied when a Gazepoint folder contains one recording group; omit it for multi-recording folders so identifiers can be derived from filenames.")
    include=set(include)
    for group in groups:
        d=user[user.group.eq(group)]; gaze=d[d.export_type.isin(["gaze","combined_biometrics"])]
        if not gaze.empty and bool(include & {"gaze","biometrics"}):
            gf=Path(gaze.file.iloc[0]); ident=_gp_filename_identity(gf,participant_id,recording_id,session_id)
            g=read_gazepoint(gf,participant_id=ident["participant_id"],recording_id=ident["recording_id"],session_id=ident["session_id"],keep_raw=keep_raw,quiet=True,**kwargs); objs.append(g)
            if "fixations" in include:
                for ff in d.loc[d.export_type.eq("fixations"),"file"]: objs.append(read_gazepoint_fixations(ff,participant_id=ident["participant_id"],recording_id=ident["recording_id"],session_id=ident["session_id"],keep_raw=keep_raw,quiet=True,**kwargs))
        elif "fixations" in include:
            for ff in d.loc[d.export_type.eq("fixations"),"file"]: objs.append(read_gazepoint_fixations(ff,participant_id=participant_id,recording_id=recording_id,session_id=session_id,keep_raw=keep_raw,quiet=True,**kwargs))
    if not objs: raise ValueError("No requested Gazepoint export types were found.")
    out=combine_eye_datasets(objs,resolve_ids=False); out.vendor_metadata["gazepoint_folder"]=profile
    out=add_provenance(out,"import_gazepoint_folder_v7","dataset",f"Files: {len(files)}; paired groups: {len(groups)}",source_files=[str(f) for f in files]); out.validation=validate_eye_dataset(out); return out


def read_gazepoint_biometrics(path,participant_id=None,recording_id=None,session_id="S001",keep_raw=True,quiet=False,**kwargs):
    data=_read_delimited(path,**kwargs); channels=_gp_biometric_mapping(data)
    if not channels: raise ValueError("No recognized Gazepoint biometric channels found.")
    if any(c in data.columns for c in ["BPOGX","FPOGX","LPOGX","RPOGX"]):
        out=read_gazepoint(path,participant_id=participant_id,recording_id=recording_id,session_id=session_id,keep_raw=keep_raw,quiet=True,**kwargs)
        out["gaze_samples"]=empty_eye_table("gaze_samples"); out["episodes"]=empty_eye_table("episodes"); out["events"]=empty_eye_table("events"); out["responses"]=empty_eye_table("responses")
        out["streams"]=out["streams"][out["streams"].stream_type.ne("gaze_combined")].reset_index(drop=True)
        out=add_provenance(out,"select_gazepoint_biometrics","biometrics|eye_samples",",".join(channels),source_files=str(path)); out.validation=validate_eye_dataset(out); return out
    identity=_gp_filename_identity(path,participant_id,recording_id,session_id)
    info=_gp_time_info(data)
    time_col=info["tick_col"] or info["media_time_col"] or _gp_pick(data.columns,"time")
    if time_col is None: raise ValueError("Cannot identify a Gazepoint time column.")
    data=data.copy(); data[".eye_dummy_x"]=np.nan; data[".eye_dummy_y"]=np.nan
    mapping=eye_mapping(timestamp=time_col,x=".eye_dummy_x",y=".eye_dummy_y",pupil_left=_gp_pick(data.columns,"pupil_left"),pupil_right=_gp_pick(data.columns,"pupil_right"),pupil_left_valid=_gp_pick(data.columns,"pupil_left_valid"),pupil_right_valid=_gp_pick(data.columns,"pupil_right_valid"),trial=_gp_pick(data.columns,"trial"),stimulus=_gp_pick(data.columns,"stimulus"),biometric_channels=channels)
    out=read_eye_generic(data,mapping=mapping,vendor="Gazepoint Biometrics",participant_id=identity["participant_id"],recording_id=identity["recording_id"],session_id=identity["session_id"],time_unit="ticks" if info["tick_col"] else "seconds",coordinate_space="display_normalized_top_left",pupil_unit="millimetres" if any(c in data.columns for c in ["LPMM","RPMM"]) else "vendor_units",keep_raw=keep_raw,quiet=True)
    out=_gp_apply_timebase(out,data,info); out=_gp_apply_biometric_validity(out,data)
    out["gaze_samples"]=empty_eye_table("gaze_samples"); out["streams"]=out["streams"][out["streams"].stream_type.ne("gaze_combined")].reset_index(drop=True)
    out.vendor_metadata["gazepoint_biometrics"]={"source_columns":list(map(str,data.columns)),"channels":channels}
    out=add_provenance(out,"import_gazepoint_biometrics_v7","biometrics|eye_samples",",".join(channels),source_files=str(path)); out.validation=validate_eye_dataset(out); return out


def read_gazepoint_combined(gaze,fixations=None,biometrics=None,**kwargs):
    g=read_gazepoint(gaze,**kwargs); xs=[g]; ident={"participant_id":g["recordings"].participant_id.iloc[0],"recording_id":g["recordings"].recording_id.iloc[0],"session_id":g["recordings"].session_id.iloc[0]}
    if fixations is not None: xs.append(read_gazepoint_fixations(fixations,participant_id=ident["participant_id"],recording_id=ident["recording_id"],session_id=ident["session_id"],quiet=True))
    if biometrics is not None: xs.append(read_gazepoint_biometrics(biometrics,participant_id=ident["participant_id"],recording_id=ident["recording_id"],session_id=ident["session_id"],quiet=True))
    return combine_eye_datasets(xs,resolve_ids=False)


def read_gazepoint_events(path,**kwargs):
    x=read_gazepoint(path,**kwargs); x["gaze_samples"]=empty_eye_table("gaze_samples"); x["eye_samples"]=empty_eye_table("eye_samples"); x["biometrics"]=empty_eye_table("biometrics"); return x
