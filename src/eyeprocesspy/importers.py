from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import re
import numpy as np
import pandas as pd

from .dataset import add_provenance, new_eye_dataset, validate_eye_dataset
from .mapping import eye_mapping
from .schema import empty_eye_table, new_coordinate_space
from .timebase import estimate_sampling_rate

_TIME_MULTIPLIERS = {
    "seconds": 1.0, "second": 1.0, "s": 1.0,
    "milliseconds": 1e-3, "millisecond": 1e-3, "ms": 1e-3,
    "microseconds": 1e-6, "microsecond": 1e-6, "us": 1e-6,
    "nanoseconds": 1e-9, "nanosecond": 1e-9, "ns": 1e-9,
    "ticks": 1.0,
}


def _time_multiplier(unit: str) -> float:
    key = str(unit or "seconds").lower()
    if key not in _TIME_MULTIPLIERS:
        raise ValueError(f"Unsupported time unit `{key}`.")
    return _TIME_MULTIPLIERS[key]


def _read_delimited(path: str | Path, delimiter: str | None = None, encoding: str = "UTF-8", **kwargs: Any) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    if delimiter is None:
        with p.open("r", encoding=encoding, errors="replace") as f:
            first = f.readline()
        candidates = ["\t", ",", ";", "|"]
        delimiter = max(candidates, key=first.count)
    read_kwargs = dict(kwargs)
    # R's read.table(fill=TRUE, check.names=FALSE) is most closely matched by
    # pandas' default header handling and Python engine for irregular rows.
    read_kwargs.setdefault("engine", "python")
    return pd.read_csv(p, sep=delimiter, encoding=encoding, **read_kwargs)


def _map_column(data: pd.DataFrame, mapping: Mapping[str, Any], key: str, default: Any = np.nan) -> pd.Series:
    col = mapping.get(key)
    if col is None or (isinstance(col, (list, tuple)) and len(col) == 0):
        return pd.Series([default] * len(data), index=data.index)
    if isinstance(col, (list, tuple)):
        col = col[0]
    if col not in data.columns:
        raise ValueError(f"Mapped column `{col}` for `{key}` does not exist.")
    return data[col]


def _safe_numeric(x: pd.Series | Any) -> pd.Series:
    s = x if isinstance(x, pd.Series) else pd.Series(x)
    if not pd.api.types.is_numeric_dtype(s):
        s = s.astype("string").str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _safe_logical(x: pd.Series | Any) -> pd.Series:
    s = x if isinstance(x, pd.Series) else pd.Series(x)
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")
    z = s.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out[z.isin(["1", "true", "t", "yes", "y", "valid"])] = True
    out[z.isin(["0", "false", "f", "no", "n", "invalid"])] = False
    return out


def _as_character_id(x: pd.Series | Any) -> pd.Series:
    s = x if isinstance(x, pd.Series) else pd.Series(x)
    # R as.character(NA) remains NA; pandas StringDtype gives equivalent missing semantics.
    return s.astype("string")


def validate_eye_mapping(mapping: Mapping[str, Any], data: pd.DataFrame | None = None, required=("timestamp", "x", "y")):
    """Validate a generic import mapping (R ``validate_eye_mapping``)."""
    if not isinstance(mapping, Mapping):
        raise TypeError("`mapping` must be an eye mapping object or mapping.")
    missing = [key for key in required if key not in mapping]
    if missing:
        raise ValueError(f"Required mapping field(s) absent: {', '.join(missing)}.")
    if data is not None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be a pandas DataFrame.")
        mapped: list[str] = []
        for value in mapping.values():
            if isinstance(value, Mapping):
                mapped.extend(str(v) for v in value.values())
            elif isinstance(value, (list, tuple, set)):
                mapped.extend(str(v) for v in value)
            elif value is not None:
                mapped.append(str(value))
        absent = sorted(set(mapped) - set(map(str, data.columns)))
        if absent:
            raise ValueError(f"Mapped source column(s) absent: {', '.join(absent)}.")
    return mapping


def _first_existing(names: list[str], candidates: list[str]) -> str | None:
    lower = {str(n).lower(): str(n) for n in names}
    for candidate in candidates:
        found = lower.get(candidate.lower())
        if found is not None:
            return found
    return None


def infer_eye_mapping(data: pd.DataFrame, vendor: str | None = None):
    """Infer common eye/process fields using the frozen R candidate order."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    nms = list(map(str, data.columns))
    pick = lambda *xs: _first_existing(nms, list(xs))
    return eye_mapping(
        participant=pick("participant_id", "participant", "subject", "subject_id", "user", "USER", "Recording participant"),
        recording=pick("recording_id", "recording", "Recording name", "recording id", "Recording UUID"),
        session=pick("session_id", "session", "session_name"),
        timestamp=pick("timestamp_seconds", "timestamp", "time", "TIME", "TIMETICK", "Recording timestamp", "gaze_timestamp", "world_timestamp", "timestamp [ns]"),
        x=pick("gaze_x", "x", "FPOGX", "BPOGX", "Gaze point X", "Gaze2d x", "norm_pos_x", "x [px]"),
        y=pick("gaze_y", "y", "FPOGY", "BPOGY", "Gaze point Y", "Gaze2d y", "norm_pos_y", "y [px]"),
        gaze_valid=pick("gaze_valid", "valid", "FPOGV", "BPOGV", "Validity left", "Validity right"),
        confidence=pick("confidence", "Confidence"),
        pupil_left=pick("pupil_left", "LPMM", "LPMMV", "Pupil diameter left", "Pupil diameter left [mm]", "diameter_3d_left"),
        pupil_right=pick("pupil_right", "RPMM", "RPMMV", "Pupil diameter right", "Pupil diameter right [mm]", "diameter_3d_right"),
        fixation_id=pick("fixation_id", "FPOGID", "Fixation index", "fixation id"),
        trial=pick("trial_id", "trial", "TRIAL_INDEX", "Trial", "trial number"),
        stimulus=pick("stimulus_id", "stimulus", "MEDIA_ID", "Presented Stimulus name", "world_index"),
        response=pick("response", "response_value", "answer"),
        score=pick("score", "correct", "accuracy"),
        response_time=pick("response_time", "rt", "reaction_time"),
        event_name=pick("event_name", "event", "Event", "USER_DATA", "message"),
        event_value=pick("event_value", "value", "Event value", "message_value"),
    )


def _make_recordings(data: pd.DataFrame, mapping: Mapping[str, Any], vendor: str, source_file: str,
                     nominal_sampling_rate=np.nan, screen_width=np.nan, screen_height=np.nan) -> pd.DataFrame:
    participant = _as_character_id(_map_column(data, mapping, "participant", "P001"))
    recording = _as_character_id(_map_column(data, mapping, "recording", pd.NA))
    session = _as_character_id(_map_column(data, mapping, "session", "S001"))
    missing = recording.isna() | recording.fillna("").eq("")
    recording = recording.copy()
    recording.loc[missing] = "rec_" + participant.loc[missing].fillna("<NA>") + "_" + session.loc[missing].fillna("<NA>")
    tmp = pd.DataFrame({"recording": recording, "participant": participant, "session": session})
    keep = ~tmp.duplicated()
    n = int(keep.sum())
    return pd.DataFrame({
        "recording_id": recording[keep].reset_index(drop=True),
        "participant_id": participant[keep].reset_index(drop=True),
        "session_id": session[keep].reset_index(drop=True),
        "vendor": [str(vendor)] * n,
        "vendor_family": [str(vendor)] * n,
        "device_model": [pd.NA] * n,
        "firmware_version": [pd.NA] * n,
        "software_name": [pd.NA] * n,
        "software_version": [pd.NA] * n,
        "experiment_type": [pd.NA] * n,
        "nominal_sampling_rate": [float(nominal_sampling_rate)] * n,
        "screen_width_px": [float(screen_width)] * n,
        "screen_height_px": [float(screen_height)] * n,
        "recording_start": [pd.NA] * n,
        "source_timezone": [pd.NA] * n,
        "source_file_set": [source_file] * n,
    })


def _resolve_recording_vector(data: pd.DataFrame, mapping: Mapping[str, Any]) -> pd.Series:
    participant = _as_character_id(_map_column(data, mapping, "participant", "P001"))
    session = _as_character_id(_map_column(data, mapping, "session", "S001"))
    recording = _as_character_id(_map_column(data, mapping, "recording", pd.NA)).copy()
    missing = recording.isna() | recording.fillna("").eq("")
    recording.loc[missing] = "rec_" + participant.loc[missing].fillna("<NA>") + "_" + session.loc[missing].fillna("<NA>")
    return recording


def _make_eye_samples(data, mapping, rec_vec, time_native, time_seconds, pupil_unit):
    rows=[]
    for eye in ("left", "right"):
        pupil_key=f"pupil_{eye}"; x_key=f"{eye}_x"; y_key=f"{eye}_y"; valid_key=f"pupil_{eye}_valid"; gaze_valid_key=f"{eye}_valid"
        if not any(k in mapping for k in (pupil_key, x_key, y_key, valid_key, gaze_valid_key)):
            continue
        pupil=_safe_numeric(_map_column(data,mapping,pupil_key))
        pvalid=_safe_logical(_map_column(data,mapping,valid_key,pd.NA))
        if pvalid.isna().all():
            pvalid=pd.Series(np.isfinite(pupil.to_numpy(dtype=float)), index=data.index, dtype="boolean")
        seq=pd.Series(np.arange(1,len(data)+1),index=data.index).groupby(rec_vec.astype(str),sort=False).cumcount()+1
        rows.append(pd.DataFrame({
            "recording_id":rec_vec.astype("string"),
            "sample_id":rec_vec.astype("string")+f"_eye_{eye}_"+seq.map(lambda z:f"{z:09d}"),
            "timestamp_native":time_native,"timestamp_seconds":time_seconds,"eye":eye,
            "pupil_diameter":pupil,"pupil_unit":pupil_unit,"pupil_valid":pvalid,"eye_openness":np.nan,
            "gaze_origin_x":_safe_numeric(_map_column(data,mapping,f"{eye}_origin_x")),
            "gaze_origin_y":_safe_numeric(_map_column(data,mapping,f"{eye}_origin_y")),
            "gaze_origin_z":_safe_numeric(_map_column(data,mapping,f"{eye}_origin_z")),
            "gaze_origin_valid":_safe_logical(_map_column(data,mapping,gaze_valid_key,pd.NA)),
            "corneal_reflection_x":np.nan,"corneal_reflection_y":np.nan,"detector_method":pd.NA,
            "confidence":_safe_numeric(_map_column(data,mapping,"confidence")),
            "trial_id":_as_character_id(_map_column(data,mapping,"trial")),
            "stimulus_id":_as_character_id(_map_column(data,mapping,"stimulus")),
        }))
    return pd.concat(rows,ignore_index=True) if rows else empty_eye_table("eye_samples")


def _make_events(data,mapping,rec_vec,time_native,time_seconds):
    if "event_name" not in mapping: return empty_eye_table("events")
    event_name=_map_column(data,mapping,"event_name").astype("string")
    keep=event_name.notna() & event_name.ne("")
    if not keep.any(): return empty_eye_table("events")
    idx=np.flatnonzero(keep.to_numpy())
    return pd.DataFrame({
        "event_id":[f"{rec_vec.iloc[i]}_event_{j:07d}" for j,i in enumerate(idx,1)],
        "recording_id":rec_vec.iloc[idx].astype("string").to_numpy(),
        "timestamp_native":time_native.iloc[idx].to_numpy(),"timestamp_seconds":time_seconds.iloc[idx].to_numpy(),
        "event_type":_map_column(data,mapping,"event_type","marker").astype("string").iloc[idx].to_numpy(),
        "event_name":event_name.iloc[idx].to_numpy(),
        "event_value":_map_column(data,mapping,"event_value").astype("string").iloc[idx].to_numpy(),
        "duration":np.nan,"source":"generic_import","native_record":pd.NA,
        "trial_id":_as_character_id(_map_column(data,mapping,"trial")).iloc[idx].to_numpy(),
        "stimulus_id":_as_character_id(_map_column(data,mapping,"stimulus")).iloc[idx].to_numpy(),
    })


def _make_responses(data,mapping,rec_vec):
    if not any(k in mapping for k in ("response","score","response_time")): return empty_eye_table("responses")
    trial=_as_character_id(_map_column(data,mapping,"trial")); item=_as_character_id(_map_column(data,mapping,"item"))
    participant=_as_character_id(_map_column(data,mapping,"participant","P001"))
    response=_map_column(data,mapping,"response").astype("string")
    score=_safe_numeric(_map_column(data,mapping,"score")); rt=_safe_numeric(_map_column(data,mapping,"response_time"))
    keep=response.notna() | np.isfinite(score) | np.isfinite(rt)
    if not bool(keep.any()): return empty_eye_table("responses")
    tmp=pd.DataFrame({"recording_id":rec_vec[keep].astype("string"),"participant_id":participant[keep],"trial_id":trial[keep],"item_id":item[keep],"response":response[keep],"score":score[keep],"response_time":rt[keep]})
    tmp=tmp.drop_duplicates(subset=["recording_id","trial_id","item_id"],keep="last").reset_index(drop=True)
    tmp["response_id"]=[f"{r}_response_{i:07d}" for i,r in enumerate(tmp.recording_id,1)]
    tmp["response_timestamp"]=np.nan
    tmp["response_type"]=np.where(np.isfinite(pd.to_numeric(tmp.score,errors="coerce")),"scored","observed")
    tmp["valid_response"]=True
    cols=["response_id","recording_id","participant_id","trial_id","item_id","response","score","response_time","response_timestamp","response_type","valid_response"]
    return tmp[cols]


def _canonical_biometric_unit(channel: str) -> str:
    units={"heart_rate":"beats_per_minute","interbeat_interval":"milliseconds","eda":"microsiemens","gsr":"microsiemens","skin_conductance":"microsiemens","skin_conductance_level":"microsiemens","skin_conductance_response":"microsiemens","engagement_dial":"vendor_units"}
    return units.get(str(channel).strip().lower(),"vendor_units")


def _make_biometrics(data,mapping,rec_vec,time_native,time_seconds):
    channels=mapping.get("biometric_channels")
    if not channels: return empty_eye_table("biometrics")
    if not isinstance(channels,Mapping): raise ValueError("`biometric_channels` must be a named mapping of canonical channels to source columns.")
    rows=[]
    for channel,col in channels.items():
        if col not in data.columns: continue
        value=_safe_numeric(data[col])
        rows.append(pd.DataFrame({"recording_id":rec_vec.astype("string"),"stream_id":rec_vec.astype("string")+"_"+str(channel),"timestamp_native":time_native,"timestamp_seconds":time_seconds,"channel":str(channel),"value":value,"unit":_canonical_biometric_unit(channel),"valid":np.isfinite(value),"processing_level":"raw_imported","source_device":pd.NA,"trial_id":_as_character_id(_map_column(data,mapping,"trial")),"stimulus_id":_as_character_id(_map_column(data,mapping,"stimulus"))}))
    return pd.concat(rows,ignore_index=True) if rows else empty_eye_table("biometrics")


def _make_streams(recordings,coordinate_space_id,time_unit,eyes=(),pupil_unit=pd.NA,vendor="generic"):
    base=pd.DataFrame({"stream_id":recordings.recording_id.astype("string")+"_gaze","recording_id":recordings.recording_id.astype("string"),"stream_type":"gaze_combined","source_device":vendor,"source_clock":"native","sampling_type":"sampled","nominal_rate_hz":recordings.nominal_sampling_rate,"observed_rate_hz":np.nan,"timestamp_unit":time_unit,"value_unit":pd.NA,"coordinate_space_id":coordinate_space_id,"processing_level":"raw_imported"})
    rows=[base]
    for eye in [e for e in ("left","right") if e in set(map(str,eyes))]:
        d=base.copy(); d["stream_id"]=d.recording_id.astype("string")+f"_pupil_{eye}"; d["stream_type"]=f"pupil_{eye}"; d["value_unit"]=pupil_unit; d["coordinate_space_id"]=pd.NA; rows.append(d)
    return pd.concat(rows,ignore_index=True)


def _make_biometric_streams(biometrics,recordings,time_unit,vendor="generic"):
    if biometrics.empty: return pd.DataFrame()
    keys=biometrics[["recording_id","stream_id","channel","unit"]].drop_duplicates().reset_index(drop=True)
    rates=[]
    for row in keys.itertuples(index=False):
        z=biometrics.loc[(biometrics.recording_id==row.recording_id)&(biometrics.stream_id==row.stream_id),"timestamp_seconds"]
        rates.append(estimate_sampling_rate(z))
    return pd.DataFrame({"stream_id":keys.stream_id,"recording_id":keys.recording_id,"stream_type":keys.channel,"source_device":vendor,"source_clock":"native","sampling_type":"sampled","nominal_rate_hz":np.nan,"observed_rate_hz":rates,"timestamp_unit":time_unit,"value_unit":keys.unit,"coordinate_space_id":pd.NA,"processing_level":"raw_imported"})


def read_eye_generic(path, mapping=None, delimiter=None, time_unit="seconds", coordinate_space="display_normalized_top_left", screen_width=np.nan, screen_height=np.nan, pupil_unit=pd.NA, vendor="generic", recording_id=None, participant_id=None, session_id="S001", nominal_sampling_rate=np.nan, keep_raw=True, keep_extra=True, encoding="UTF-8", quiet=False, **kwargs):
    """Import a mapped delimited table into the canonical eyeprocess data contract."""
    is_frame=isinstance(path,pd.DataFrame)
    source_path=None if is_frame else str(path)
    data=path.copy() if is_frame else _read_delimited(path,delimiter=delimiter,encoding=encoding,**kwargs)
    data=data.reset_index(drop=True)
    if data.empty: raise ValueError("Input contains no rows.")
    if mapping is None: mapping=infer_eye_mapping(data,vendor=vendor)
    mapping=dict(mapping)
    if recording_id is not None: data[".eye_recording_id"]=recording_id; mapping["recording"]=".eye_recording_id"
    if participant_id is not None: data[".eye_participant_id"]=participant_id; mapping["participant"]=".eye_participant_id"
    if session_id is not None: data[".eye_session_id"]=session_id; mapping["session"]=".eye_session_id"
    validate_eye_mapping(mapping,data,required=("timestamp","x","y"))
    source_label="<data.frame>" if source_path is None else str(Path(source_path).resolve())
    coord_id="coord_"+re.sub(r"[^A-Za-z0-9]+","_",coordinate_space)
    known={"display_normalized_top_left","display_pixels_top_left","surface_normalized_bottom_left","world_camera_pixels","reference_image_pixels","user_coordinates_3d","headset_coordinates_3d","gaze_direction_vector","custom"}
    coords=new_coordinate_space(coord_id,space_type=coordinate_space if coordinate_space in known else "custom",width=screen_width,height=screen_height,reference_object="display" if "display" in coordinate_space else pd.NA)
    recordings=_make_recordings(data,mapping,vendor,source_label,nominal_sampling_rate,screen_width,screen_height)
    rec_vec=_resolve_recording_vector(data,mapping).reset_index(drop=True)
    time_native=_safe_numeric(_map_column(data,mapping,"timestamp")).reset_index(drop=True)
    time_seconds=time_native*_time_multiplier(time_unit)
    valid=_safe_logical(_map_column(data,mapping,"gaze_valid",True)).reset_index(drop=True)
    if valid.isna().all(): valid=pd.Series([True]*len(data),dtype="boolean")
    seq=pd.Series(np.arange(len(data))).groupby(rec_vec.astype(str),sort=False).cumcount()+1
    gaze=pd.DataFrame({"recording_id":rec_vec,"stream_id":rec_vec.astype("string")+"_gaze","sample_id":rec_vec.astype("string")+"_sample_"+seq.map(lambda z:f"{z:09d}"),"timestamp_native":time_native,"timestamp_seconds":time_seconds,"gaze_x":_safe_numeric(_map_column(data,mapping,"x")).reset_index(drop=True),"gaze_y":_safe_numeric(_map_column(data,mapping,"y")).reset_index(drop=True),"gaze_z":_safe_numeric(_map_column(data,mapping,"z")).reset_index(drop=True),"azimuth_deg":np.nan,"elevation_deg":np.nan,"valid":valid,"confidence":_safe_numeric(_map_column(data,mapping,"confidence")).reset_index(drop=True),"fixation_id_source":_as_character_id(_map_column(data,mapping,"fixation_id")).reset_index(drop=True),"blink_id_source":_as_character_id(_map_column(data,mapping,"blink_id")).reset_index(drop=True),"trial_id":_as_character_id(_map_column(data,mapping,"trial")).reset_index(drop=True),"stimulus_id":_as_character_id(_map_column(data,mapping,"stimulus")).reset_index(drop=True),"coordinate_space_id":coord_id})
    eye_samples=_make_eye_samples(data,mapping,rec_vec,time_native,time_seconds,pupil_unit)
    responses=_make_responses(data,mapping,rec_vec)
    events=_make_events(data,mapping,rec_vec,time_native,time_seconds)
    biometrics=_make_biometrics(data,mapping,rec_vec,time_native,time_seconds)
    streams=_make_streams(recordings,coord_id,time_unit,eyes=pd.unique(eye_samples.eye.dropna()) if not eye_samples.empty else (),pupil_unit=pupil_unit,vendor=vendor)
    for rid in recordings.recording_id.astype(str):
        rate=estimate_sampling_rate(time_seconds[rec_vec.astype(str)==rid])
        streams.loc[(streams.recording_id.astype(str)==rid)&(streams.stream_type=="gaze_combined"),"observed_rate_hz"]=rate
    bstreams=_make_biometric_streams(biometrics,recordings,time_unit,vendor)
    if not bstreams.empty: streams=pd.concat([streams,bstreams],ignore_index=True,sort=False)
    raw={"generic":data.copy()} if keep_raw else {}
    metadata={"generic":{"mapping":mapping,"source_columns":list(map(str,data.columns)),"source_file":source_label,"time_unit":time_unit,"coordinate_space":coordinate_space,"keep_extra":bool(keep_extra)}}
    out=new_eye_dataset(recordings=recordings,streams=streams,gaze_samples=gaze,eye_samples=eye_samples,events=events,responses=responses,biometrics=biometrics,coordinate_spaces=coords,raw=raw,vendor_metadata=metadata,validate=False)
    out=add_provenance(out,"import_generic","dataset",f"Imported {len(data)} rows; vendor={vendor}.",source_files=source_label,reversible=bool(keep_raw))
    out.validation=validate_eye_dataset(out)
    return out
