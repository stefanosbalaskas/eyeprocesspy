# Gazepoint and Gazepoint Biometrics Workflows

Gazepoint is a first-class source while the downstream representation remains vendor-neutral.

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/gazepoint-workflows.Rmd`.

## Profile before import

```python
import eyeprocesspy as ep

ep.gp_profile_export("data/P001")
ep.gp_audit_file_pairs("data/P001")
ep.gp_list_export_fields("data/P001/P001-user.csv")
ep.gp_validate_export("data/P001")
```

## Folder import

```python
x = ep.read_gazepoint_folder(
    "data/P001",
    include=("gaze", "fixations", "events", "biometrics", "aoi"),
    participant_id="P001",
)

x = ep.gp_reconstruct_trials(
    x,
    start_events=("TRIAL_START", "START_TRIAL"),
    end_events=("TRIAL_END", "END_TRIAL"),
)

x = ep.gp_reconstruct_stimuli(x)
x = ep.gp_align_media_ids(x)
```

## Gazepoint-specific audits

```python
ep.gp_check_sampling_rate(x)
ep.gp_check_validity_fields(x)
ep.gp_check_fixation_ids(x)
ep.gp_check_media_timing(x)
ep.gp_check_pupil_channels(x)
ep.gp_check_biometrics_sync(x)
```

## Separate biometrics and synchronization

```python
gaze = ep.read_gazepoint_gaze("P001-user.csv")
bio = ep.read_gazepoint_biometrics("P001-biometrics.csv")

source_markers = bio.events.loc[
    bio.events["event_name"] == "SYNC", "timestamp_seconds"
].to_numpy()
target_markers = gaze.events.loc[
    gaze.events["event_name"] == "SYNC", "timestamp_seconds"
].to_numpy()

x = ep.synchronize_eye_biometrics(
    gaze,
    bio,
    source_markers=source_markers,
    target_markers=target_markers,
    method="linear",
)
```

Different native sampling rates and clocks are preserved. Alignment parameters are recorded in provenance rather than hidden by automatic resampling.

## Gazepoint Analysis 7.2.0 paired exports

Gazepoint Analysis 7.2.0 may export files named `User 3_all_gaze.csv` and `User 3_fixations.csv`, together with multi-section `Data_Summary_export_*.csv` reports. The folder importer pairs these files by their `User N` stem.

```python
root = "C:/path/to/gazepoint-export-folder"

ep.gp_pair_exports(root)
x = ep.read_gazepoint_folder(root)
```

The sample export contains two clocks with different meanings. `TIMETICK(f=10000000)` remains monotonic across the recording and is used to construct zero-based `timestamp_seconds`. `TIME(...)` restarts when the media item changes and is retained as `media_time_seconds`. Neither clock is silently discarded.

Fixation identifiers can restart for each media item. `eyeprocesspy` therefore constructs canonical episode identifiers from recording, media, and source fixation identity while retaining the original identifier as `source_fixation_id`.

```python
summary = ep.read_gazepoint_summary(
    f"{root}/Data_Summary_export_02-20-26-01.28.43.csv"
)
aoi_data = ep.read_gazepoint_aoi_statistics(summary.path)
```

The Data Summary parser retains both aggregate AOI information and per-user AOI statistics. Canonical AOI definitions and participant-AOI features are created without inventing spatial geometry that is absent from the source report.
