# Importing and Harmonizing Eye-Tracking Exports

The adapter layer converts source-specific exports into explicit canonical records, streams, samples, eye samples, episodes, events, intervals, AOIs, biometrics, quality results, and provenance.

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/importing-and-harmonizing.Rmd`.

## Automatic detection

```python
import eyeprocesspy as ep

fmt = ep.detect_eye_format("participant-01.tsv")
x = ep.read_eye_export("participant-01.tsv", vendor="auto")
```

Detection is intentionally conservative. If no adapter is sufficiently confident, use an explicit mapping rather than guessing source semantics.

## Generic mapping

```python
mapping = ep.eye_mapping(
    participant="subject",
    recording="recording",
    timestamp="timestamp_us",
    x="gaze_x",
    y="gaze_y",
    pupil_left="pupil_left_mm",
    pupil_right="pupil_right_mm",
    trial="trial_id",
    item="item_id",
    response="answer",
    score="correct",
    response_time="response_time_ms",
)

x = ep.read_eye_generic(
    "export.csv",
    mapping=mapping,
    time_unit="microseconds",
    coordinate_space="display_pixels_top_left",
    screen_width=1920,
    screen_height=1080,
    pupil_unit="millimetres",
)
```

## Dedicated vendor adapters

```python
tobii = ep.read_tobii("tobii-pro-lab.tsv")
neon = ep.read_pupil_neon("neon-export-folder")
core = ep.read_pupil_core("pupil-player-export")
eyelink = ep.read_eyelink_asc("recording.asc")
smi = ep.read_smi("begaze-export.txt")
```

The canonical representation does not imply that all vendors provide equivalent fields, clocks, validity semantics, events, or coordinate systems.

## Preserve source meaning

```python
x.raw
x.vendor_metadata
x.coordinate_spaces
x.streams
x.provenance

ep.audit_coordinate_spaces(x)
ep.audit_timebase(x)
ep.audit_clock_sync(x)
```

No coordinate conversion, interpolation, resampling, or event reclassification should occur silently. Native source fields and transformations remain available for audit so harmonization is semantic rather than merely a column-renaming exercise.
