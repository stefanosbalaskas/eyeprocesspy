# Gazepoint import and quality-control workflow

`eyeprocesspy` includes dedicated Gazepoint readers and file-workflow helpers while converting outputs into the same vendor-neutral canonical dataset used by the rest of the package.

## Identify an export before importing it

```python
import eyeprocesspy as ep

kind = ep.gp_identify_export_type("recording.csv")
profile = ep.gp_profile_export("recording.csv")
fields = ep.gp_list_export_fields("recording.csv")
```

This is useful when a study directory mixes gaze, fixation, event, biometric, or other export families.

## Validate the source export

```python
report = ep.gp_validate_export("recording.csv")
```

Validation at the vendor-file stage can detect problems that become harder to diagnose after files are combined.

## Read individual Gazepoint export families

```python
gaze = ep.read_gazepoint_gaze("gaze.csv")
fixations = ep.read_gazepoint_fixations("fixations.csv")
events = ep.read_gazepoint_events("events.csv")
biometrics = ep.read_gazepoint_biometrics("biometrics.csv")
```

Use the most specific reader when you know the export family. Use `read_gazepoint()` or `read_eye_export(..., vendor="gazepoint")` when automatic routing is more appropriate.

## Work with a folder, not file order

```python
pairs = ep.gp_pair_exports("export-folder")
audit = ep.gp_audit_file_pairs(pairs)
eye = ep.read_gazepoint_folder("export-folder")
```

When gaze and biometric exports must be reconciled, use the matching/pairing helpers rather than relying on alphabetical file order or manually constructed participant IDs.

Related helpers include:

- `gp_match_recordings()`;
- `gp_match_biometrics()`;
- `read_gazepoint_combined()`;
- `gp_audit_file_pairs()`.

## Parse task and media events

```python
user_events = ep.gp_parse_user_events(eye)
media_events = ep.gp_parse_media_events(eye)
```

Event parsing must be checked against the experiment's real task protocol. A parser can identify encoded events; it cannot reconstruct an undocumented study design.

## Validate after canonicalization

```python
issues = ep.validate_eye_dataset(eye)
```

Then audit the measurement conditions relevant to the study:

```python
rates = ep.audit_sampling_rate(eye)
quality = ep.audit_signal_quality(eye)
missing = ep.audit_missingness(eye)
spaces = ep.audit_coordinate_spaces(eye)
```

## Visual QC

```python
ax = ep.plot_gaze_heatmap(eye, trial_id="T1")
ax.figure.tight_layout()
```

![Gaze heatmap](../assets/gallery/gaze-heatmap.svg)

Also inspect traces, fixations, scanpaths, pupil streams, and trial/event timing when those data are present.

## Preserve device and export provenance

For Gazepoint-specific research, retain:

- Gazepoint software/export version where available;
- the exact files associated with each recording;
- whether vendor fixations or eyeprocess-derived fixations were analyzed;
- coordinate space and stimulus geometry;
- expected vs empirical sampling context;
- event parsing rules;
- biometric synchronization/matching decisions;
- exclusions and QC thresholds.

## Recommended route

1. profile/identify source files;
2. validate vendor files;
3. pair related exports;
4. import into the canonical model;
5. validate the canonical model;
6. audit sampling, missingness, coordinates, and events;
7. inspect plots;
8. derive features only after the observation pipeline is understood;
9. preserve provenance with the analysis outputs.

[Continue to the end-to-end workflow](end-to-end-eye-tracking.md){ .md-button }
