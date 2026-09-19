# Building a Detector Multiverse

Every branch begins with an immutable `EventDetectorSpec`. No universal velocity or dispersion threshold is inserted by the multiverse API.

```python
import eyeprocesspy as ep

ivt30 = ep.define_event_detector_spec(
    "ivt30",
    "ivt",
    velocity_threshold=30,
    minimum_duration_ms=60,
    maximum_gap_ms=75,
    sampling_rate=60,
    coordinate_unit="degrees",
)

idt_a = ep.define_event_detector_spec(
    "idt_A",
    "idt",
    dispersion_threshold=1.2,
    minimum_duration_ms=80,
    sampling_rate=60,
    coordinate_unit="degrees",
)

multiverse = ep.create_detector_multiverse([ivt30, idt_a])
```

## Parameter grids

A grid is appropriate when each level is defensible before seeing the outcome of interest:

```python
base = ep.define_event_detector_spec(
    "ivt",
    "ivt",
    velocity_threshold=30,
    minimum_duration_ms=60,
    maximum_gap_ms=75,
    sampling_rate=60,
    coordinate_unit="degrees",
)

multiverse = ep.create_detector_multiverse(
    base_spec=base,
    parameter_grid={
        "velocity_threshold": [20, 25, 30, 35, 40],
        "minimum_duration_ms": [60, 80, 100, 120],
    },
)
```

These numbers are **example values**, not universal defaults.

## Adaptive velocity reference

`algorithm="adaptive_velocity"` exposes a transparent robust-MAD reference detector. It requires the analyst to declare both `noise_factor` and `minimum_velocity_threshold`. It is deliberately named as a reference implementation and must not be reported as REMoDNaV, Nyström–Holmqvist, Engbert–Kliegl, or another named algorithm.

```python
adaptive = ep.define_event_detector_spec(
    "adaptive",
    "adaptive_velocity",
    minimum_duration_ms=60,
    maximum_gap_ms=75,
    sampling_rate=60,
    parameters={
        "noise_factor": 4,
        "minimum_velocity_threshold": 20,
    },
)
```

## REMoDNaV bridge

Use `algorithm="remodnav"` when REMoDNaV is the intended detector. The package calls the installed REMoDNaV implementation and records its version. If REMoDNaV is not installed, the branch fails explicitly; no internal approximation replaces it.

Pixel coordinates require an explicit `px2deg` conversion factor. Normalized coordinates must be converted before the bridge is called.

## External callbacks and vendor events

Use `algorithm="external"` for a validated external detector callback. The callback must return a canonicalizable event table or `EyeDataset`. Use `algorithm="vendor"` to carry vendor-supplied episodes into the same comparison framework.

## Ordering and reproducibility

Multiverse branches are sorted deterministically by detector id and specification fingerprint. Reordering the input specification list does not change the scientific output.

Next: [Comparing Event Catalogues](comparing-event-catalogues.md).
