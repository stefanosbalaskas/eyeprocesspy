"""End-to-end deterministic eyeprocesspy workflow using a canonical EyeDataset."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eyeprocesspy as ep

OUT = Path("workflow-output")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(20260903)

# 1. Build a small vendor-neutral canonical dataset.
n = 180
t = np.linspace(0.0, 3.0, n)
aoi = np.where(t < 1.0, "Headline", np.where(t < 2.2, "Evidence", "CTA"))
centers = {"Headline": (0.25, 0.28), "Evidence": (0.68, 0.34), "CTA": (0.52, 0.73)}
xy = np.array([centers[x] for x in aoi], dtype=float) + rng.normal(0, 0.025, (n, 2))
xy = np.clip(xy, 0.01, 0.99)

recordings = pd.DataFrame({"recording_id": ["demo-001"], "participant_id": ["P001"]})
streams = pd.DataFrame({"stream_id": ["gaze-main"], "recording_id": ["demo-001"], "stream_type": ["gaze"]})
gaze = pd.DataFrame(
    {
        "sample_id": [f"s{i:04d}" for i in range(n)],
        "recording_id": "demo-001",
        "stream_id": "gaze-main",
        "trial_id": "T1",
        "timestamp_seconds": t,
        "gaze_x": xy[:, 0],
        "gaze_y": xy[:, 1],
        "valid": True,
        "aoi_id": aoi,
    }
)

episode_rows = []
visit_sequence = [
    ("Headline", 0.10, 0.62),
    ("Evidence", 0.70, 1.28),
    ("Headline", 1.36, 1.72),
    ("Evidence", 1.80, 2.34),
    ("CTA", 2.42, 2.92),
]
for i, (name, start, end) in enumerate(visit_sequence, 1):
    cx, cy = centers[name]
    episode_rows.append(
        {
            "episode_id": f"v{i}",
            "episode_type": "aoi_visit",
            "recording_id": "demo-001",
            "trial_id": "T1",
            "start_time": start,
            "end_time": end,
            "duration_ms": (end - start) * 1000.0,
            "centroid_x": cx,
            "centroid_y": cy,
            "aoi_id": name,
            "derived_by": "eyeprocess",
        }
    )

episodes = pd.DataFrame(episode_rows)
intervals = pd.DataFrame(
    {
        "interval_id": ["trial-T1"],
        "recording_id": ["demo-001"],
        "trial_id": ["T1"],
        "interval_type": ["trial"],
        "start_time": [0.0],
        "end_time": [3.0],
    }
)

ds = ep.new_eye_dataset(
    recordings=recordings,
    streams=streams,
    gaze_samples=gaze,
    episodes=episodes,
    intervals=intervals,
    validate=False,
)

# 2. Validate before analysis.
issues = ep.validate_eye_dataset(ds)
if not issues.empty:
    raise RuntimeError(issues.to_string(index=False))
print("validation: OK")

# 3. Derive sequence/process summaries.
sequence = ep.scanpath_sequence(ds, trial_id="T1", source="visits")
transitions = ep.transition_matrix(ds, source="visits", normalize="row")
entropy = ep.gaze_entropy(ds, level="trial", source="samples")
print("\nscanpath\n", sequence.to_string(index=False))
print("\ntransition matrix\n", transitions)
print("\nentropy\n", entropy.to_string(index=False))

# 4. Render auditable figures. The numerical payload remains attached to the axes.
for name, axis in {
    "gaze-trace": ep.plot_eye_trace(ds, trial_id="T1"),
    "scanpath": ep.plot_scanpath(ds, trial_id="T1"),
    "transition-matrix": ep.plot_transition_matrix(ds, normalize="row", source="visits"),
}.items():
    axis.figure.set_size_inches(7.2, 4.5)
    axis.figure.tight_layout()
    axis.figure.savefig(OUT / f"{name}.svg", format="svg", bbox_inches="tight")
    plt.close(axis.figure)

# 5. Create a reproducibility/provenance payload.
manifest = ep.provenance_manifest(ds)
print("\nschema version:", manifest["schema_version"])
print("validation rows:", len(manifest["validation"]))
print("saved figures to:", OUT.resolve())
