"""Generate a deterministic eyeprocesspy core-plot gallery.

Run from a checkout with plotting dependencies installed:
    python examples/core_gallery.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eyeprocesspy as ep

OUT = Path("gallery-output")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(20260903)
n = 360
t = np.linspace(0, 6, n)
centers = np.array([[0.25, 0.30], [0.70, 0.32], [0.52, 0.72]])
states = np.repeat(np.tile([0, 1, 2, 1], 3), n // 12)
states = np.pad(states, (0, max(0, n - len(states))), mode="edge")[:n]
gxy = np.clip(centers[states] + rng.normal(scale=[0.035, 0.03], size=(n, 2)), 0.02, 0.98)
valid = np.ones(n, dtype=bool)
valid[[40, 41, 180, 181, 182, 300]] = False

recordings = pd.DataFrame({"recording_id": ["demo-001"], "participant_id": ["P001"]})
streams = pd.DataFrame({"stream_id": ["gaze-main"], "recording_id": ["demo-001"], "stream_type": ["gaze"]})
gaze = pd.DataFrame(
    {
        "sample_id": [f"s{i:04d}" for i in range(n)],
        "recording_id": "demo-001",
        "stream_id": "gaze-main",
        "trial_id": "T1",
        "timestamp_seconds": t,
        "gaze_x": gxy[:, 0],
        "gaze_y": gxy[:, 1],
        "valid": valid,
        "aoi_id": np.array(["Headline", "Evidence", "CTA"])[states],
    }
)

eye_rows = []
for eye, phase in [("left", 0.0), ("right", 0.16)]:
    pupil = (
        3.15
        + 0.20 * np.sin(2 * np.pi * (t / 3) + phase)
        + 0.10 * np.exp(-0.5 * ((t - 3.2) / 0.55) ** 2)
        + rng.normal(0, 0.018, n)
    )
    eye_rows.extend(
        {
            "recording_id": "demo-001",
            "sample_id": f"s{i:04d}",
            "eye": eye,
            "trial_id": "T1",
            "timestamp_seconds": t[i],
            "pupil_diameter": pupil[i],
        }
        for i in range(n)
    )
eye_samples = pd.DataFrame(eye_rows)

starts = np.array([0.15, 0.72, 1.30, 1.93, 2.52, 3.15, 3.82, 4.48, 5.15, 5.70])
sequence = ["Headline", "Evidence", "CTA", "Evidence", "Headline", "Evidence", "CTA", "Evidence", "Headline", "CTA"]
center_map = {"Headline": centers[0], "Evidence": centers[1], "CTA": centers[2]}
fixations, visits = [], []
for i, (start, aoi) in enumerate(zip(starts, sequence)):
    duration = 300 + 35 * (i % 4)
    cx, cy = center_map[aoi] + rng.normal(0, 0.012, 2)
    common = {
        "recording_id": "demo-001",
        "trial_id": "T1",
        "start_time": start,
        "end_time": start + duration / 1000,
        "duration_ms": duration,
        "centroid_x": cx,
        "centroid_y": cy,
        "aoi_id": aoi,
        "derived_by": "eyeprocess",
    }
    fixations.append({"episode_id": f"f{i + 1}", "episode_type": "fixation", **common})
    visits.append({"episode_id": f"v{i + 1}", "episode_type": "aoi_visit", **common})

episodes = pd.DataFrame(fixations + visits)
intervals = pd.DataFrame(
    {
        "interval_id": ["trial-T1"],
        "recording_id": ["demo-001"],
        "trial_id": ["T1"],
        "interval_type": ["trial"],
        "start_time": [0.0],
        "end_time": [6.0],
    }
)
features = pd.DataFrame(
    [
        {"feature_id": "d1", "recording_id": "demo-001", "trial_id": "T1", "aoi_id": "Headline", "feature_name": "dwell_time_ms", "value": 1180.0},
        {"feature_id": "d2", "recording_id": "demo-001", "trial_id": "T1", "aoi_id": "Evidence", "feature_name": "dwell_time_ms", "value": 1950.0},
        {"feature_id": "d3", "recording_id": "demo-001", "trial_id": "T1", "aoi_id": "CTA", "feature_name": "dwell_time_ms", "value": 870.0},
    ]
)

ds = ep.new_eye_dataset(
    recordings=recordings,
    streams=streams,
    gaze_samples=gaze,
    eye_samples=eye_samples,
    episodes=episodes,
    intervals=intervals,
    features=features,
    validate=False,
)
issues = ep.validate_eye_dataset(ds)
if not issues.empty:
    raise RuntimeError(issues.to_string(index=False))

plots = [
    ("eye-overview", lambda: ep.plot_eye_overview(ds)),
    ("gaze-trace", lambda: ep.plot_eye_trace(ds, trial_id="T1")),
    ("fixations", lambda: ep.plot_fixations(ds, trial_id="T1")),
    ("scanpath", lambda: ep.plot_scanpath(ds, trial_id="T1")),
    ("gaze-heatmap", lambda: ep.plot_gaze_heatmap(ds, trial_id="T1", bins=(32, 24))),
    ("pupil-timeseries", lambda: ep.plot_pupil_timeseries(ds, trial_id="T1")),
    ("aoi-dwell", lambda: ep.plot_aoi_dwell(ds)),
    ("transition-matrix", lambda: ep.plot_transition_matrix(ds, normalize="row", source="visits")),
]
for name, make_plot in plots:
    ax = make_plot()
    ax.figure.set_size_inches(7.2, 4.5)
    ax.figure.tight_layout()
    ax.figure.savefig(OUT / f"{name}.svg", format="svg", bbox_inches="tight")
    plt.close(ax.figure)
    print(f"saved {OUT / f'{name}.svg'}")
