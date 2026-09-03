from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.core_plots_10 as cp
import eyeprocesspy.evidence_graph as eg
import eyeprocesspy.gazepoint_real_10 as gr
import eyeprocesspy.partitioned_storage_10 as ps
import eyeprocesspy.plots_completion_08 as pc


def _close(ax):
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def _dataset(**tables):
    x = ep.new_eye_dataset(validate=False)
    for key, value in tables.items():
        x[key] = value.copy()
    return x


def test_core_plot_nonempty_release_paths(monkeypatch):
    episodes = pd.DataFrame(
        {
            "episode_type": ["fixation", "fixation", "saccade"],
            "centroid_x": [0.1, 0.3, 0.8],
            "centroid_y": [0.2, 0.4, 0.8],
            "duration_ms": [100.0, np.nan, 20.0],
            "derived_by": ["vendor", "vendor", "vendor"],
            "start_time": [0.2, 0.1, 0.3],
            "recording_id": ["R1", "R1", "R1"],
            "trial_id": ["T1", "T1", "T1"],
        }
    )
    ds = _dataset(episodes=episodes)
    ax = cp.plot_fixations(ds, source="vendor", reverse_y=True, scale=0.02)
    assert len(ax.collections) == 1 and ax.yaxis_inverted()
    assert len(ax.eyeprocess_plot_data) == 2
    _close(ax)

    # Exercise fixation fallback, labels, and the reverse-y branch in scanpath.
    ax = cp.plot_scanpath(ds, label=True, reverse_y=True)
    assert len(ax.texts) == 2 and ax.yaxis_inverted()
    _close(ax)

    features = pd.DataFrame(
        {
            "feature_name": ["dwell_time_ms", "dwell_time_ms", "other"],
            "aoi_id": ["A", "B", "A"],
            "value": [100.0, "200", 9],
        }
    )
    ds2 = _dataset(features=features)
    ax = cp.plot_aoi_dwell(ds2, feature="dwell_time_ms", aggregate=np.mean)
    assert list(ax.eyeprocess_plot_data.index) == ["A", "B"]
    _close(ax)

    monkeypatch.setattr(cp, "transition_matrix", lambda *a, **k: pd.DataFrame([[0.0, 1.0], [2.0, 0.0]], index=["A", "B"], columns=["A", "B"]))
    ax = cp.plot_transition_matrix(ds2, normalize="row", source="visits")
    assert ax.eyeprocess_plot_matrix.shape == (2, 2)
    _close(ax)

    ax = cp.plot_feature_distribution(ds2, "missing")
    assert ax.eyeprocess_plot_data.empty
    _close(ax)


def test_gazepoint_private_and_empty_summary_release_paths(monkeypatch, tmp_path):
    summary = gr.GazepointSummary(software="Gazepoint", software_version="1", aoi_summary=pd.DataFrame(), aoi_statistics=pd.DataFrame())
    assert "gazepoint_summary" in repr(summary)
    with pytest.raises(AttributeError):
        _ = summary.no_such_attribute

    assert gr._parse_csv_block(["x"], "missing").empty
    assert gr._parse_csv_block(["TITLE"], "TITLE").empty

    # Force the tolerant csv.reader fallback, including ragged rows.
    real_read_csv = gr.pd.read_csv
    monkeypatch.setattr(gr.pd, "read_csv", lambda *a, **k: (_ for _ in ()).throw(ValueError("force fallback")))
    out = gr._parse_csv_block(["TITLE", "a,b", " 1 , 2,extra", "3"], "TITLE")
    assert out.shape == (2, 2)
    monkeypatch.setattr(gr.pd, "read_csv", real_read_csv)

    frame = pd.DataFrame({"present": [1, 2]})
    assert gr._character_column(frame, "missing").isna().all()
    assert gr._numeric_column(frame, "missing").isna().all()
    ids = gr._gp_aoi_id(["M1", "M2"], [pd.NA, "  "])
    assert ids.isna().all()
    assert gr.re_sub_aoi_prefix("AOI target") == "target"
    assert gr.re_sub_aoi_prefix("target") == "target"
    with pytest.raises(ValueError, match="equal length"):
        gr._recording_frame(["P1"], ["R1", "R2"], "S1", source_path=str(tmp_path))

    assert gr._summary_features(summary).empty
    no_metrics = gr.GazepointSummary(
        software="Gazepoint",
        software_version="1",
        processed_on="today",
        aoi_statistics=pd.DataFrame({"Media ID": ["M"], "AOI ID": ["A"], "User ID": ["U"]}),
        aoi_summary=pd.DataFrame(),
    )
    assert gr._summary_features(no_metrics).empty


def test_plots_completion_ablation_source_shapes():
    f = pd.DataFrame({"channel": ["full", "gaze"], "metric": ["rmse", "rmse"], "value": [1.0, 1.5]})
    assert pc._extract_ablation_table(None, f).equals(f)
    assert pc._extract_ablation_table(f, None).equals(f)
    assert pc._extract_ablation_table({"results": f}, None).equals(f)
    assert pc._extract_ablation_table({"results": None, "metrics": f}, None).equals(f)
    assert pc._extract_ablation_table(SimpleNamespace(summary=f), None).equals(f)
    with pytest.raises(ep.EyeProcessValidationError, match="compatible ablation"):
        pc._extract_ablation_table({}, None)

    ax = pc.plot_process_channel_ablation_delta(table=f, metric="rmse")
    assert not ax.eyeprocess_plot_data.empty
    _close(ax)


def test_evidence_graph_explicit_edges_cycles_and_outcome_plots(monkeypatch):
    with pytest.raises(ep.EyeProcessValidationError, match="At least one"):
        eg.build_evidence_graph(None)

    nodes = pd.DataFrame({"node_id": ["r1", "r2"], "label": ["raw", "raw2"]})
    graph = eg.build_evidence_graph(nodes, transformations={"clean": 1}, decisions="decide")
    assert "stage" in graph.nodes and len(graph.edges) > 0

    with pytest.raises(ep.EyeProcessValidationError, match="from and to"):
        eg.build_evidence_graph(["r"], edges=[{"from": "r"}])
    explicit = eg.build_evidence_graph(["r"], decisions=["d"], edges=[{"from": "raw_data::r", "to": "decisions::d"}])
    assert set(explicit.edges.relation) == {"supports"}

    cyclic = eg.build_evidence_graph(
        ["r"], decisions=["d"],
        edges=[
            {"from": "raw_data::r", "to": "decisions::d", "relation": "supports"},
            {"from": "decisions::d", "to": "raw_data::r", "relation": "supports"},
        ],
    )
    assert bool(eg.audit_evidence_dependencies(cyclic).summary.has_cycle.iloc[0])

    # Stub recurrence internals so outcome/covariate regression branches are deterministic.
    monkeypatch.setattr(eg, "cross_recurrence", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(eg, "recurrence_features", lambda r: pd.DataFrame({"recurrence_rate": [0.25]}))
    model = eg.crossmodal_recurrence_model([1], [2], outcome=[1.0, 2.0, 3.0], covariates={"z": [0.0, 1.0, 2.0]})
    assert model.model is not None and "z" in model.model["predictors"]

    ax = eg.plot_crossmodal_recurrence_model(model, type="diagnostics")
    assert not ax.eyeprocess_plot_data.empty
    _close(ax)
    ax = eg.plot_crossmodal_recurrence_model(model, type="coefficients")
    assert not ax.eyeprocess_plot_data.empty
    _close(ax)

    no_model = eg.crossmodal_recurrence_model([1], [2])
    ax = eg.plot_crossmodal_recurrence_model(no_model, type="coefficients")
    assert ax.eyeprocess_plot_data.empty
    _close(ax)


def test_partitioned_storage_overwrite_and_atomic_restore(monkeypatch, tmp_path):
    tables = {"responses": pd.DataFrame({"participant_id": ["P1", "P2"], "score": [1, 0]})}
    spec = ep.partition_eye_storage(by=["participant_id"], format="csv", max_rows=1)

    # Default-spec line and normal overwrite/backup removal path.
    monkeypatch.setattr(ps, "partition_eye_storage", lambda: spec)
    default_store = ps.write_partitioned_eye_storage(tables, tmp_path / "default", spec=None, tables="responses")
    assert default_store.path
    target = tmp_path / "store"
    ps.write_partitioned_eye_storage(tables, target, spec)
    updated = ps.write_partitioned_eye_storage(tables, target, spec, overwrite=True)
    assert Path(updated.path).resolve() == target.resolve()

    # Force the staging->target rename to fail after target->backup. The
    # transactional exception branch must restore the original target.
    target2 = tmp_path / "restore"
    ps.write_partitioned_eye_storage(tables, target2, spec)
    original_rename = ps.Path.rename

    def flaky_rename(self, destination):
        src = str(self)
        if ".staging-" in src and Path(destination).resolve() == target2.resolve():
            raise OSError("forced atomic commit failure")
        return original_rename(self, destination)

    monkeypatch.setattr(ps.Path, "rename", flaky_rename)
    with pytest.raises(OSError, match="forced atomic"):
        ps.write_partitioned_eye_storage(tables, target2, spec, overwrite=True)
    assert target2.is_dir()
    assert (target2 / "_eyeprocess_storage.json").is_file()
