from __future__ import annotations

import builtins

import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.foundation_09 as fo
import eyeprocesspy.plots_completion_08 as pc
import eyeprocesspy.validation_completion_10 as vc


def _close(ax):
    import matplotlib.pyplot as plt

    plt.close(ax.figure)


def test_validation_completion_multiverse_jobs_and_public_benchmark(monkeypatch, tmp_path):
    with pytest.raises(ep.EyeProcessValidationError, match="non-empty"):
        vc.preprocessing_multiverse(1, {}, lambda x, s: x, lambda x: x)
    with pytest.raises(ep.EyeProcessValidationError, match="non-empty"):
        vc.preprocessing_multiverse(1, [], lambda x, s: x, lambda x: x)

    jobs = vc._jobs({"": 1, "job-2": 2, "job-2 ": 3}, "job-")
    assert len(jobs) == 3 and len(set(jobs)) == 3

    x = ep.new_eye_dataset(validate=False)
    x["recordings"] = ep.standardize_eye_table(
        pd.DataFrame(
            {
                "recording_id": ["R1", "R2", "R3"],
                "participant_id": ["P1", "P2", "P3"],
                "device_model": ["D", "D", "D"],
            }
        ),
        "recordings",
    )
    x["events"] = ep.standardize_eye_table(
        pd.DataFrame({"event_id": ["E1", "E2", "E3"], "recording_id": ["R1", "R2", "R3"]}),
        "events",
    )
    monkeypatch.setattr(vc, "anonymize_eye_dataset", lambda data, **kwargs: data.copy())

    def fake_export(data, output, **kwargs):
        output.mkdir(parents=True, exist_ok=True)
        return str(output)

    monkeypatch.setattr(vc, "export_canonical", fake_export)
    monkeypatch.setattr(vc, "reporting_guideline_audit", lambda data: pd.DataFrame({"covered": [True]}))
    outdir = tmp_path / "bench"
    outdir.mkdir()
    (outdir / "old").write_text("x", encoding="utf-8")
    path = vc.create_public_benchmark(x, outdir, max_participants=2, overwrite=True)
    assert path == str(outdir.resolve())


def test_foundation_dataframe_conversion_empty_guards_and_aoi_missing_column():
    raw = pd.DataFrame({"timestamp": [0.0], "x": [0.2], "y": [0.3]})
    ds = fo.as_eye_dataset(raw, mapping={"timestamp": "timestamp", "x": "x", "y": "y"})
    assert ep.is_eye_dataset(ds)

    empty_intervals = ep.empty_eye_table("intervals")
    found = fo._find_interval_id([0.0], ["R1"], empty_intervals, "trial_id")
    assert found.tolist() == [None]

    x = ep.new_eye_dataset(validate=False)
    interval = ep.standardize_eye_table(
        pd.DataFrame(
            {
                "interval_id": ["i"],
                "recording_id": ["R1"],
                "interval_type": ["trial"],
                "start_time": [0.0],
                "end_time": [1.0],
                "trial_id": ["T1"],
            }
        ),
        "intervals",
    )
    x["intervals"] = interval
    assigned = fo.assign_trials(x)
    assert ep.is_eye_dataset(assigned)

    with pytest.raises(TypeError, match="DataFrame"):
        fo.store_quality(x, [])
    assert fo.audit_sampling_rate(x).empty
    assert fo.audit_missingness(x).empty

    x["gaze_samples"] = pd.DataFrame({"recording_id": ["R1"]})
    assert fo.compare_aoi_definitions(x).empty


def test_plots_completion_private_validation_and_transition_paths():
    import matplotlib.pyplot as plt

    _, ax0 = plt.subplots()
    assert pc._axis(ax0) is ax0
    _close(ax0)

    class BadFrame:
        def __iter__(self):
            raise RuntimeError("bad")

    with pytest.raises(ep.EyeProcessValidationError, match="coercible"):
        pc._as_frame(BadFrame())
    with pytest.raises(ep.EyeProcessValidationError, match="Missing required"):
        pc._require_columns(pd.DataFrame({"a": [1]}), ["a", "b"])
    assert pc._normalize_choice([]) == "from"
    assert pc._normalize_choice(["none"]) == "none"
    with pytest.raises(ep.EyeProcessValidationError, match="normalize"):
        pc._normalize_choice("bad")
    with pytest.raises(ep.EyeProcessValidationError, match="No complete"):
        pc._transition_table(pd.DataFrame({"from": [pd.NA], "to": [pd.NA]}))

    table = pd.DataFrame({"from": ["A", "A", "B"], "to": ["B", "B", "A"]})
    all_norm = pc._transition_table(table, normalize="all")
    assert all_norm.to_numpy().sum() == pytest.approx(1.0)
    none = pc._transition_table(table, normalize="none")
    assert none.to_numpy().sum() == 3

    with pytest.raises(ep.EyeProcessValidationError, match="positive integer"):
        pc.plot_aoi_transition_rank(table, top_n=object())
    ax = pc.plot_aoi_transition_rank(table, top_n=2, normalize="none")
    assert len(ax.eyeprocess_plot_data) == 2
    _close(ax)

    with pytest.raises(ep.EyeProcessValidationError, match="No ablation rows"):
        pc.plot_process_channel_ablation_delta(table=pd.DataFrame(columns=["channel", "value"]))


def test_plots_completion_matplotlib_backend_guard(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ep.EyeProcessBackendError, match="matplotlib"):
        pc._get_plt()
