from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.adapters as ad
import eyeprocesspy.coordinates as co
import eyeprocesspy.dataset as ds
import eyeprocesspy.schema as sc


def _dataset(recording_id="R1", coordinate_spaces=None, gaze_samples=None, episodes=None, aoi_geometry=None, raw=None, metadata=None):
    recordings = pd.DataFrame({"recording_id": [recording_id], "participant_id": ["P1"]})
    return ep.new_eye_dataset(
        recordings=recordings,
        coordinate_spaces=coordinate_spaces,
        gaze_samples=gaze_samples,
        episodes=episodes,
        aoi_geometry=aoi_geometry,
        raw=[] if raw is None else raw,
        vendor_metadata={} if metadata is None else metadata,
        validate=False,
    )


def _coordinate_dataset():
    spaces = pd.concat(
        [
            ep.new_coordinate_space("norm", "display_normalized_top_left"),
            ep.new_coordinate_space("surf", "surface_normalized_bottom_left"),
            ep.new_coordinate_space("pix", "display_pixels_top_left", width=100, height=200),
        ],
        ignore_index=True,
    )
    gaze = pd.DataFrame(
        {
            "recording_id": ["R1"],
            "sample_id": ["S1"],
            "gaze_x": [1.2],
            "gaze_y": [-0.2],
            "coordinate_space_id": ["norm"],
        }
    )
    episodes = pd.DataFrame(
        {
            "episode_id": ["E1"],
            "recording_id": ["R1"],
            "centroid_x": [0.5], "centroid_y": [0.5],
            "start_x": [0.1], "start_y": [0.2],
            "end_x": [0.8], "end_y": [0.9],
            "coordinate_space_id": ["norm"],
        }
    )
    geometry = pd.DataFrame(
        {
            "aoi_id": ["A1"], "x": [0.2], "y": [0.3],
            "width": [0.4], "height": [0.5], "coordinate_space_id": ["norm"]
        }
    )
    return _dataset(coordinate_spaces=spaces, gaze_samples=gaze, episodes=episodes, aoi_geometry=geometry)


def test_schema_residual_guards_and_defaults():
    custom = ep.eye_schema("custom")
    assert custom["version"] == "custom"
    with pytest.raises(ep.EyeProcessSchemaError, match="non-empty"):
        ep.schema_table("")
    with pytest.raises(ep.EyeProcessSchemaError, match="Unknown schema table"):
        ep.schema_table("missing")
    with pytest.raises(TypeError, match="pandas DataFrame"):
        ep.standardize_eye_table([], "recordings")
    standardized = ep.standardize_eye_table(pd.DataFrame({"recording_id": ["R"], "extra": [1]}), "recordings", keep_extra=False)
    assert "extra" not in standardized
    with pytest.raises(TypeError, match="pandas DataFrame"):
        ep.validate_eye_table([], "recordings")
    issues = ep.validate_eye_table(pd.DataFrame({"recording_id": ["R"], "extra": [1]}), "recordings", strict=True)
    assert {"missing_schema_field", "extra_field"}.issubset(set(issues.code))
    with pytest.raises(ValueError, match="Invalid `space_type`"):
        ep.new_coordinate_space("x", "invalid")
    custom_space = ep.new_coordinate_space("x", "custom", origin="o", x_unit="xu", y_unit="yu", width=10, height=20)
    assert custom_space.loc[0, "origin"] == "o" and custom_space.loc[0, "width"] == 10


def test_dataset_repr_table_and_provenance_residuals(tmp_path: Path):
    x = _dataset()
    assert "eye_dataset" in repr(x)
    copied = x.copy()
    assert copied is not x
    with pytest.raises(TypeError, match="table.*string"):
        ep.get_eye_table(x, 1)
    with pytest.raises(ep.EyeProcessSchemaError, match="Unknown component"):
        ep.get_eye_table(x, "missing")
    with pytest.raises(ep.EyeProcessSchemaError, match="Unknown canonical"):
        ep.set_eye_table(x, "missing", pd.DataFrame())

    rec2 = pd.DataFrame({"recording_id": ["R2"], "participant_id": ["P2"]})
    appended = ep.append_eye_table(x, "recordings", rec2, validate=False)
    assert len(appended["recordings"]) == 2

    source = tmp_path / "source.txt"
    source.write_text("payload\n", encoding="utf-8")
    p1 = ep.add_provenance(x, "a", source_files=[source, tmp_path / "absent"], warnings=["w1", "w2"])
    assert "|" in str(p1["provenance"].source_files.iloc[-1])
    assert "NA" in str(p1["provenance"].file_hashes.iloc[-1])
    assert "w1 | w2" == p1["provenance"].warnings.iloc[-1]
    p2 = ep.add_provenance(x, "b", source_files=str(source), file_hashes="manual", warnings="warning", software_version="v")
    assert p2["provenance"].file_hashes.iloc[-1] == "manual"
    p3 = ep.add_provenance(x, "c", source_files=pd.NA, file_hashes=None)
    assert p3["provenance"].file_hashes.iloc[-1] == ""

    manifest = ep.provenance_manifest(p1)
    assert manifest["actions"].shape[0] >= 1
    compact = ep.compact_eye_dataset(_dataset(raw=[{"x": 1}]), drop_raw=True, drop_empty=True)
    assert compact.raw == [] and compact.empty_components


def test_coordinate_registration_lookup_and_low_level_conversion_guards():
    x = _coordinate_dataset()
    extra = ep.new_coordinate_space("extra")
    with pytest.raises(TypeError, match="pandas DataFrame"):
        ep.register_coordinate_space(x, {})
    added = ep.register_coordinate_space(x, extra)
    assert "extra" in set(added["coordinate_spaces"].coordinate_space_id)
    with pytest.raises(ep.EyeProcessCoordinateError, match="already exists"):
        ep.register_coordinate_space(added, extra)
    replaced = ep.register_coordinate_space(added, extra, overwrite=True)
    assert (replaced["coordinate_spaces"].coordinate_space_id == "extra").sum() == 1

    with pytest.raises(ep.EyeProcessCoordinateError, match="Unknown coordinate-space"):
        ep.coordinate_space(x, "missing")
    lookup = ep.coordinate_space(x, ["norm", "pix"])
    assert lookup.coordinate_space_id.tolist() == ["norm", "pix"]

    with pytest.raises(ep.EyeProcessCoordinateError, match="supported 2D"):
        co._convert_xy([0], [0], "custom", "display_normalized_top_left")
    with pytest.raises(ep.EyeProcessCoordinateError, match="Source width"):
        co._convert_xy([1], [1], "display_pixels_top_left", "display_normalized_top_left")
    with pytest.raises(ep.EyeProcessCoordinateError, match="Destination width"):
        co._convert_xy([0.5], [0.5], "display_normalized_top_left", "display_pixels_top_left")

    pix_to_norm = co._convert_xy([50], [100], "display_pixels_top_left", "display_normalized_top_left", 100, 200)
    np.testing.assert_allclose(pix_to_norm[["x", "y"]], [[0.5, 0.5]])
    surf_to_norm = co._convert_xy([0.25], [0.1], "surface_normalized_bottom_left", "display_normalized_top_left")
    np.testing.assert_allclose(surf_to_norm[["x", "y"]], [[0.25, 0.9]])
    norm_to_surf = co._convert_xy([0.25], [0.9], "display_normalized_top_left", "surface_normalized_bottom_left")
    np.testing.assert_allclose(norm_to_surf[["x", "y"]], [[0.25, 0.1]])
    clipped_norm = co._convert_xy([-1, 2], [2, -1], "display_normalized_top_left", "display_normalized_top_left", clip=True)
    assert clipped_norm.x.tolist() == [0.0, 1.0]
    clipped_pix = co._convert_xy([-1, 2], [2, -1], "display_normalized_top_left", "display_pixels_top_left", to_width=100, to_height=200, clip=True)
    assert clipped_pix.x.tolist() == [0.0, 100.0]


def test_convert_coordinates_all_component_and_overwrite_paths():
    x = _coordinate_dataset()
    copied = ep.convert_coordinates(x, "norm", "pix", components="gaze_samples", overwrite=False)
    assert len(copied["gaze_samples"]) == 2
    assert "S1_pix" in set(copied["gaze_samples"].sample_id.astype(str))

    overwritten = ep.convert_coordinates(x, "norm", "pix", components="gaze_samples", overwrite=True, clip=True)
    row = overwritten["gaze_samples"].iloc[0]
    assert row.coordinate_space_id == "pix" and 0 <= row.gaze_x <= 100 and 0 <= row.gaze_y <= 200

    episodes = ep.convert_coordinates(x, "norm", "pix", components="episodes")
    assert episodes["episodes"].coordinate_space_id.iloc[0] == "pix"
    geometry = ep.convert_coordinates(x, "norm", "pix", components="aoi_geometry")
    assert geometry["aoi_geometry"].width.iloc[0] == pytest.approx(40)
    surface_geometry = ep.convert_coordinates(x, "norm", "surf", components="aoi_geometry")
    assert surface_geometry["aoi_geometry"].coordinate_space_id.iloc[0] == "surf"

    no_rows = ep.convert_coordinates(x, "surf", "pix", components=["gaze_samples"])
    assert len(no_rows["gaze_samples"]) == 1
    empty_component = ep.convert_coordinates(x, "norm", "pix", components=["events"])
    assert empty_component["events"].empty

    unregistered = _dataset(
        gaze_samples=pd.DataFrame({"recording_id": ["R1"], "sample_id": ["S"], "coordinate_space_id": ["ghost"]})
    )
    audit = ep.audit_coordinate_spaces(unregistered)
    assert audit.loc[0, "registered"] == False and audit.loc[0, "status"] == "error"


def test_adapter_registration_detection_and_read_guards(tmp_path: Path):
    original = dict(ad._ADAPTERS)
    try:
        ad._ADAPTERS.clear()
        assert ep.supported_eye_formats().empty
        for args in [
            ("", lambda *a, **k: 1, lambda *a, **k: None, None),
            ("x", 1, lambda *a, **k: None, None),
            ("x", lambda *a, **k: 1, 1, None),
            ("x", lambda *a, **k: 1, lambda *a, **k: None, 1),
        ]:
            with pytest.raises(TypeError):
                ep.register_eye_adapter(args[0], args[1], args[2], validate=args[3])
        ep.register_eye_adapter("a", lambda *a, **k: True, lambda *a, **k: "A", priority=2)
        with pytest.raises(ValueError, match="already registered"):
            ep.register_eye_adapter("a", lambda *a, **k: True, lambda *a, **k: "A")
        with pytest.raises(TypeError):
            ep.unregister_eye_adapter(1)

        f = tmp_path / "x.csv"
        f.write_text("x,y\n1,2\n", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            ep.detect_eye_format(tmp_path / "missing.csv")

        ep.register_eye_adapter("nan", lambda *a, **k: np.nan, lambda *a, **k: "N", priority=1)
        ep.register_eye_adapter("err", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")), lambda *a, **k: "E")
        ep.register_eye_adapter("high", lambda *a, **k: 5, lambda *a, **k: "H", priority=0)
        detected = ep.detect_eye_format(f)
        assert detected.loc[detected.format == "a", "confidence"].iloc[0] == 1
        assert detected.loc[detected.format == "nan", "confidence"].iloc[0] == 0
        assert detected.loc[detected.format == "err", "confidence"].iloc[0] == 0
        assert detected.loc[detected.format == "high", "confidence"].iloc[0] == 1

        with pytest.raises(ValueError, match="Unknown adapter"):
            ep.read_eye_export(f, vendor="missing")
        with pytest.raises(ValueError, match="confidence threshold"):
            ep.read_eye_export(f, vendor="auto", confidence_threshold=1.1)

        ep.register_eye_adapter("tie", lambda *a, **k: True, lambda *a, **k: "T", priority=1)
        with pytest.warns(RuntimeWarning, match="tied"):
            ep.read_eye_export(f, vendor="auto", confidence_threshold=0.5)
    finally:
        ad._ADAPTERS.clear()
        ad._ADAPTERS.update(original)


def test_adapter_folder_remap_combine_and_generic_detector(tmp_path: Path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="Directory"):
        ep.read_eye_folder(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No files found"):
        ep.read_eye_folder(empty)

    folder = tmp_path / "files"
    folder.mkdir()
    (folder / "a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (folder / "b.txt").write_text("x,y\n3,4\n", encoding="utf-8")
    listed = ep.read_eye_folder(folder, vendor="generic", combine=False, mapping={"timestamp": "x", "x": "x", "y": "y"})
    assert len(listed) == 2

    with monkeypatch.context() as ctx:
        ctx.setattr(ad, "read_eye_export", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad")))
        with pytest.raises(ValueError, match="could be imported"):
            ep.read_eye_folder(folder)

    x1 = _dataset(recording_id="R", raw={"a": 1}, metadata={"v1": 1})
    x2 = _dataset(recording_id="R", raw=[("b", 2)], metadata={"v2": 2})
    with pytest.raises(TypeError, match="eye_dataset"):
        ep.remap_recording_ids({}, {})
    remapped = ep.remap_recording_ids(x1, [("R", "RX")])
    assert remapped["recordings"].recording_id.iloc[0] == "RX"

    with pytest.raises(TypeError, match="All inputs"):
        ep.combine_eye_datasets([])
    combined = ep.combine_eye_datasets([x1, x2], resolve_ids=True)
    assert len(combined["recordings"]) == 2
    assert any(str(v).endswith("_set2") for v in combined["recordings"].recording_id)
    assert len(combined.raw) >= 2 and combined.vendor_metadata["v2"] == 2
    no_resolve = ep.combine_eye_datasets(x1, x2, resolve_ids=False)
    assert len(no_resolve["recordings"]) == 1

    directory_score = ad._detect_generic_delimited(folder)
    assert directory_score == 0.0
    unsupported = tmp_path / "x.bin"
    unsupported.write_text("x,y\n1,2\n", encoding="utf-8")
    assert ad._detect_generic_delimited(unsupported) == 0.0
    onecol = tmp_path / "one.csv"
    onecol.write_text("x\n1\n", encoding="utf-8")
    assert ad._detect_generic_delimited(onecol) == 0.0
    twocol = tmp_path / "two.csv"
    twocol.write_text("x,y\n1,2\n", encoding="utf-8")
    assert ad._detect_generic_delimited(twocol) == 0.1
    with monkeypatch.context() as ctx:
        ctx.setattr(ad, "_read_delimited", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad")))
        assert ad._detect_generic_delimited(twocol) == 0.0
