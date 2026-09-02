from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.reproducibility_provenance_09 as rp


class _ScalarOnly:
    def __repr__(self):
        return "scalar-only"


class _BadFrame:
    def __iter__(self):
        raise RuntimeError("no iteration")


def test_private_list_frame_require_and_recycle_fallbacks():
    assert rp._as_list(None) == []
    assert rp._as_list(pd.Series([1, 2])) == [1, 2]
    scalar = _ScalarOnly()
    assert rp._as_list(scalar) == [scalar]

    frame = rp._as_frame({"x": [1, 2]}, name="frame")
    assert frame["x"].tolist() == [1, 2]
    with pytest.raises(ep.EyeProcessValidationError, match="coercible"):
        rp._as_frame(_BadFrame(), name="bad")
    with pytest.raises(ep.EyeProcessValidationError, match="missing required"):
        rp._require_columns(pd.DataFrame({"x": [1]}), ["x", "y"], name="frame")

    assert rp._recycle([], 3) == [None, None, None]
    assert rp._recycle(["a", "b"], 3) == ["a", "b", "a"]


def test_private_scalar_and_canonicalization_residual_types(tmp_path):
    assert rp._clean_scalar(np.int64(4)) == 4
    assert rp._clean_scalar(tmp_path) == str(tmp_path)
    assert rp._clean_scalar(date(2026, 9, 2)) == "2026-09-02"
    assert rp._clean_scalar(pd.NA) is None
    assert rp._clean_scalar(float("nan")) is None

    series = rp._canonicalize(pd.Series([1, pd.NA], name="s"))
    assert series["__type__"] == "series"
    assert series["data"][1] is None

    array = rp._canonicalize(np.array([[1, 2]], dtype=np.int64))
    assert array["__type__"] == "ndarray" and array["shape"] == [1, 2]

    aset = rp._canonicalize({3, 1, 2})
    assert aset == [1, 2, 3]

    assert rp._canonicalize(datetime(2026, 9, 2, 12, 0)) == "2026-09-02T12:00:00"
    assert rp._canonicalize(pd.NA) is None
    assert rp._canonicalize(float("inf")) == {"__float__": "inf"}
    assert rp._canonicalize(float("-inf")) == {"__float__": "-inf"}

    custom = rp._canonicalize(_ScalarOnly())
    assert custom["__type__"].endswith("._ScalarOnly")
    assert custom["repr"] == "scalar-only"


def test_file_hash_invalid_algorithm_guard():
    with pytest.raises(ep.EyeProcessValidationError, match="algorithm"):
        ep.file_hash_manifest([], algorithm="sha1")


def test_environment_locale_and_timezone_exception_fallbacks(monkeypatch):
    with monkeypatch.context() as mp:
        def fail_locale(*args, **kwargs):
            raise RuntimeError("locale unavailable")

        mp.setattr(rp.locale, "setlocale", fail_locale)
        snapshot = rp.analysis_environment_snapshot(packages=[])
        assert snapshot["locale"] is None

    class BadDateTime:
        @classmethod
        def now(cls, *args, **kwargs):
            raise RuntimeError("timezone unavailable")

    with monkeypatch.context() as mp:
        mp.setattr(rp, "datetime", BadDateTime)
        assert rp.time_zone_name() is None


def test_jsonify_remaining_container_and_nonfinite_paths():
    assert rp._jsonify(pd.Series([1, np.nan])) == [1, None]
    assert rp._jsonify([1, np.inf]) == [1, None]
    assert rp._jsonify(np.array([1.0, -np.inf])) == [1.0, None]
    assert rp._jsonify(np.inf) is None


def test_fingerprint_invalid_formats_and_restore_guards(tmp_path):
    fingerprint = ep.eye_reproducibility_fingerprint(data=[1, 2])

    with pytest.raises(ep.EyeProcessValidationError, match="format must"):
        ep.write_reproducibility_fingerprint(
            fingerprint,
            tmp_path / "fingerprint.yaml",
            format="yaml",
        )

    with pytest.raises(ep.EyeProcessValidationError, match="object"):
        rp._restore_fingerprint_payload([1, 2, 3])

    restored = rp._restore_fingerprint_payload(
        {
            "schema_version": "eyeprocess-reproducibility-0.9",
            "file_manifest": pd.DataFrame(),
            "environment": None,
        }
    )
    assert restored["eyeprocess_class"] == "eye_reproducibility_fingerprint"
    assert isinstance(restored["file_manifest"], pd.DataFrame)

    with pytest.raises(ep.EyeProcessValidationError, match="format must"):
        ep.read_reproducibility_fingerprint(
            tmp_path / "fingerprint.txt",
            format="txt",
        )


def test_provenance_empty_invalid_type_relation_and_validation_guards():
    with pytest.raises(ep.EyeProcessValidationError, match="At least one"):
        ep.provenance_lineage_table([], type=[], label=[], value=[])

    with pytest.raises(ep.EyeProcessValidationError, match="types"):
        ep.provenance_lineage_table("node", type="")

    empty_edges = ep.provenance_edge_table([], [], relation=[])
    assert empty_edges.empty
    assert list(empty_edges.columns) == ["from", "to", "relation"]

    with pytest.raises(ep.EyeProcessValidationError, match="relations"):
        ep.provenance_edge_table("a", "b", relation="")

    with pytest.raises(ep.EyeProcessValidationError, match="eye_prov_graph"):
        ep.validate_eye_prov_graph({})

    invalid = {
        "eyeprocess_class": "eye_prov_graph",
        "nodes": pd.DataFrame(
            {"id": [""], "type": ["entity"], "label": ["bad"]}
        ),
        "edges": pd.DataFrame(columns=["from", "to", "relation"]),
    }
    with pytest.raises(ep.EyeProcessValidationError, match="ids cannot"):
        ep.validate_eye_prov_graph(invalid)


def test_ro_crate_without_files_exercises_minimal_dataset_branch(tmp_path):
    output = tmp_path / "ro-crate-metadata.json"
    returned = ep.export_ro_crate_metadata(
        output,
        name="No-files crate",
        description="Minimal provenance crate",
        files=None,
    )
    assert Path(returned) == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    dataset = next(item for item in payload["@graph"] if item["@id"] == "./")
    assert dataset["name"] == "No-files crate"
    assert "hasPart" not in dataset
