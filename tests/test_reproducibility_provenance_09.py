from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "analysis_environment_snapshot",
    "compare_reproducibility_fingerprints",
    "export_prov_json",
    "export_ro_crate_metadata",
    "eye_prov_graph",
    "eye_reproducibility_fingerprint",
    "eye_session_manifest",
    "file_hash_manifest",
    "object_hash",
    "provenance_edge_table",
    "provenance_lineage_table",
    "read_reproducibility_fingerprint",
    "validate_eye_prov_graph",
    "verify_reproducibility_fingerprint",
    "write_prov_dot",
    "write_reproducibility_fingerprint",
]


def test_public_r078_exports_are_callable():
    assert len(TARGETS) == 16
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_frozen_fingerprint_detects_changed_result_and_verifies():
    first = ep.eye_reproducibility_fingerprint(
        data=[1, 2, 3, 4, 5],
        result=10,
    )
    second = ep.eye_reproducibility_fingerprint(
        data=[1, 2, 3, 4, 5],
        result=11,
    )

    assert first["eyeprocess_class"] == "eye_reproducibility_fingerprint"
    assert ep.verify_reproducibility_fingerprint(first) is True

    comparison = ep.compare_reproducibility_fingerprints(
        first,
        second,
    )
    assert comparison["identical"] is False
    assert comparison["eyeprocess_class"] == "eye_reproducibility_comparison"
    assert not bool(comparison["detail"].set_index("field").loc["result_hash", "identical"])


def test_object_hash_is_deterministic_for_supported_python_structures():
    value = {
        "b": [1, 2, 3],
        "a": pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": ["a", "b"],
            }
        ),
    }
    first = ep.object_hash(value)
    second = ep.object_hash(
        {
            "a": value["a"].copy(),
            "b": [1, 2, 3],
        }
    )
    assert first == second
    assert len(first) == 64


def test_file_hash_manifest_md5_and_sha256(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text(
        "eyeprocess reproducibility\n",
        encoding="utf-8",
    )

    md5 = ep.file_hash_manifest(
        [source],
        algorithm="md5",
    )
    sha = ep.file_hash_manifest(
        [source],
        algorithm="sha256",
    )

    assert len(md5) == 1
    assert md5.loc[0, "algorithm"] == "md5"
    assert len(md5.loc[0, "hash"]) == 32
    assert md5.loc[0, "size_bytes"] == source.stat().st_size

    assert sha.loc[0, "algorithm"] == "sha256"
    assert len(sha.loc[0, "hash"]) == 64

    empty = ep.file_hash_manifest([], algorithm="md5")
    assert list(empty.columns) == [
        "path",
        "algorithm",
        "hash",
        "size_bytes",
        "modified",
    ]


def test_environment_and_session_manifest_have_source_contract(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n", encoding="utf-8")

    environment = ep.analysis_environment_snapshot(
        packages=["numpy", "pandas"],
    )
    assert "python_version" in environment
    assert "platform" in environment
    assert "packages" in environment
    assert environment["packages"]["package"].tolist() == [
        "numpy",
        "pandas",
    ]

    manifest = ep.eye_session_manifest(
        data={"x": [1, 2]},
        files=[source],
        adapter="synthetic",
        decisions={"drop": False},
        pipeline={"step": "x"},
        seeds={"main": 1},
        notes="test",
    )
    assert manifest["eyeprocess_version"] == "0.11.1"
    assert isinstance(manifest["data_hash"], str)
    assert len(manifest["files"]) == 1
    assert manifest["adapter"] == "synthetic"
    assert isinstance(manifest["decisions_hash"], str)
    assert isinstance(manifest["pipeline_hash"], str)


def test_json_fingerprint_roundtrip_and_r_formats_are_explicitly_gated(
    tmp_path,
):
    fingerprint = ep.eye_reproducibility_fingerprint(
        data={"x": [1, 2]},
        analysis_spec={"method": "confirmatory"},
        result={"effect": 0.25},
        seeds={"fit": 7},
        label="analysis-A",
    )

    path = tmp_path / "fingerprint.json"
    returned = ep.write_reproducibility_fingerprint(
        fingerprint,
        path,
        format="json",
    )
    assert Path(returned).resolve() == path.resolve()

    imported = ep.read_reproducibility_fingerprint(path)
    assert imported["eyeprocess_class"] == "eye_reproducibility_fingerprint"
    assert ep.verify_reproducibility_fingerprint(imported) is True
    assert imported["fingerprint_hash"] == fingerprint["fingerprint_hash"]

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="R-specific",
    ):
        ep.write_reproducibility_fingerprint(
            fingerprint,
            tmp_path / "fingerprint.rds",
            format="rds",
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="R-specific",
    ):
        ep.read_reproducibility_fingerprint(
            tmp_path / "fingerprint.rds",
            format="rds",
        )


def test_tampered_json_fingerprint_warns(tmp_path):
    fingerprint = ep.eye_reproducibility_fingerprint(
        data=[1, 2, 3],
        result=10,
    )
    path = tmp_path / "fingerprint.json"
    ep.write_reproducibility_fingerprint(
        fingerprint,
        path,
        format="json",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["result_hash"] = "tampered"
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.warns(
        RuntimeWarning,
        match="does not match",
    ):
        imported = ep.read_reproducibility_fingerprint(path)
    assert ep.verify_reproducibility_fingerprint(imported) is False


def test_provenance_tables_and_graph_follow_frozen_validation():
    nodes = ep.provenance_lineage_table(
        ["raw", "clean", "model"],
        type=["entity", "entity", "activity"],
        label=["Raw", "Clean", "Model"],
    )
    edges = ep.provenance_edge_table(
        ["clean", "model"],
        ["raw", "clean"],
        relation=["wasDerivedFrom", "used"],
    )
    graph = ep.eye_prov_graph(
        nodes,
        edges,
        metadata={"study": "demo"},
    )

    assert graph["eyeprocess_class"] == "eye_prov_graph"
    assert ep.validate_eye_prov_graph(graph) is True

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="unique",
    ):
        ep.provenance_lineage_table(["n", "n"])

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="endpoints",
    ):
        ep.provenance_edge_table("", "b")

    bad_edges = ep.provenance_edge_table(
        "unknown",
        "raw",
    )
    with pytest.raises(
        ep.EyeProcessValidationError,
        match="unknown node",
    ):
        ep.eye_prov_graph(nodes, bad_edges)


def test_prov_json_dot_and_ro_crate_exports(tmp_path):
    nodes = ep.provenance_lineage_table(
        ["raw", "model"],
        type=["entity", "activity"],
        label=['Raw "file"', "Model"],
    )
    edges = ep.provenance_edge_table(
        "model",
        "raw",
        "used",
    )
    graph = ep.eye_prov_graph(nodes, edges)

    prov_path = tmp_path / "prov.json"
    returned = ep.export_prov_json(
        graph,
        prov_path,
    )
    assert Path(returned) == prov_path
    payload = json.loads(prov_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "eyeprocess-prov-0.9"
    assert len(payload["nodes"]) == 2
    assert len(payload["edges"]) == 1

    dot = ep.write_prov_dot(graph)
    assert dot.startswith("digraph eyeprocess_provenance {")
    assert '\\"file\\"' in dot
    assert '"model" -> "raw"' in dot

    data_file = tmp_path / "data.csv"
    data_file.write_text("x\n1\n", encoding="utf-8")
    crate_path = tmp_path / "ro-crate-metadata.json"

    returned_crate = ep.export_ro_crate_metadata(
        path=crate_path,
        name="Demo",
        description="Demo crate",
        files=[data_file],
        creator="Researcher",
        license="MIT",
        doi="10.0000/example",
    )
    assert Path(returned_crate) == crate_path

    crate = json.loads(crate_path.read_text(encoding="utf-8"))
    assert crate["@context"] == "https://w3id.org/ro/crate/1.3/context"
    dataset = next(item for item in crate["@graph"] if item["@id"] == "./")
    assert dataset["name"] == "Demo"
    assert dataset["license"] == "MIT"
    assert dataset["identifier"] == "10.0000/example"
    assert dataset["hasPart"] == [{"@id": "data.csv"}]


def test_ro_crate_rejects_duplicate_basenames(tmp_path):
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "same.txt"
    second = second_dir / "same.txt"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="basenames must be unique",
    ):
        ep.export_ro_crate_metadata(
            tmp_path / "crate.json",
            files=[first, second],
        )


def test_fingerprint_input_validation_and_graph_validation():
    with pytest.raises(
        ep.EyeProcessValidationError,
        match="label",
    ):
        ep.eye_reproducibility_fingerprint(
            label="",
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="fingerprint",
    ):
        ep.verify_reproducibility_fingerprint({"fingerprint_hash": "x"})

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="provenance node",
    ):
        ep.provenance_lineage_table([])

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="duplicate",
    ):
        ep.eye_prov_graph(
            pd.DataFrame(
                {
                    "id": ["a", "a"],
                    "type": ["entity", "entity"],
                    "label": ["A", "A2"],
                }
            )
        )
