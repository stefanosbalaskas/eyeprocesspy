from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "benchmark_eye_storage",
    "detect_corrupt_partitions",
    "migrate_eye_storage_schema",
    "open_partitioned_eye_storage",
    "partition_eye_storage",
    "query_eye_storage",
    "storage_transaction_manifest",
    "upgrade_eye_dataset",
    "validate_eye_storage_metadata",
    "write_partitioned_eye_storage",
]


def _tables():
    return {
        "responses": pd.DataFrame(
            {
                "participant_id": ["P1", "P1", "P2", "P2"],
                "recording_id": ["R1", "R1", "R2", "R2"],
                "score": [1, 0, 1, 1],
            }
        ),
        "gaze_samples": pd.DataFrame(
            {
                "participant_id": ["P1"] * 3 + ["P2"] * 3,
                "recording_id": ["R1"] * 3 + ["R2"] * 3,
                "sample_id": list(range(1, 7)),
                "x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        ),
    }


def test_public_r028_storage_exports_are_callable():
    assert len(TARGETS) == 10
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_upgrade_eye_dataset_normalizes_indexes_and_retains_source(tmp_path):
    x = ep.new_eye_dataset(validate=False)
    x["responses"] = pd.DataFrame(
        {
            "response_id": ["a", "b"],
            "recording_id": ["r1", "r1"],
        },
        index=[4, 8],
    )
    x.schema_version = "1.0.0"

    upgraded = ep.upgrade_eye_dataset(x)
    assert upgraded is not x
    assert x["responses"].index.tolist() == [4, 8]
    assert upgraded["responses"].index.tolist() == [0, 1]
    assert upgraded.schema_version == "2.0.0"
    assert upgraded.eyeprocess_schema_version == "2.0.0"
    assert upgraded.eyeprocess_migration_log["from"].iloc[-1] == "1.0.0"
    assert (
        upgraded.eyeprocess_migration_log["operation"].iloc[-1]
        == "normalize canonical tables; preserve existing content"
    )


def test_partition_spec_and_rds_backend_boundary_are_non_destructive(tmp_path):
    spec = ep.partition_eye_storage(
        by=["participant_id", "participant_id", "recording_id"],
        format="csv",
        max_rows=2,
    )
    assert spec.eyeprocess_class == "eye_partition_spec"
    assert spec.by == ("participant_id", "recording_id")
    assert spec.max_rows == 2

    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ep.EyeProcessBackendError, match="RDS"):
        ep.write_partitioned_eye_storage(
            _tables(),
            target,
            ep.partition_eye_storage(format="rds"),
            overwrite=True,
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_csv_partitioned_storage_roundtrip_query_and_metadata(tmp_path):
    storage = ep.write_partitioned_eye_storage(
        _tables(),
        tmp_path / "csv-store",
        ep.partition_eye_storage(
            by=["participant_id", "recording_id"],
            format="csv",
            max_rows=2,
        ),
    )
    assert storage.eyeprocess_class == "eye_partitioned_storage"
    assert storage.metadata["schema_version"] == "2.0.0"
    assert storage.metadata["format"] == "csv"
    assert storage.partitions["fingerprint"].str.len().eq(32).all()
    assert (
        storage.partitions.loc[
            storage.partitions["table"].eq("gaze_samples"),
            "rows",
        ].max()
        <= 2
    )

    p1 = ep.query_eye_storage(
        storage,
        "responses",
        filters={"participant_id": "P1"},
        columns=["participant_id", "score"],
    )
    assert len(p1) == 2
    assert p1.columns.tolist() == ["participant_id", "score"]
    assert p1["participant_id"].eq("P1").all()

    reopened = ep.open_partitioned_eye_storage(storage.path)
    assert reopened.metadata["transaction_id"] == storage.metadata["transaction_id"]
    assert len(reopened.partitions) == len(storage.partitions)

    manifest = ep.storage_transaction_manifest(reopened)
    assert manifest["action"].tolist() == ["write"]
    assert manifest["status"].tolist() == ["committed"]


def test_parquet_storage_supports_lazy_dataset_and_collection(tmp_path):
    pytest.importorskip("pyarrow")
    storage = ep.write_partitioned_eye_storage(
        _tables(),
        tmp_path / "parquet-store",
        ep.partition_eye_storage(
            by=["participant_id"],
            format="parquet",
            max_rows=2,
        ),
    )
    lazy = ep.query_eye_storage(
        storage,
        "gaze_samples",
        collect=False,
    )
    assert lazy.__class__.__name__ == "FileSystemDataset"

    p2 = ep.query_eye_storage(
        storage,
        "gaze_samples",
        filters={"participant_id": ["P2"]},
    )
    assert len(p2) == 3
    assert p2["participant_id"].eq("P2").all()


def test_storage_validation_and_corruption_detection(tmp_path):
    storage = ep.write_partitioned_eye_storage(
        _tables(),
        tmp_path / "store",
        ep.partition_eye_storage(format="csv"),
    )
    validation = ep.validate_eye_storage_metadata(storage)
    assert validation.eyeprocess_class == "eye_storage_validation"
    assert validation["valid"] is True
    assert len(ep.detect_corrupt_partitions(storage)) == 0

    first = Path(storage.path) / storage.partitions["relative_path"].iloc[0]
    first.write_text(
        first.read_text(encoding="utf-8") + "\nmodified",
        encoding="utf-8",
    )
    bad = ep.detect_corrupt_partitions(storage)
    assert len(bad) == 1
    assert bad["relative_path"].iloc[0] == storage.partitions["relative_path"].iloc[0]


def test_storage_migration_retains_rows_and_adds_transaction(tmp_path):
    source = ep.write_partitioned_eye_storage(
        _tables(),
        tmp_path / "source",
        ep.partition_eye_storage(format="csv"),
    )
    target = ep.migrate_eye_storage_schema(
        source,
        tmp_path / "target",
        format="csv",
    )
    assert target.metadata["schema_version"] == "2.0.0"
    assert target.metadata["format"] == "csv"
    assert len(ep.query_eye_storage(target, "responses")) == 4
    assert len(ep.query_eye_storage(target, "gaze_samples")) == 6
    manifest = ep.storage_transaction_manifest(target)
    assert manifest["action"].tolist() == ["write", "migrate"]


def test_storage_benchmark_runs_supported_formats_and_excludes_rds(tmp_path):
    with pytest.warns(RuntimeWarning, match="RDS benchmarking"):
        result = ep.benchmark_eye_storage(
            _tables(),
            formats=["rds", "csv"],
            repetitions=1,
            directory=tmp_path,
        )
    assert result.eyeprocess_class == "eye_storage_benchmark"
    assert result["format"].tolist() == ["csv"]
    assert result["replication"].tolist() == [1]
    assert result["write_seconds"].ge(0).all()
    assert result["read_seconds"].ge(0).all()
    assert result["bytes"].gt(0).all()

    with pytest.raises(ep.EyeProcessBackendError, match="RDS benchmarking"):
        ep.benchmark_eye_storage(
            _tables(),
            formats=["rds"],
            repetitions=1,
            directory=tmp_path,
        )


def test_empty_table_still_has_a_physical_partition(tmp_path):
    storage = ep.write_partitioned_eye_storage(
        {"responses": pd.DataFrame(columns=["participant_id", "score"])},
        tmp_path / "empty-store",
        ep.partition_eye_storage(
            by=["participant_id"],
            format="csv",
        ),
    )
    assert len(storage.partitions) == 1
    assert storage.partitions["rows"].tolist() == [0]
    restored = ep.query_eye_storage(storage, "responses")
    assert restored.empty
    assert restored.columns.tolist() == ["participant_id", "score"]
