from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.partitioned_storage_10 as mod


def _tables():
    return {
        "responses": pd.DataFrame(
            {
                "participant_id": ["P1", "P1", "P2"],
                "recording_id": ["R1", "R1", "R2"],
                "score": [1, 0, 1],
            }
        ),
        "gaze_samples": pd.DataFrame(
            {
                "participant_id": ["P1", "P2"],
                "recording_id": ["R1", "R2"],
                "sample_id": [1, 2],
                "x": [0.1, 0.2],
            }
        ),
    }


def _csv_store(tmp_path):
    return ep.write_partitioned_eye_storage(
        _tables(),
        tmp_path / "store",
        ep.partition_eye_storage(by=["participant_id"], format="csv", max_rows=2),
    )


def test_eye_dict_repr_properties_and_tagged_frame():
    mapping = mod._EyeDict(a=1)
    assert mapping.a == 1
    with pytest.raises(AttributeError):
        _ = mapping.missing

    frame = mod._tag_frame(pd.DataFrame({"x": [1]}), "custom_class")
    assert frame.eyeprocess_class == "custom_class"
    assert frame[["x"]].eyeprocess_class == "custom_class"

    spec = ep.partition_eye_storage(format="csv", max_rows=2)
    assert "eye_partition_spec" in repr(spec)
    handle = mod.EyePartitionedStorage(
        path="/tmp/x",
        metadata={"format": "csv"},
        partitions=pd.DataFrame(),
        transactions=pd.DataFrame(),
    )
    assert handle.path == "/tmp/x"
    assert handle.metadata["format"] == "csv"
    assert handle.partitions.empty
    assert handle.transactions.empty
    assert "eye_partitioned_storage" in repr(handle)


def test_coerce_normalize_and_partition_spec_guards():
    with pytest.raises(ep.EyeProcessValidationError, match="named list"):
        mod._coerce_tables(object())
    out = mod._coerce_tables({"a": pd.DataFrame({"x": [1]}), "b": [1]})
    assert list(out) == ["a"]

    assert mod._normalize_format(("csv", "parquet")) == "csv"
    with pytest.raises(ep.EyeProcessValidationError, match="format"):
        mod._normalize_format([])
    with pytest.raises(ep.EyeProcessValidationError, match="format"):
        mod._normalize_format("feather")

    with pytest.raises(ep.EyeProcessValidationError, match="max_rows"):
        ep.partition_eye_storage(format="csv", max_rows="bad")
    with pytest.raises(ep.EyeProcessValidationError, match="max_rows"):
        ep.partition_eye_storage(format="csv", max_rows=0)
    with pytest.raises(ep.EyeProcessValidationError, match="character vector"):
        ep.partition_eye_storage(by=3, format="csv")
    with pytest.raises(ep.EyeProcessValidationError, match="character vector"):
        ep.partition_eye_storage(by=["participant_id", None], format="csv")

    spec = ep.partition_eye_storage(by="participant_id", format=("csv", "rds"))
    assert spec.by == ("participant_id",)
    assert spec.format == "csv"


def test_upgrade_dataset_guards_copy_false_and_existing_migration_log():
    with pytest.raises(ep.EyeProcessValidationError, match="eye_dataset"):
        ep.upgrade_eye_dataset({})
    x = ep.new_eye_dataset(validate=False)
    with pytest.raises(ep.EyeProcessValidationError, match="only to schema"):
        ep.upgrade_eye_dataset(x, target_version="3.0.0")

    x.schema_version = None
    x.pop("provenance", None)
    x.eyeprocess_migration_log = pd.DataFrame(
        [{"from": "0.9", "to": "1.0", "operation": "old", "timestamp_utc": "old"}]
    )
    out = ep.upgrade_eye_dataset(x, copy=False)
    assert out is x
    assert out.schema_version == "2.0.0"
    assert len(out.eyeprocess_migration_log) == 2
    assert isinstance(out["provenance"], pd.DataFrame)


def test_partition_path_groups_chunks_and_transaction_helpers(tmp_path):
    path = mod._partition_path(
        tmp_path, "responses", ["participant_id"], [pd.NA], 2, "csv"
    )
    assert "participant_id=" in path.as_posix()
    assert path.name == "part-000002.csv"

    empty = pd.DataFrame(columns=["participant_id"])
    assert mod._partition_groups(empty, ["participant_id"]) == [[]]
    data = pd.DataFrame({"x": [1, 2, 3]})
    assert mod._partition_groups(data, []) == [[0, 1, 2]]
    grouped = mod._partition_groups(
        pd.DataFrame({"g": ["a", "a", pd.NA], "x": [1, 2, 3]}), ["g"]
    )
    assert sorted(map(len, grouped)) == [1, 2]

    assert mod._chunks([], 2) == [[]]
    assert mod._chunks([0, 1, 2], 2) == [[0, 1], [2]]

    tx = mod._transaction_id(
        tmp_path,
        "write",
        pd.DataFrame({"rows": [1, "bad", 2]}),
        "2026-01-01T00:00:00Z",
    )
    assert tx.startswith("tx-")


def test_piece_csv_empty_and_rds_boundaries(tmp_path):
    frame = pd.DataFrame({"x": [1, 2]})
    csv_path = tmp_path / "piece.csv"
    mod._write_piece(frame, csv_path, "csv", "zstd")
    assert mod._read_piece(csv_path, "csv").shape == (2, 1)

    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert mod._read_piece(empty, "csv").empty

    with pytest.raises(ep.EyeProcessBackendError, match="RDS"):
        mod._write_piece(frame, tmp_path / "piece.rds", "rds", "zstd")
    with pytest.raises(ep.EyeProcessBackendError, match="RDS"):
        mod._read_piece(tmp_path / "piece.rds", "rds")


def test_write_partitioned_guards_selection_and_existing_targets(tmp_path):
    with pytest.raises(ep.EyeProcessValidationError, match="spec"):
        ep.write_partitioned_eye_storage(_tables(), tmp_path / "a", spec={})

    target = tmp_path / "dir"
    target.mkdir()
    (target / "marker").write_text("x", encoding="utf-8")
    with pytest.raises(ep.EyeProcessValidationError, match="not empty"):
        ep.write_partitioned_eye_storage(
            _tables(), target, ep.partition_eye_storage(format="csv")
        )

    file_target = tmp_path / "file"
    file_target.write_text("x", encoding="utf-8")
    with pytest.raises(ep.EyeProcessValidationError, match="already exists"):
        ep.write_partitioned_eye_storage(
            _tables(), file_target, ep.partition_eye_storage(format="csv")
        )

    with pytest.raises(ep.EyeProcessValidationError, match="No data-frame tables"):
        ep.write_partitioned_eye_storage(
            _tables(),
            tmp_path / "none",
            ep.partition_eye_storage(format="csv"),
            tables=["missing"],
        )

    single = ep.write_partitioned_eye_storage(
        _tables(),
        tmp_path / "single",
        ep.partition_eye_storage(by=["participant_id"], format="csv"),
        tables="responses",
    )
    assert set(single.partitions["table"].astype(str)) == {"responses"}

    no_keys = ep.write_partitioned_eye_storage(
        {"responses": pd.DataFrame({"score": [1, 2, 3]})},
        tmp_path / "nokeys",
        ep.partition_eye_storage(by=["participant_id"], format="csv", max_rows=2),
    )
    assert len(no_keys.partitions) == 2


def test_open_partitioned_storage_guards_and_missing_manifest_columns(tmp_path):
    with pytest.raises(ep.EyeProcessValidationError, match="does not exist"):
        ep.open_partitioned_eye_storage(tmp_path / "missing")

    dput = tmp_path / "dput"
    dput.mkdir()
    (dput / "_eyeprocess_storage.dput").write_text("list()", encoding="utf-8")
    with pytest.raises(ep.EyeProcessBackendError, match="DPUT"):
        ep.open_partitioned_eye_storage(dput)

    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ep.EyeProcessValidationError, match="Not an eyeprocess"):
        ep.open_partitioned_eye_storage(plain)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "_eyeprocess_storage.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ep.EyeProcessValidationError, match="incomplete"):
        ep.open_partitioned_eye_storage(incomplete)

    root = tmp_path / "minimal"
    root.mkdir()
    (root / "_eyeprocess_storage.json").write_text(
        json.dumps({"format": "csv"}), encoding="utf-8"
    )
    pd.DataFrame({"table": ["responses"], "relative_path": ["x.csv"]}).to_csv(
        root / "_partitions.csv", index=False
    )
    pd.DataFrame({"transaction_id": ["t"]}).to_csv(
        root / "_transactions.csv", index=False
    )
    out = ep.open_partitioned_eye_storage(root)
    assert set(mod._PARTITION_COLUMNS) == set(out.partitions.columns)
    assert set(mod._TRANSACTION_COLUMNS) == set(out.transactions.columns)


def test_ensure_and_query_guard_branches(tmp_path):
    with pytest.raises(ep.EyeProcessValidationError, match="Expected partitioned"):
        mod._ensure_storage(object())

    storage = _csv_store(tmp_path)
    reopened = mod._ensure_storage(storage.path)
    assert isinstance(reopened, mod.EyePartitionedStorage)

    with pytest.raises(ep.EyeProcessValidationError, match="absent"):
        ep.query_eye_storage(storage, "missing")
    with pytest.raises(ep.EyeProcessValidationError, match="filters"):
        ep.query_eye_storage(storage, "responses", filters=[("x", 1)])
    with pytest.raises(ep.EyeProcessValidationError, match="Filter column"):
        ep.query_eye_storage(storage, "responses", filters={"missing": 1})
    with pytest.raises(ep.EyeProcessValidationError, match="Selected columns"):
        ep.query_eye_storage(storage, "responses", columns=["missing"])

    scalar = ep.query_eye_storage(
        storage, "responses", filters={"participant_id": "P1"}, columns="score"
    )
    assert list(scalar.columns) == ["score"]
    accepted_set = ep.query_eye_storage(
        storage, "responses", filters={"score": {0, 1}}
    )
    assert len(accepted_set) == 3

    fake_rds = mod.EyePartitionedStorage(
        path=storage.path,
        metadata={"format": "rds"},
        partitions=storage.partitions.copy(),
        transactions=storage.transactions.copy(),
    )
    with pytest.raises(ep.EyeProcessBackendError, match="RDS"):
        ep.query_eye_storage(fake_rds, "responses")

    fake_unknown = mod.EyePartitionedStorage(
        path=storage.path,
        metadata={"format": "unknown"},
        partitions=storage.partitions.copy(),
        transactions=storage.transactions.copy(),
    )
    with pytest.raises(ep.EyeProcessValidationError, match="Unknown storage format"):
        ep.query_eye_storage(fake_unknown, "responses")


def test_validation_hash_options_missing_file_and_empty_findings(tmp_path):
    storage = _csv_store(tmp_path)
    no_hash = ep.validate_eye_storage_metadata(storage, verify_hashes=False)
    assert no_hash["findings"]["fingerprint_match"].isna().all()
    assert no_hash["valid"] is True

    first = Path(storage.path) / str(storage.partitions["relative_path"].iloc[0])
    first.unlink()
    bad = ep.validate_eye_storage_metadata(storage)
    assert bad["valid"] is False
    assert (~bad["findings"]["exists"]).any()

    empty = mod.EyePartitionedStorage(
        path=storage.path,
        metadata={"format": "csv"},
        partitions=pd.DataFrame(columns=mod._PARTITION_COLUMNS),
        transactions=pd.DataFrame(columns=mod._TRANSACTION_COLUMNS),
    )
    valid = ep.validate_eye_storage_metadata(empty)
    assert valid["valid"] is True
    corrupt = ep.detect_corrupt_partitions(empty)
    assert corrupt.empty
    assert corrupt.eyeprocess_class == "eye_corrupt_partitions"


def test_transaction_manifest_and_migration_guards(tmp_path):
    storage = _csv_store(tmp_path)
    manifest = ep.storage_transaction_manifest(storage.path)
    assert manifest["action"].tolist() == ["write"]

    with pytest.raises(ep.EyeProcessValidationError, match="only to schema"):
        ep.migrate_eye_storage_schema(
            storage, tmp_path / "target", target_version="3.0.0"
        )
    with pytest.raises(ep.EyeProcessBackendError, match="RDS"):
        ep.migrate_eye_storage_schema(
            storage, tmp_path / "target-rds", format="rds"
        )

    migrated = ep.migrate_eye_storage_schema(
        storage, tmp_path / "target", format=None
    )
    assert migrated.metadata["format"] == "csv"
    assert migrated.transactions["action"].tolist()[-1] == "migrate"


def test_benchmark_argument_guards_string_format_and_temp_directory(tmp_path):
    with pytest.raises(ep.EyeProcessValidationError, match="repetitions"):
        ep.benchmark_eye_storage(_tables(), repetitions="bad")
    with pytest.raises(ep.EyeProcessValidationError, match="repetitions"):
        ep.benchmark_eye_storage(_tables(), repetitions=0)
    with pytest.raises(ep.EyeProcessValidationError, match="Unknown storage formats"):
        ep.benchmark_eye_storage(_tables(), formats=["csv", "bad"], repetitions=1)
    with pytest.raises(ep.EyeProcessBackendError, match="RDS benchmarking"):
        ep.benchmark_eye_storage(_tables(), formats="rds", repetitions=1)

    result = ep.benchmark_eye_storage(
        _tables(), formats="csv", repetitions=1, directory=tmp_path
    )
    assert result.eyeprocess_class == "eye_storage_benchmark"
    assert result["format"].tolist() == ["csv"]
