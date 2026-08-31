"""Stable-schema upgrade and atomic partitioned storage.

Source-port of the ten still-missing public non-S3 contracts from frozen
``R/028-api-storage-adapters.R`` in eyeprocess 0.11.1.

Native R ``.rds`` serialization is deliberately not impersonated. The ``rds``
format may be represented by a partition specification, but Python write/query
operations raise :class:`EyeProcessBackendError`. CSV is native and Parquet
uses PyArrow lazily through the existing ``arrow`` extra.
"""

from __future__ import annotations

import copy as _copy
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .dataset import is_eye_dataset
from .exceptions import EyeProcessBackendError, EyeProcessValidationError
from .interoperability_storage_10 import _arrow_table, _require_pyarrow, _resolve_compression
from .schema import canonical_table_names
from .validation_orchestration_10 import _hash_int, _now_utc, _safe_name

__all__ = [
    "EyePartitionSpec",
    "EyePartitionedStorage",
    "EyeStorageValidation",
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

_STORAGE_SCHEMA_VERSION = "2.0.0"
_API_VERSION = "0.5.0"
_PARTITION_COLUMNS = [
    "table",
    "relative_path",
    "rows",
    "bytes",
    "fingerprint",
    "partition_columns",
    "partition_values",
]
_TRANSACTION_COLUMNS = [
    "transaction_id",
    "action",
    "timestamp_utc",
    "tables",
    "partitions",
    "rows",
    "status",
]


class _EyeFrame(pd.DataFrame):
    _metadata = ["eyeprocess_class"]

    @property
    def _constructor(self):
        return _EyeFrame


def _tag_frame(frame: pd.DataFrame, class_name: str) -> _EyeFrame:
    out = _EyeFrame(frame.copy())
    out.eyeprocess_class = class_name
    return out


class _EyeDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(slots=True)
class EyePartitionSpec:
    """Python counterpart of frozen R ``eye_partition_spec``."""

    by: tuple[str, ...]
    format: str
    compression: str
    max_rows: int
    schema_version: str = _STORAGE_SCHEMA_VERSION

    eyeprocess_class = "eye_partition_spec"

    def __repr__(self) -> str:
        return f"<eye_partition_spec by={list(self.by)!r} format={self.format!r} max_rows={self.max_rows}>"


class EyePartitionedStorage(_EyeDict):
    """Disk-backed handle for atomic partitioned eyeprocess storage."""

    eyeprocess_class = "eye_partitioned_storage"

    @property
    def path(self) -> str:
        return self["path"]

    @property
    def metadata(self) -> dict[str, Any]:
        return self["metadata"]

    @property
    def partitions(self) -> pd.DataFrame:
        return self["partitions"]

    @property
    def transactions(self) -> pd.DataFrame:
        return self["transactions"]

    def __repr__(self) -> str:
        return (
            "<eye_partitioned_storage "
            f"path={self.path!r} format={self.metadata.get('format')!r} "
            f"partitions={len(self.partitions)}>"
        )


class EyeStorageValidation(_EyeDict):
    eyeprocess_class = "eye_storage_validation"


def _stop(message: str) -> None:
    raise EyeProcessValidationError(message)


def _metadata_path(root: Path) -> Path:
    return root / "_eyeprocess_storage.json"


def _r_dput_metadata_path(root: Path) -> Path:
    return root / "_eyeprocess_storage.dput"


def _transactions_path(root: Path) -> Path:
    return root / "_transactions.csv"


def _partitions_path(root: Path) -> Path:
    return root / "_partitions.csv"


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".csv",
        dir=path.parent,
    )
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        frame.to_csv(
            temporary_path,
            index=False,
            na_rep="",
            lineterminator="\n",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".json",
        dir=path.parent,
    )
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _source_class(x: Any) -> list[str]:
    names = [x.__class__.__name__]
    if is_eye_dataset(x):
        names.insert(0, "eye_dataset")
    return list(dict.fromkeys(names))


def _coerce_tables(x: Any) -> dict[str, pd.DataFrame]:
    if not isinstance(x, Mapping):
        _stop("`x` must be an eye dataset or named list of tables.")
    return {str(name): value for name, value in x.items() if isinstance(value, pd.DataFrame)}


def _normalize_format(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        value = value[0] if value else ""
    text = str(value)
    if text not in {"parquet", "csv", "rds"}:
        _stop("`format` must be 'parquet', 'csv', or 'rds'.")
    return text


def partition_eye_storage(
    by=("participant_id", "session_id", "recording_id"),
    format=("parquet", "csv", "rds"),
    compression="zstd",
    max_rows=1_000_000,
):
    """Create a partition specification using frozen R/028 semantics."""
    storage_format = _normalize_format(format)
    try:
        maximum = int(max_rows)
    except (TypeError, ValueError):
        _stop("`max_rows` must be a positive integer.")
    if maximum < 1:
        _stop("`max_rows` must be a positive integer.")
    if isinstance(by, str):
        by = [by]
    try:
        values = list(by)
    except TypeError:
        _stop("`by` must be a character vector without missing values.")
    if any(value is None or pd.isna(value) for value in values):
        _stop("`by` must be a character vector without missing values.")
    normalized = []
    for value in values:
        text = str(value)
        if text and text not in normalized:
            normalized.append(text)
    return EyePartitionSpec(
        by=tuple(normalized),
        format=storage_format,
        compression=str(compression),
        max_rows=maximum,
    )


def upgrade_eye_dataset(x, target_version="2.0.0", copy=True):
    """Upgrade a legacy ``EyeDataset`` to the stable 2.0.0 object contract."""
    if not is_eye_dataset(x):
        _stop("Expected an `eye_dataset`.")
    if str(target_version) != _STORAGE_SCHEMA_VERSION:
        _stop("This release can upgrade datasets only to schema version 2.0.0.")
    out = _copy.deepcopy(x) if copy else x
    current = getattr(out, "schema_version", None) or "1.0.0"
    for table in canonical_table_names():
        value = out.get(table)
        if isinstance(value, pd.DataFrame):
            out[table] = value.reset_index(drop=True)
    if "provenance" not in out or not isinstance(out["provenance"], pd.DataFrame):
        out["provenance"] = pd.DataFrame()

    log = pd.DataFrame(
        [
            {
                "from": str(current),
                "to": str(target_version),
                "operation": ("normalize canonical tables; preserve existing content"),
                "timestamp_utc": _now_utc(),
            }
        ]
    )
    existing = getattr(out, "eyeprocess_migration_log", None)
    if isinstance(existing, pd.DataFrame) and not existing.empty:
        log = pd.concat([existing, log], ignore_index=True, sort=False)
    out.schema_version = str(target_version)
    out.eyeprocess_schema_version = str(target_version)
    out.eyeprocess_migration_log = log
    return out


def _partition_path(
    root: Path,
    table: str,
    keys: Sequence[str],
    values: Sequence[Any],
    file_index: int,
    extension: str,
) -> Path:
    directory = root / str(table)
    for key, value in zip(keys, values, strict=True):
        raw = "<NA>" if pd.isna(value) else str(value)
        label = f"{_safe_name(raw, 'NA')}-{_hash_int(raw):010d}"
        directory = directory / f"{_safe_name(str(key), 'key')}={label}"
    return directory / f"part-{file_index:06d}.{extension}"


def _write_piece(
    data: pd.DataFrame,
    path: Path,
    storage_format: str,
    compression: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        if storage_format == "parquet":
            _, _, pq = _require_pyarrow()
            resolved = _resolve_compression(
                compression,
                allow_fallback=str(compression).strip().lower() == "zstd",
            )
            pq.write_table(
                _arrow_table(data),
                temporary,
                compression=resolved,
            )
        elif storage_format == "csv":
            data.to_csv(
                temporary,
                index=False,
                na_rep="",
                lineterminator="\n",
            )
        else:
            raise EyeProcessBackendError(
                "Native R RDS serialization is not available in eyeprocesspy. "
                "Use `csv` or `parquet`, or write RDS with the R package."
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_piece(path: Path, storage_format: str) -> pd.DataFrame:
    if storage_format == "parquet":
        _, _, pq = _require_pyarrow()
        return pq.read_table(path).to_pandas()
    if storage_format == "csv":
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    raise EyeProcessBackendError("Native R RDS collection is not available in eyeprocesspy.")


def _partition_groups(
    data: pd.DataFrame,
    keys: Sequence[str],
) -> list[list[int]]:
    if not len(data):
        return [[]]
    if not keys:
        return [list(range(len(data)))]

    tokens = []
    for row in data[list(keys)].itertuples(index=False, name=None):
        token = "\r".join("<NA>" if pd.isna(value) else str(value) for value in row)
        tokens.append(token)

    groups: dict[str, list[int]] = {}
    for index, token in enumerate(tokens):
        groups.setdefault(token, []).append(index)
    return list(groups.values())


def _chunks(values: Sequence[int], maximum: int) -> list[list[int]]:
    if not values:
        return [[]]
    return [list(values[start : start + maximum]) for start in range(0, len(values), maximum)]


def _transaction_id(
    path: Path,
    action: str,
    partitions: pd.DataFrame,
    timestamp: str,
) -> str:
    payload = "|".join(
        [
            str(path),
            action,
            timestamp,
            str(len(partitions)),
            str(
                int(
                    pd.to_numeric(
                        partitions.get("rows", pd.Series(dtype=float)),
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                )
            ),
        ]
    )
    return f"tx-{_hash_int(payload):010d}"


def write_partitioned_eye_storage(
    x,
    path,
    spec=None,
    overwrite=False,
    tables=None,
):
    """Write named tables as fingerprinted, atomically committed partitions."""
    if spec is None:
        spec = partition_eye_storage()
    if not isinstance(spec, EyePartitionSpec):
        _stop("`spec` must be created by `partition_eye_storage()`.")

    table_map = _coerce_tables(x)
    target = Path(path).expanduser().resolve()

    # Gate the R-only backend before any destructive target operation.
    if spec.format == "rds":
        raise EyeProcessBackendError(
            "Native R RDS serialization is not available in eyeprocesspy. "
            "Use `csv` or `parquet`, or write RDS with the R package."
        )
    if spec.format == "parquet":
        _require_pyarrow()

    if target.exists() and target.is_dir():
        nonempty = any(target.iterdir())
        if nonempty and not overwrite:
            _stop("Storage directory is not empty; use `overwrite = TRUE`.")
    elif target.exists() and not overwrite:
        _stop("Storage target already exists; use `overwrite = TRUE`.")

    if tables is None:
        selected = list(table_map)
    elif isinstance(tables, str):
        selected = [tables]
    else:
        selected = [str(value) for value in tables]
    selected = [name for name in dict.fromkeys(selected) if name and name in table_map]
    if not selected:
        _stop("No data-frame tables were selected for storage.")

    staging = target.with_name(f"{target.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
    backup = target.with_name(f"{target.name}.backup-{os.getpid()}-{uuid.uuid4().hex}")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    extension = "parquet" if spec.format == "parquet" else "csv"
    manifest_rows: list[dict[str, Any]] = []
    file_index = 0

    try:
        for table in selected:
            data = table_map[table]
            keys = [key for key in spec.by if key in data.columns]
            for group in _partition_groups(data, keys):
                for chunk in _chunks(group, spec.max_rows):
                    file_index += 1
                    values = [data.iloc[chunk[0]][key] for key in keys] if keys and chunk else ["NA"] * len(keys)
                    destination = _partition_path(
                        staging,
                        table,
                        keys,
                        values,
                        file_index,
                        extension,
                    )
                    piece = data.iloc[chunk].copy() if chunk else data.iloc[0:0].copy()
                    _write_piece(
                        piece,
                        destination,
                        spec.format,
                        spec.compression,
                    )
                    manifest_rows.append(
                        {
                            "table": table,
                            "relative_path": destination.relative_to(staging).as_posix(),
                            "rows": int(len(piece)),
                            "bytes": int(destination.stat().st_size),
                            "fingerprint": _md5(destination),
                            "partition_columns": ",".join(keys),
                            "partition_values": ",".join(str(value) for value in values),
                        }
                    )

        partitions = pd.DataFrame(
            manifest_rows,
            columns=_PARTITION_COLUMNS,
        )
        _atomic_write_csv(partitions, _partitions_path(staging))

        timestamp = _now_utc()
        transaction_id = _transaction_id(
            target,
            "write",
            partitions,
            timestamp,
        )
        transactions = pd.DataFrame(
            [
                {
                    "transaction_id": transaction_id,
                    "action": "write",
                    "timestamp_utc": timestamp,
                    "tables": ",".join(selected),
                    "partitions": int(len(partitions)),
                    "rows": int(
                        pd.to_numeric(
                            partitions["rows"],
                            errors="coerce",
                        )
                        .fillna(0)
                        .sum()
                    ),
                    "status": "committed",
                }
            ],
            columns=_TRANSACTION_COLUMNS,
        )
        _atomic_write_csv(
            transactions,
            _transactions_path(staging),
        )

        metadata = {
            "class": "eye_partitioned_storage",
            "schema_version": spec.schema_version,
            "api_version": _API_VERSION,
            "format": spec.format,
            "compression": spec.compression,
            "partition_by": list(spec.by),
            "created_utc": timestamp,
            "source_class": _source_class(x),
            "tables": selected,
            "transaction_id": transaction_id,
        }
        _atomic_write_json(
            metadata,
            _metadata_path(staging),
        )

        if target.exists():
            if backup.exists():
                if backup.is_dir():
                    shutil.rmtree(backup)
                else:
                    backup.unlink()
            target.rename(backup)

        try:
            staging.rename(target)
        except Exception:
            if backup.exists() and not target.exists():
                backup.rename(target)
            raise

        if backup.exists():
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()

    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise

    return open_partitioned_eye_storage(target)


def open_partitioned_eye_storage(path):
    """Open a partitioned storage handle without collecting its tables."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        _stop(f"Partitioned storage does not exist: {root}")

    metadata_path = _metadata_path(root)
    dput_path = _r_dput_metadata_path(root)
    if not metadata_path.is_file():
        if dput_path.is_file():
            raise EyeProcessBackendError(
                "This store uses R DPUT metadata. Open or migrate it with "
                "the R eyeprocess package before using it in eyeprocesspy."
            )
        _stop("Not an eyeprocess partitioned store.")
    if not _partitions_path(root).is_file() or not _transactions_path(root).is_file():
        _stop("Storage manifests are incomplete.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    partitions = pd.read_csv(
        _partitions_path(root),
        dtype_backend="numpy_nullable",
    )
    transactions = pd.read_csv(
        _transactions_path(root),
        dtype_backend="numpy_nullable",
    )
    for column in _PARTITION_COLUMNS:
        if column not in partitions:
            partitions[column] = pd.NA
    for column in _TRANSACTION_COLUMNS:
        if column not in transactions:
            transactions[column] = pd.NA

    return EyePartitionedStorage(
        path=root.as_posix(),
        metadata=metadata,
        partitions=partitions.loc[:, _PARTITION_COLUMNS],
        transactions=transactions.loc[:, _TRANSACTION_COLUMNS],
    )


def _ensure_storage(storage) -> EyePartitionedStorage:
    if isinstance(storage, (str, Path)):
        return open_partitioned_eye_storage(storage)
    if not isinstance(storage, EyePartitionedStorage):
        _stop("Expected partitioned storage.")
    return storage


def query_eye_storage(
    storage,
    table,
    filters=None,
    columns=None,
    collect=True,
):
    """Query one partitioned table, returning a lazy Dataset when requested."""
    handle = _ensure_storage(storage)
    table = str(table)
    partition = handle.partitions.loc[handle.partitions["table"].astype(str).eq(table)]
    if partition.empty:
        _stop(f"Table `{table}` is absent.")
    files = [Path(handle.path) / str(relative) for relative in partition["relative_path"]]
    storage_format = str(handle.metadata.get("format", ""))

    filters = {} if filters is None else filters
    if not isinstance(filters, Mapping):
        _stop("`filters` must be a mapping of equality filters.")
    projection = (
        [] if columns is None else ([columns] if isinstance(columns, str) else [str(value) for value in columns])
    )

    if storage_format == "rds":
        raise EyeProcessBackendError("Native R RDS collection is not available in eyeprocesspy.")

    if storage_format == "parquet":
        _, ds, _ = _require_pyarrow()
        dataset = ds.dataset(
            [str(path) for path in files],
            format="parquet",
        )
        if not collect and not filters and not projection:
            return dataset
        result = dataset.to_table().to_pandas()
    elif storage_format == "csv":
        frames = [_read_piece(path, "csv") for path in files]
        result = pd.concat(frames, ignore_index=True, sort=False)
    else:
        _stop(f"Unknown storage format `{storage_format}`.")

    for name, accepted in filters.items():
        name = str(name)
        if name not in result.columns:
            _stop(f"Filter column `{name}` is absent.")
        if isinstance(accepted, (str, bytes)) or not isinstance(
            accepted,
            (Sequence, set, np.ndarray, pd.Series),
        ):
            accepted = [accepted]
        result = result.loc[result[name].isin(list(accepted))]

    if projection:
        missing = [name for name in projection if name not in result.columns]
        if missing:
            _stop("Selected columns are absent: " + ", ".join(missing) + ".")
        result = result.loc[:, projection]

    return result.reset_index(drop=True)


def validate_eye_storage_metadata(storage, verify_hashes=True):
    """Validate partition existence, byte sizes, and MD5 fingerprints."""
    handle = _ensure_storage(storage)
    rows = []
    root = Path(handle.path)

    for row in handle.partitions.itertuples(index=False):
        relative = str(row.relative_path)
        file = root / relative
        exists = file.is_file()
        actual_bytes = file.stat().st_size if exists else math.nan
        expected_bytes = pd.to_numeric(
            pd.Series([row.bytes]),
            errors="coerce",
        ).iloc[0]
        bytes_match = bool(exists) and not pd.isna(expected_bytes) and int(actual_bytes) == int(expected_bytes)
        if verify_hashes and exists:
            fingerprint_match: Any = _md5(file) == str(row.fingerprint)
        elif verify_hashes:
            fingerprint_match = False
        else:
            fingerprint_match = pd.NA
        rows.append(
            {
                "relative_path": relative,
                "exists": bool(exists),
                "bytes_match": bool(bytes_match),
                "fingerprint_match": fingerprint_match,
            }
        )

    findings = pd.DataFrame(
        rows,
        columns=[
            "relative_path",
            "exists",
            "bytes_match",
            "fingerprint_match",
        ],
    )
    if findings.empty:
        valid = True
    else:
        fingerprint_ok = findings["fingerprint_match"].fillna(True).astype(bool)
        valid = bool((findings["exists"].astype(bool) & findings["bytes_match"].astype(bool) & fingerprint_ok).all())
    return EyeStorageValidation(
        valid=valid,
        findings=findings,
        metadata=dict(handle.metadata),
    )


def detect_corrupt_partitions(storage):
    """Return partitions that are missing, truncated, or fingerprint-modified."""
    validation = validate_eye_storage_metadata(
        storage,
        verify_hashes=True,
    )
    findings = validation["findings"]
    if findings.empty:
        out = findings.copy()
    else:
        bad_fingerprint = findings["fingerprint_match"].notna() & ~findings["fingerprint_match"].fillna(False).astype(
            bool
        )
        out = findings.loc[
            ~findings["exists"].astype(bool) | ~findings["bytes_match"].astype(bool) | bad_fingerprint
        ].reset_index(drop=True)
    return _tag_frame(out, "eye_corrupt_partitions")


def storage_transaction_manifest(storage):
    """Return a copy of the storage transaction manifest."""
    return _ensure_storage(storage).transactions.copy()


def migrate_eye_storage_schema(
    storage,
    target_path,
    target_version="2.0.0",
    format=None,
    overwrite=False,
):
    """Collect a store and atomically rewrite it under storage schema 2.0.0."""
    handle = _ensure_storage(storage)
    if str(target_version) != _STORAGE_SCHEMA_VERSION:
        _stop("This release can migrate storage only to schema version 2.0.0.")
    target_format = str(handle.metadata.get("format")) if format is None else _normalize_format(format)
    if target_format == "rds":
        raise EyeProcessBackendError("Native R RDS serialization is not available in eyeprocesspy.")

    tables = list(dict.fromkeys(handle.partitions["table"].dropna().astype(str)))
    data = {table: query_eye_storage(handle, table) for table in tables}
    partition_by = handle.metadata.get("partition_by", [])
    compression = handle.metadata.get("compression", "zstd")
    spec = partition_eye_storage(
        by=partition_by,
        format=target_format,
        compression=compression,
    )
    spec.schema_version = str(target_version)

    result = write_partitioned_eye_storage(
        data,
        target_path,
        spec,
        overwrite=overwrite,
    )
    timestamp = _now_utc()
    migration = pd.DataFrame(
        [
            {
                "transaction_id": (f"tx-migrate-{_hash_int(f'{handle.path}|{Path(target_path).resolve()}'):010d}"),
                "action": "migrate",
                "timestamp_utc": timestamp,
                "tables": ",".join(tables),
                "partitions": int(len(result.partitions)),
                "rows": int(
                    pd.to_numeric(
                        result.partitions["rows"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                ),
                "status": "committed",
            }
        ],
        columns=_TRANSACTION_COLUMNS,
    )
    transactions = pd.concat(
        [result.transactions, migration],
        ignore_index=True,
        sort=False,
    )
    _atomic_write_csv(
        transactions,
        _transactions_path(Path(result.path)),
    )
    result["transactions"] = transactions
    return result


def benchmark_eye_storage(
    x,
    formats=("rds", "csv", "parquet"),
    partition_by=("participant_id", "recording_id"),
    repetitions=3,
    directory=None,
):
    """Benchmark supported partitioned-storage routes.

    Python deliberately excludes native RDS from execution rather than
    benchmarking a non-R serialization under an RDS label.
    """
    try:
        repetitions = int(repetitions)
    except (TypeError, ValueError):
        _stop("`repetitions` must be a positive integer.")
    if repetitions < 1:
        _stop("`repetitions` must be a positive integer.")

    if isinstance(formats, str):
        formats = [formats]
    formats = list(dict.fromkeys(str(value) for value in formats))
    invalid = [value for value in formats if value not in {"rds", "csv", "parquet"}]
    if invalid:
        _stop("Unknown storage formats: " + ", ".join(invalid) + ".")

    supported = [value for value in formats if value != "rds"]
    if not supported:
        raise EyeProcessBackendError("Native R RDS benchmarking is unavailable in eyeprocesspy.")
    if "rds" in formats:
        warnings.warn(
            "Native R RDS benchmarking is unavailable in eyeprocesspy and "
            "is excluded. Use the R package for the RDS benchmark route.",
            RuntimeWarning,
            stacklevel=2,
        )

    root = Path(directory).expanduser().resolve() if directory is not None else Path(tempfile.gettempdir()).resolve()
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    for storage_format in supported:
        for replication in range(1, repetitions + 1):
            path = root / (f"eye-storage-benchmark-{storage_format}-{replication}-{uuid.uuid4().hex}")
            try:
                write_start = time.perf_counter()
                storage = write_partitioned_eye_storage(
                    x,
                    path,
                    partition_eye_storage(
                        by=partition_by,
                        format=storage_format,
                    ),
                    overwrite=True,
                )
                write_seconds = time.perf_counter() - write_start

                read_start = time.perf_counter()
                for table in dict.fromkeys(storage.partitions["table"].dropna().astype(str)):
                    query_eye_storage(storage, table)
                read_seconds = time.perf_counter() - read_start

                rows.append(
                    {
                        "format": storage_format,
                        "replication": replication,
                        "write_seconds": float(write_seconds),
                        "read_seconds": float(read_seconds),
                        "bytes": int(
                            pd.to_numeric(
                                storage.partitions["bytes"],
                                errors="coerce",
                            )
                            .fillna(0)
                            .sum()
                        ),
                        "partitions": int(len(storage.partitions)),
                        "rows": int(
                            pd.to_numeric(
                                storage.partitions["rows"],
                                errors="coerce",
                            )
                            .fillna(0)
                            .sum()
                        ),
                    }
                )
            finally:
                shutil.rmtree(path, ignore_errors=True)

    return _tag_frame(
        pd.DataFrame(rows),
        "eye_storage_benchmark",
    )
