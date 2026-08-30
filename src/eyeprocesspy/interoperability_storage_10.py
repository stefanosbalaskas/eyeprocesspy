"""Storage, Eye-Tracking-BIDS, and external interoperability.

Source port of the still-missing public contracts from frozen eyeprocess
0.11.1 R/020-interoperability-storage.R.

Native R RDS serialization is deliberately not impersonated. The ``rds``
route is accepted by the specification/handle API but write/collect operations
raise :class:`EyeProcessBackendError`. Parquet/Arrow datasets use PyArrow.
"""

from __future__ import annotations

import gzip
import json
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import (
    _assert_eye_dataset,
    add_provenance,
    new_eye_dataset,
)
from .exceptions import EyeProcessBackendError
from .importers import infer_eye_mapping, read_eye_generic
from .schema import (
    SCHEMA_VERSION,
    canonical_table_names,
    new_coordinate_space,
)

__all__ = [
    "EyeStorage",
    "EyeStorageSpec",
    "as_eyeprocess_eyeris",
    "as_eyeprocess_eyetools",
    "as_eyeprocess_eyetrackingr",
    "as_eyeprocess_gazer",
    "as_eyeprocess_pupillometryr",
    "collect_eye_storage",
    "export_eye_bids",
    "eye_storage_spec",
    "import_eye_bids",
    "open_eye_storage",
    "write_eye_storage",
]


@dataclass(slots=True)
class EyeStorageSpec:
    """Python counterpart of the R ``eye_storage_spec`` object."""

    path: str
    format: str
    tables: tuple[str, ...]
    partitioning: tuple[str, ...] | None
    compression: str

    def __repr__(self) -> str:
        return f"<eye_storage_spec format={self.format!r} tables={len(self.tables)} path={self.path!r}>"


class EyeStorage(dict):
    """Lightweight disk-backed storage handle."""

    @property
    def spec(self) -> EyeStorageSpec:
        return self["spec"]

    @property
    def manifest(self) -> pd.DataFrame:
        return self["manifest"]

    def __repr__(self) -> str:
        tables = ", ".join(self.manifest.get("table", pd.Series(dtype=str)).astype(str))
        return f"<eye_storage>\n  Format: {self.spec.format}\n  Path:   {self.spec.path}\n  Tables: {tables}"


def eye_storage_spec(
    path,
    format=("rds", "parquet", "arrow_dataset"),
    tables=None,
    partitioning=None,
    compression="zstd",
):
    """Specify disk-backed storage using the frozen R/020 contract."""
    if isinstance(format, (tuple, list)):
        format = format[0]
    format = str(format)
    if format not in {"rds", "parquet", "arrow_dataset"}:
        raise ValueError("`format` must be 'rds', 'parquet', or 'arrow_dataset'.")

    canonical = canonical_table_names()
    if tables is None:
        tables = canonical
    selected = tuple(name for name in map(str, tables) if name in canonical)
    if not selected:
        raise ValueError("No canonical tables were selected for storage.")

    if partitioning is None:
        partitions = None
    elif isinstance(partitioning, str):
        partitions = (partitioning,)
    else:
        partitions = tuple(map(str, partitioning))

    return EyeStorageSpec(
        path=str(Path(path).expanduser().resolve()),
        format=format,
        tables=selected,
        partitioning=partitions,
        compression=str(compression).strip().lower(),
    )


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except Exception as exc:
        raise EyeProcessBackendError(
            "PyArrow is required for Parquet/Arrow eyeprocess storage. Install eyeprocesspy with the `arrow` extra."
        ) from exc
    return pa, ds, pq


def _resolve_compression(codec, *, allow_fallback):
    pa, _, _ = _require_pyarrow()
    codec = str(codec).strip().lower()
    if not codec:
        raise ValueError("`compression` must be one non-empty codec name.")

    def available(name):
        try:
            return bool(pa.Codec.is_available(name))
        except Exception:
            return False

    if codec == "uncompressed":
        return "NONE"
    if available(codec):
        return codec
    if not allow_fallback:
        raise EyeProcessBackendError(f"Arrow compression codec `{codec}` is unavailable in this PyArrow build.")
    if available("snappy"):
        return "snappy"
    return "NONE"


def _metadata_path(root: Path) -> Path:
    return root / "storage-metadata.pkl"


def _write_metadata(x, root: Path, retain_metadata: bool) -> None:
    metadata = {
        "schema_version": x.schema_version,
        "raw": x.raw if retain_metadata else [],
        "vendor_metadata": x.vendor_metadata if retain_metadata else {},
    }
    with _metadata_path(root).open("wb") as handle:
        pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _read_metadata(root: Path) -> dict[str, Any]:
    path = _metadata_path(root)
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        value = pickle.load(handle)
    return value if isinstance(value, dict) else {}


def _arrow_table(data: pd.DataFrame):
    pa, _, _ = _require_pyarrow()
    try:
        return pa.Table.from_pandas(data, preserve_index=False)
    except Exception as exc:
        raise EyeProcessBackendError(
            "A canonical table could not be converted to Arrow. Inspect object-valued extra columns before storage."
        ) from exc


def write_eye_storage(
    x,
    path,
    format=("rds", "parquet", "arrow_dataset"),
    tables=None,
    partitioning=None,
    compression="zstd",
    overwrite=False,
    retain_metadata=True,
):
    """Write canonical tables to Parquet/Arrow storage.

    The R ``rds`` route is an explicit backend boundary in Python.
    """
    _assert_eye_dataset(x)
    if isinstance(format, (tuple, list)):
        format = format[0]
    spec = eye_storage_spec(
        path,
        format=format,
        tables=tables,
        partitioning=partitioning,
        compression=compression,
    )
    target = Path(spec.path)

    if spec.format == "rds":
        raise EyeProcessBackendError(
            "Native R RDS serialization is not available in eyeprocesspy. "
            "Use `parquet` or `arrow_dataset`, or write RDS from the R package."
        )

    if target.exists() and not overwrite:
        raise ValueError(f"Storage target already exists: {target}")
    if target.exists() and overwrite:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    _, ds, pq = _require_pyarrow()
    target.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_compression(
        spec.compression,
        allow_fallback=spec.compression == "zstd",
    )
    spec.compression = "uncompressed" if resolved == "NONE" else resolved

    manifest_rows = []
    for name in spec.tables:
        data = x[name]
        arrow_table = _arrow_table(data)

        if spec.format == "parquet":
            table_path = target / f"{name}.parquet"
            pq.write_table(arrow_table, table_path, compression=resolved)
        else:
            table_path = target / name
            table_path.mkdir(parents=True, exist_ok=True)
            partitions = [value for value in (spec.partitioning or ()) if value in data.columns]
            if data.empty:
                pq.write_table(
                    arrow_table,
                    table_path / "part-0.parquet",
                    compression=resolved,
                )
            else:
                ds.write_dataset(
                    arrow_table,
                    base_dir=str(table_path),
                    format="parquet",
                    partitioning=partitions or None,
                    partitioning_flavor="hive" if partitions else None,
                    existing_data_behavior="overwrite_or_ignore",
                    file_options=ds.ParquetFileFormat().make_write_options(compression=resolved),
                )

        manifest_rows.append(
            {
                "table": name,
                "path": str(table_path.resolve()),
                "rows": int(len(data)),
            }
        )

    manifest = pd.DataFrame(manifest_rows, columns=["table", "path", "rows"])
    manifest.to_csv(target / "storage-manifest.csv", index=False, lineterminator="\n")
    _write_metadata(x, target, bool(retain_metadata))

    return EyeStorage(spec=spec, manifest=manifest)


def open_eye_storage(path, format=None):
    """Open a storage handle without collecting all canonical tables."""
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Storage path does not exist: {target}")

    if format is None:
        if target.is_dir():
            manifest_path = target / "storage-manifest.csv"
            if not manifest_path.is_file():
                raise ValueError(f"Storage manifest is missing from: {target}")
            manifest = pd.read_csv(manifest_path)
            paths = manifest.get("path", pd.Series(dtype=str)).astype(str)
            format = "parquet" if len(paths) and paths.str.lower().str.endswith(".parquet").all() else "arrow_dataset"
        else:
            format = "rds"

    if isinstance(format, (tuple, list)):
        format = format[0]
    spec = eye_storage_spec(target, format=format)

    if spec.format == "rds":
        manifest = pd.DataFrame([{"table": "eye_dataset", "path": str(target), "rows": np.nan}])
    else:
        manifest_path = target / "storage-manifest.csv"
        if not manifest_path.is_file():
            raise ValueError(f"Storage manifest is missing from: {target}")
        manifest = pd.read_csv(manifest_path)

    return EyeStorage(spec=spec, manifest=manifest)


def collect_eye_storage(x, tables=None):
    """Collect a disk-backed storage handle into an ``EyeDataset``."""
    if isinstance(x, (str, Path)):
        x = open_eye_storage(x)
    if not isinstance(x, EyeStorage):
        raise TypeError("Expected an `EyeStorage` handle or storage path.")
    if x.spec.format == "rds":
        raise EyeProcessBackendError("Native R RDS collection is not available in eyeprocesspy.")

    _, ds, pq = _require_pyarrow()
    available = list(x.manifest["table"].astype(str))
    selected = available if tables is None else [name for name in map(str, tables) if name in available]
    table_values: dict[str, pd.DataFrame] = {}
    for name in selected:
        row = x.manifest.loc[x.manifest["table"].astype(str).eq(name)].iloc[0]
        source = Path(str(row["path"]))
        if x.spec.format == "parquet":
            data = pq.read_table(source).to_pandas()
        else:
            data = ds.dataset(str(source), format="parquet", partitioning="hive").to_table().to_pandas()
        table_values[name] = data

    metadata = _read_metadata(Path(x.spec.path))
    kwargs = {name: value for name, value in table_values.items() if name in canonical_table_names()}
    return new_eye_dataset(
        **kwargs,
        raw=metadata.get("raw", []),
        vendor_metadata=metadata.get("vendor_metadata", {}),
        schema_version=metadata.get("schema_version", SCHEMA_VERSION),
        validate=True,
    )


def _sanitize_id(value, prefix="id"):
    if value is None or pd.isna(value) or not str(value).strip():
        value = prefix
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(value))
    return cleaned or prefix


def _first_text(values, default="unknown"):
    for value in pd.Series(values).tolist():
        if pd.isna(value):
            continue
        if str(value).strip():
            return str(value)
    return default


def _sampling_rate(x, recording_id):
    streams = x["streams"]
    rows = streams[streams["recording_id"].astype(str).eq(str(recording_id))]
    for column in ("observed_rate_hz", "nominal_rate_hz"):
        values = pd.to_numeric(rows[column], errors="coerce")
        values = values[np.isfinite(values.to_numpy(dtype=float)) & (values > 0)]
        if len(values):
            return float(values.iloc[0])
    return np.nan


def _timestamp_unit(x, recording_id):
    streams = x["streams"]
    rows = streams[
        streams["recording_id"].astype(str).eq(str(recording_id))
        & streams["stream_type"].astype("string").str.contains("gaze", case=False, na=False)
    ]
    return _first_text(rows["timestamp_unit"], "unknown")


def _coordinate_metadata(x, recording_id):
    gaze = x["gaze_samples"]
    gaze = gaze[gaze["recording_id"].astype(str).eq(str(recording_id))]
    coord_id = _first_text(gaze["coordinate_space_id"], "")
    spaces = x["coordinate_spaces"]
    rows = spaces[spaces["coordinate_space_id"].astype(str).eq(coord_id)]
    if rows.empty:
        return {"system": "custom", "x_unit": "unknown", "y_unit": "unknown"}

    row = rows.iloc[0]
    space_type = str(row["space_type"])
    if re.search(r"display|screen", space_type, re.I):
        system = "gaze-on-screen"
    elif re.search(r"world", space_type, re.I):
        system = "gaze-in-world"
    elif re.search(r"head|user|direction", space_type, re.I):
        system = "eye-in-head"
    else:
        system = "custom"
    return {
        "system": system,
        "x_unit": str(row["x_unit"]),
        "y_unit": str(row["y_unit"]),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def _write_json(value, path: Path):
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_tsv_gz_no_header(data: pd.DataFrame, path: Path):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        data.to_csv(
            handle,
            sep="\t",
            header=False,
            index=False,
            na_rep="n/a",
            lineterminator="\n",
        )


def _recorded_eye(value):
    text = str(value).strip().lower()
    if text in {"l", "left", "os"}:
        return "left"
    if text in {"r", "right", "od"}:
        return "right"
    return "cyclopean"


def export_eye_bids(
    x,
    path,
    task="task",
    dataset_name="eyeprocess eye-tracking dataset",
    overwrite=False,
    screen_distance_m=None,
    screen_size_m=None,
):
    """Export Eye-Tracking-BIDS physiological files and sidecars."""
    _assert_eye_dataset(x)
    root = Path(path).expanduser().resolve()
    if root.exists() and root.is_file():
        raise ValueError(f"BIDS output path exists as a file: {root}")
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise ValueError(f"BIDS output directory is not empty: {root}")
    if root.exists() and overwrite:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    task = _sanitize_id(task, "task")
    _write_json(
        {
            "Name": str(dataset_name),
            "BIDSVersion": "1.11.1",
            "DatasetType": "raw",
            "GeneratedBy": [{"Name": "eyeprocesspy"}],
        },
        root / "dataset_description.json",
    )

    recordings = x["recordings"].copy()
    original = recordings["participant_id"].astype("string")
    bids_ids = "sub-" + original.map(lambda value: _sanitize_id(value, "unknown"))
    unique_map = pd.DataFrame({"source": original, "participant_id": bids_ids}).drop_duplicates()
    if unique_map["participant_id"].duplicated().any():
        raise ValueError(
            "Participant identifiers collide after BIDS sanitization: " + ", ".join(unique_map["source"].astype(str))
        )
    unique_map[["participant_id"]].to_csv(
        root / "participants.tsv",
        sep="\t",
        index=False,
        na_rep="n/a",
        lineterminator="\n",
    )

    run_counts: dict[str, int] = {}
    written = []
    for _, rec in recordings.iterrows():
        rec_id = str(rec["recording_id"])
        participant = "sub-" + _sanitize_id(rec["participant_id"], "unknown")
        run_counts[participant] = run_counts.get(participant, 0) + 1
        run_label = f"{run_counts[participant]:02d}"
        beh = root / participant / "beh"
        beh.mkdir(parents=True, exist_ok=True)

        gaze = x["gaze_samples"][x["gaze_samples"]["recording_id"].astype(str).eq(rec_id)].copy()
        eye_all = x["eye_samples"][x["eye_samples"]["recording_id"].astype(str).eq(rec_id)].copy()
        eyes = [value for value in pd.unique(eye_all["eye"].dropna().astype(str)) if value.strip()]
        if not eyes:
            eyes = ["cyclopean"]

        coord = _coordinate_metadata(x, rec_id)
        rate = _sampling_rate(x, rec_id)
        if not np.isfinite(rate):
            rate = float(pd.to_numeric(pd.Series([rec["nominal_sampling_rate"]]), errors="coerce").fillna(1).iloc[0])
        if not np.isfinite(rate) or rate <= 0:
            rate = 1.0

        for eye_index, eye in enumerate(eyes, start=1):
            eye_label = f"eye{eye_index}"
            eye_data = eye_all[eye_all["eye"].astype(str).eq(str(eye))].copy()
            cols = pd.DataFrame(
                {
                    "timestamp": gaze["timestamp_native"].to_numpy(),
                    "x_coordinate": pd.to_numeric(gaze["gaze_x"], errors="coerce"),
                    "y_coordinate": pd.to_numeric(gaze["gaze_y"], errors="coerce"),
                }
            )

            if not eye_data.empty:
                eye_by_seconds = {}
                for row in eye_data.itertuples(index=False):
                    key = getattr(row, "timestamp_seconds", None)
                    if pd.notna(key):
                        eye_by_seconds[str(key)] = getattr(row, "pupil_diameter", np.nan)
                eye_by_native = {}
                for row in eye_data.itertuples(index=False):
                    key = getattr(row, "timestamp_native", None)
                    if pd.notna(key):
                        eye_by_native[str(key)] = getattr(row, "pupil_diameter", np.nan)
                pupils = []
                for native, seconds in zip(
                    gaze["timestamp_native"],
                    gaze["timestamp_seconds"],
                    strict=False,
                ):
                    value = eye_by_seconds.get(str(seconds), np.nan)
                    if pd.isna(value):
                        value = eye_by_native.get(str(native), np.nan)
                    pupils.append(value)
                cols["pupil_size"] = pupils

            base = f"{participant}_task-{task}_run-{run_label}_recording-{eye_label}_physio"
            tsv = beh / f"{base}.tsv.gz"
            sidecar_path = beh / f"{base}.json"
            _write_tsv_gz_no_header(cols, tsv)

            sidecar = {
                "Columns": list(cols.columns),
                "PhysioType": "eyetrack",
                "StartTime": 0,
                "RecordedEye": _recorded_eye(eye),
                "SampleCoordinateSystem": coord["system"],
                "SamplingFrequency": rate,
                "Manufacturer": _json_safe(rec["vendor"]),
                "ManufacturersModelName": _json_safe(rec["device_model"]),
                "SoftwareVersions": _json_safe(rec["software_version"]),
                "timestamp": {
                    "Description": "Native timestamp issued by the eye tracker",
                    "Units": _timestamp_unit(x, rec_id),
                },
                "x_coordinate": {
                    "LongName": "Gaze position (x)",
                    "Units": coord["x_unit"],
                },
                "y_coordinate": {
                    "LongName": "Gaze position (y)",
                    "Units": coord["y_unit"],
                },
            }
            if "pupil_size" in cols:
                sidecar["pupil_size"] = {
                    "Description": ("Recorded pupil diameter or area as declared by the source"),
                    "Units": _first_text(eye_data["pupil_unit"], "unknown"),
                }
            _write_json(sidecar, sidecar_path)
            written.append(
                {
                    "recording_id": rec_id,
                    "participant_id": participant,
                    "eye": eye,
                    "tsv": str(tsv.resolve()),
                    "json": str(sidecar_path.resolve()),
                }
            )

        events = x["events"][x["events"]["recording_id"].astype(str).eq(rec_id)].copy()
        if not events.empty:
            duration = pd.to_numeric(events["duration"], errors="coerce")
            duration = duration.where(np.isfinite(duration), 0)
            ev = pd.DataFrame(
                {
                    "onset": pd.to_numeric(events["timestamp_seconds"], errors="coerce"),
                    "duration": duration,
                    "trial_type": events["event_type"],
                    "value": events["event_value"],
                }
            )
            event_base = f"{participant}_task-{task}_run-{run_label}_events"
            ev.to_csv(
                beh / f"{event_base}.tsv",
                sep="\t",
                index=False,
                na_rep="n/a",
                lineterminator="\n",
            )
            event_meta = {
                "TaskName": task,
                "onset": {"Description": "Event onset in seconds"},
                "duration": {"Description": "Event duration in seconds"},
                "trial_type": {"Description": "Event type"},
            }
            if screen_distance_m is not None or screen_size_m is not None:
                event_meta["StimulusPresentation"] = {
                    "ScreenDistance": screen_distance_m,
                    "ScreenOrigin": ["top", "left"],
                    "ScreenResolution": [
                        _json_safe(rec["screen_width_px"]),
                        _json_safe(rec["screen_height_px"]),
                    ],
                    "ScreenSize": screen_size_m,
                }
            _write_json(event_meta, beh / f"{event_base}.json")

    manifest = pd.DataFrame(
        written,
        columns=["recording_id", "participant_id", "eye", "tsv", "json"],
    )
    manifest.to_csv(
        root / "eyeprocess-bids-manifest.csv",
        index=False,
        lineterminator="\n",
    )
    return manifest


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _timestamp_to_seconds(value, unit="unknown", sampling_rate=np.nan):
    values = pd.to_numeric(pd.Series(value), errors="coerce")
    unit = str(unit or "unknown").strip().lower()
    multipliers = {
        "s": 1.0,
        "sec": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "ms": 1e-3,
        "millisecond": 1e-3,
        "milliseconds": 1e-3,
        "us": 1e-6,
        "microsecond": 1e-6,
        "microseconds": 1e-6,
        "ns": 1e-9,
        "nanosecond": 1e-9,
        "nanoseconds": 1e-9,
    }
    if unit in multipliers and np.isfinite(values.to_numpy(dtype=float)).any():
        return values * multipliers[unit]
    if np.isfinite(sampling_rate) and sampling_rate > 0:
        return pd.Series(np.arange(len(values)) / float(sampling_rate))
    return values


def _acquisition_recording_id(filename: str, participant_id: str):
    acquisition = re.sub(
        r"_recording-[^_]+_physio\.tsv\.gz$",
        "",
        filename,
    )
    return "rec_" + _sanitize_id(acquisition, participant_id)


def import_eye_bids(path, validate=True):
    """Import Eye-Tracking-BIDS physiological recordings."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"BIDS dataset root does not exist: {root}")
    files = sorted(root.rglob("*_recording-*_physio.tsv.gz"))
    if not files:
        raise ValueError("No Eye-Tracking-BIDS physiological files were found.")

    recordings: dict[str, dict[str, Any]] = {}
    streams: dict[str, dict[str, Any]] = {}
    gaze_rows = []
    eye_rows = []
    for file_index, file in enumerate(files, start=1):
        sidecar_path = Path(str(file)[:-7] + ".json")
        if not sidecar_path.is_file():
            raise ValueError(f"Missing BIDS sidecar: {sidecar_path}")
        meta = _read_json(sidecar_path)
        required = {
            "Columns",
            "PhysioType",
            "StartTime",
            "RecordedEye",
            "SamplingFrequency",
        }
        missing = sorted(required - set(meta))
        if missing:
            raise ValueError("BIDS sidecar is missing required fields: " + ", ".join(missing))
        if str(meta["PhysioType"]).lower() != "eyetrack":
            raise ValueError(f"BIDS PhysioType must be `eyetrack`: {sidecar_path}")
        recorded_eye = str(meta["RecordedEye"]).lower()
        if recorded_eye not in {"left", "right", "cyclopean"}:
            raise ValueError(f"Unsupported BIDS RecordedEye value: {recorded_eye}")
        start_time = float(meta["StartTime"])
        rate = float(meta["SamplingFrequency"])
        if not np.isfinite(start_time):
            raise ValueError(f"BIDS StartTime must be finite: {sidecar_path}")
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError(f"BIDS SamplingFrequency must be positive: {sidecar_path}")

        columns = list(map(str, meta["Columns"]))
        if not {"timestamp", "x_coordinate", "y_coordinate"} <= set(columns):
            raise ValueError(f"BIDS Columns must include timestamp, x_coordinate, and y_coordinate: {sidecar_path}")
        with gzip.open(file, "rt", encoding="utf-8") as handle:
            data = pd.read_csv(
                handle,
                sep="\t",
                header=None,
                names=columns,
                na_values=["n/a", "NA"],
            )
        if data.shape[1] != len(columns):
            raise ValueError(f"BIDS sidecar Columns length does not match data: {file}")

        relative = file.relative_to(root).as_posix()
        participant = relative.split("/", 1)[0]
        participant_id = re.sub(r"^sub-", "", participant)
        match = re.search(r"_recording-([^_]+)_physio\.tsv\.gz$", file.name)
        if not match:
            raise ValueError(f"Cannot parse BIDS recording entity: {file}")
        rec_label = match.group(1)
        rec_id = _acquisition_recording_id(file.name, participant_id)
        stream_id = f"{rec_id}_{rec_label}"

        recordings[rec_id] = {
            "recording_id": rec_id,
            "participant_id": participant_id,
            "session_id": pd.NA,
            "vendor": meta.get("Manufacturer", "BIDS"),
            "vendor_family": "bids",
            "device_model": meta.get("ManufacturersModelName", pd.NA),
            "software_name": "BIDS",
            "software_version": meta.get("SoftwareVersions", pd.NA),
            "nominal_sampling_rate": rate,
            "source_file_set": str(file),
        }
        streams[stream_id] = {
            "stream_id": stream_id,
            "recording_id": rec_id,
            "stream_type": "gaze_eye",
            "source_device": meta.get("ManufacturersModelName", pd.NA),
            "source_clock": "bids_timestamp",
            "sampling_type": "regular",
            "nominal_rate_hz": rate,
            "observed_rate_hz": rate,
            "timestamp_unit": meta.get("timestamp", {}).get("Units", "unknown"),
            "value_unit": pd.NA,
            "coordinate_space_id": "coord_bids",
            "processing_level": "raw",
        }

        seconds = (
            _timestamp_to_seconds(
                data["timestamp"],
                meta.get("timestamp", {}).get("Units", "unknown"),
                rate,
            )
            + start_time
        )
        for row_index, row in data.iterrows():
            source_id = f"source_{file_index}_{row_index + 1}"
            key = (
                str(rec_id),
                str(row["timestamp"]),
                str(row["x_coordinate"]),
                str(row["y_coordinate"]),
            )
            gaze_rows.append(
                {
                    "recording_id": rec_id,
                    "stream_id": stream_id,
                    "sample_id": source_id,
                    "timestamp_native": row["timestamp"],
                    "timestamp_seconds": seconds.iloc[row_index],
                    "gaze_x": pd.to_numeric(pd.Series([row["x_coordinate"]]), errors="coerce").iloc[0],
                    "gaze_y": pd.to_numeric(pd.Series([row["y_coordinate"]]), errors="coerce").iloc[0],
                    "valid": bool(
                        np.isfinite(pd.to_numeric(pd.Series([row["x_coordinate"]]), errors="coerce").iloc[0])
                        and np.isfinite(pd.to_numeric(pd.Series([row["y_coordinate"]]), errors="coerce").iloc[0])
                    ),
                    "confidence": np.nan,
                    "coordinate_space_id": "coord_bids",
                    "_gaze_key": key,
                }
            )
            if "pupil_size" in data.columns:
                pupil = pd.to_numeric(pd.Series([row["pupil_size"]]), errors="coerce").iloc[0]
                eye_rows.append(
                    {
                        "recording_id": rec_id,
                        "sample_id": source_id,
                        "timestamp_native": row["timestamp"],
                        "timestamp_seconds": seconds.iloc[row_index],
                        "eye": recorded_eye,
                        "pupil_diameter": pupil,
                        "pupil_unit": meta.get("pupil_size", {}).get("Units", "unknown"),
                        "pupil_valid": bool(np.isfinite(pupil)),
                        "detector_method": "BIDS import",
                    }
                )

    gaze_all = pd.DataFrame(gaze_rows)
    if gaze_all.empty:
        raise ValueError("Eye-Tracking-BIDS files contained no gaze rows.")

    canonical_by_key: dict[tuple[str, ...], str] = {}
    source_to_canonical: dict[str, str] = {}
    for source_id, raw_key in zip(
        gaze_all["sample_id"],
        gaze_all["_gaze_key"],
        strict=False,
    ):
        key = tuple(raw_key)
        if key not in canonical_by_key:
            canonical_by_key[key] = f"bids_sample_{len(canonical_by_key) + 1}"
        source_to_canonical[str(source_id)] = canonical_by_key[key]

    gaze_all["sample_id"] = gaze_all["sample_id"].astype(str).map(source_to_canonical)
    gaze_all = (
        gaze_all.drop_duplicates(subset=["_gaze_key"], keep="first").drop(columns=["_gaze_key"]).reset_index(drop=True)
    )

    if eye_rows:
        eye_all = pd.DataFrame(eye_rows)
        eye_all["sample_id"] = eye_all["sample_id"].astype(str).map(source_to_canonical)
        if eye_all["sample_id"].isna().any():
            raise ValueError("Could not map BIDS pupil samples to canonical gaze samples.")
    else:
        eye_all = None

    event_rows = []
    for file in sorted(root.rglob("*_events.tsv")):
        data = pd.read_csv(
            file,
            sep="\t",
            na_values=["n/a", "NA"],
        )
        required = {"onset", "duration", "trial_type"}
        if not required <= set(data.columns):
            raise ValueError(f"BIDS events table lacks onset/duration/trial_type: {file}")
        relative = file.relative_to(root).as_posix()
        participant = relative.split("/", 1)[0]
        participant_id = re.sub(r"^sub-", "", participant)
        acquisition = re.sub(r"_events\.tsv$", "", file.name)
        rec_id = "rec_" + _sanitize_id(acquisition, participant_id)
        if rec_id not in recordings:
            continue
        for index, row in data.iterrows():
            value = row["value"] if "value" in data.columns else pd.NA
            event_rows.append(
                {
                    "event_id": f"{rec_id}_bids_event_{index + 1}",
                    "recording_id": rec_id,
                    "timestamp_native": row["onset"],
                    "timestamp_seconds": row["onset"],
                    "event_type": row["trial_type"],
                    "event_name": row["trial_type"],
                    "event_value": value,
                    "duration": row["duration"],
                    "source": "BIDS events.tsv",
                    "native_record": pd.NA,
                    "trial_id": value,
                    "stimulus_id": pd.NA,
                }
            )

    coordinate = new_coordinate_space(
        "coord_bids",
        space_type="custom",
        origin="BIDS",
        x_unit="declared",
        y_unit="declared",
    )
    out = new_eye_dataset(
        recordings=pd.DataFrame(recordings.values()),
        streams=pd.DataFrame(streams.values()),
        gaze_samples=gaze_all,
        eye_samples=eye_all,
        events=pd.DataFrame(event_rows) if event_rows else None,
        coordinate_spaces=coordinate,
        vendor_metadata={"bids_root": str(root)},
        validate=bool(validate),
    )
    return add_provenance(
        out,
        "import_eye_bids",
        "dataset",
        f"files={len(files)}",
        source_files=[str(file) for file in files],
    )


def _as_external(x, mapping=None, vendor="external", **kwargs):
    if isinstance(x, pd.DataFrame):
        data = x.copy()
    else:
        try:
            data = pd.DataFrame(x)
        except Exception as exc:
            raise TypeError("`x` must be coercible to a pandas DataFrame.") from exc
    if mapping is None:
        mapping = infer_eye_mapping(data)
    out = read_eye_generic(data, mapping=mapping, **kwargs)
    if not out["recordings"].empty:
        out["recordings"].loc[:, "vendor"] = vendor
        out["recordings"].loc[:, "vendor_family"] = vendor
    return add_provenance(
        out,
        "external_adapter",
        "dataset",
        f"source_class={type(x).__name__};vendor={vendor}",
    )


def as_eyeprocess_eyetools(x, mapping=None, **kwargs):
    return _as_external(x, mapping, "eyetools", **kwargs)


def as_eyeprocess_eyetrackingr(x, mapping=None, **kwargs):
    return _as_external(x, mapping, "eyetrackingR", **kwargs)


def as_eyeprocess_gazer(x, mapping=None, **kwargs):
    return _as_external(x, mapping, "gazeR", **kwargs)


def as_eyeprocess_eyeris(x, mapping=None, **kwargs):
    return _as_external(x, mapping, "eyeris", **kwargs)


def as_eyeprocess_pupillometryr(x, mapping=None, **kwargs):
    return _as_external(x, mapping, "PupillometryR", **kwargs)
