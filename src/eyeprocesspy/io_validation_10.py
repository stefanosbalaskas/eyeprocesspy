from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .adapters import _ADAPTERS, detect_eye_format, read_eye_export
from .coordinates import audit_coordinate_spaces
from .dataset import (
    EyeDataset,
    _assert_eye_dataset,
    add_provenance,
    is_eye_dataset,
    new_eye_dataset,
    provenance_manifest,
    validate_eye_dataset,
)
from .exceptions import EyeProcessBackendError, EyeProcessValidationError
from .foundation_09 import (
    analysis_readiness,
    audit_event_order,
    audit_sampling_rate,
    audit_signal_quality,
    audit_trial_coverage,
    interpretive_warnings,
)
from .importers import _read_delimited, infer_eye_mapping, read_eye_generic
from .schema import canonical_table_names, empty_eye_table, schema_table, standardize_eye_table
from .timebase import audit_timebase

_CANONICAL_NA_TOKEN = "__EYEPROCESS_MISSING_6E7A4D2F__"
_SERIALIZATION_JSON = ".eyeprocess-serialization.json"
_VENDOR_METADATA_JSON = "vendor_metadata.json"
_RAW_JSON = "raw.json"


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _jsonable(value: Any) -> Any:
    if value is pd.NA:
        return None
    if isinstance(value, pd.DataFrame):
        return value.where(pd.notna(value), None).to_dict(orient="records")
    if isinstance(value, pd.Series):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"class": type(value).__name__, "repr": repr(value)}


def _dtype_family(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_string_dtype(series):
        return "string"
    nonmissing = series.dropna()
    if nonmissing.empty:
        return "object"
    if nonmissing.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        return "boolean"
    if nonmissing.map(lambda x: isinstance(x, (int, float, np.number)) and not isinstance(x, bool)).all():
        return "numeric"
    return "object"


def _restore_column(series: pd.Series, family: str) -> pd.Series:
    if family == "numeric":
        return pd.to_numeric(series, errors="coerce")
    if family == "boolean":
        z = series.astype("string").str.strip().str.lower()
        out = pd.Series(pd.NA, index=series.index, dtype="boolean")
        out[z.isin(["true", "t", "1", "yes", "y"])] = True
        out[z.isin(["false", "f", "0", "no", "n"])] = False
        return out
    if family == "datetime":
        return pd.to_datetime(series, errors="coerce", utc=True)
    if family == "string":
        return series.astype("string")
    return series


def _polygon_to_text(value: Any) -> Any:
    if value is None or value is pd.NA:
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return pd.NA
    arr = arr.reshape(-1, 2)
    return ";".join(",".join(format(float(v), ".17g") for v in row) for row in arr)


def _polygon_from_text(value: Any) -> Any:
    if value is None or pd.isna(value) or str(value) == "":
        return None
    rows = []
    for row in str(value).split(";"):
        parts = row.split(",")
        if len(parts) != 2:
            raise EyeProcessValidationError(f"Invalid serialized AOI polygon row: {row!r}")
        rows.append([float(parts[0]), float(parts[1])])
    return np.asarray(rows, dtype=float)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


# R/015-export-report-bridges.R --------------------------------------------


def write_eye_dataset(
    x,
    path,
    format=None,
    include_raw=False,
    overwrite=False,
    manifest=True,
):
    """Write the canonical folder format; native RDS remains an explicit backend boundary."""
    _assert_eye_dataset(x)
    p = Path(path).expanduser()
    if format is None:
        format = "rds" if p.suffix.lower() == ".rds" else "folder"
    if format not in {"folder", "rds"}:
        raise EyeProcessValidationError("`format` must be 'folder' or 'rds'.")
    if format == "rds":
        raise EyeProcessBackendError(
            "Native R `.rds` serialization is R-specific. eyeprocesspy does not emit a "
            "different format under an `.rds` name; use format='folder' for the "
            "canonical cross-language table representation."
        )
    if p.exists() and p.is_file():
        raise EyeProcessValidationError("Canonical folder output path exists as a file.")
    if p.exists() and any(p.iterdir()) and not overwrite:
        raise EyeProcessValidationError("Output directory is not empty; use `overwrite=True`.")
    p.mkdir(parents=True, exist_ok=True)

    managed = [p / f"{name}.csv" for name in canonical_table_names()]
    managed += [
        p / _SERIALIZATION_JSON,
        p / _VENDOR_METADATA_JSON,
        p / _RAW_JSON,
        p / "manifest.json",
    ]
    if overwrite:
        for item in managed:
            if item.is_file():
                item.unlink()

    dtypes: dict[str, dict[str, str]] = {}
    for name in canonical_table_names():
        d = x[name].copy()
        dtypes[name] = {col: _dtype_family(d[col]) for col in d.columns}
        if name == "aoi_geometry" and "polygon" in d.columns:
            d["polygon"] = d["polygon"].map(_polygon_to_text)
        d.to_csv(
            p / f"{name}.csv",
            index=False,
            na_rep=_CANONICAL_NA_TOKEN,
            lineterminator="\n",
            float_format="%.17g",
        )

    serialization = {
        "format": "eyeprocess-canonical-folder",
        "format_version": 1,
        "package": "eyeprocesspy",
        "schema_version": str(x.schema_version),
        "na_token": _CANONICAL_NA_TOKEN,
        "created": _now_utc(),
        "column_families": dtypes,
        "rds_sidecars": False,
    }
    _write_json(p / _SERIALIZATION_JSON, serialization)
    _write_json(p / _VENDOR_METADATA_JSON, x.vendor_metadata)
    if include_raw:
        _write_json(p / _RAW_JSON, x.raw)
    if manifest:
        _write_json(p / "manifest.json", provenance_manifest(x))
    return str(p.resolve())


def read_eye_dataset(path, validate=True):
    """Read the canonical folder format written by R/Python-compatible table serialization."""
    p = Path(path).expanduser()
    if p.is_file() and p.suffix.lower() == ".rds":
        raise EyeProcessBackendError(
            "Native R `.rds` deserialization requires R. Use the canonical folder representation for eyeprocesspy."
        )
    if not p.is_dir():
        raise EyeProcessValidationError(f"Canonical dataset folder does not exist: {path}")

    serialization = {}
    meta_path = p / _SERIALIZATION_JSON
    if meta_path.exists():
        try:
            serialization = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            serialization = {}
    na_token = serialization.get("na_token", _CANONICAL_NA_TOKEN)
    families = serialization.get("column_families", {})

    tables = {}
    for name in canonical_table_names():
        f = p / f"{name}.csv"
        if not f.exists():
            tables[name] = empty_eye_table(name)
            continue
        d = pd.read_csv(
            f,
            dtype=object,
            keep_default_na=False,
            na_values=[na_token],
            encoding="utf-8",
        )
        for col, family in families.get(name, {}).items():
            if col in d.columns:
                d[col] = _restore_column(d[col], family)
        if name == "aoi_geometry" and "polygon" in d.columns:
            d["polygon"] = d["polygon"].map(_polygon_from_text)
        tables[name] = standardize_eye_table(d, name)

    vendor_metadata = {}
    vm = p / _VENDOR_METADATA_JSON
    if vm.exists():
        try:
            vendor_metadata = json.loads(vm.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            vendor_metadata = {}
    raw: Any = []
    rf = p / _RAW_JSON
    if rf.exists():
        try:
            raw = json.loads(rf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw = []

    out = new_eye_dataset(
        **tables,
        raw=raw,
        vendor_metadata=vendor_metadata,
        validate=False,
        schema_version=serialization.get("schema_version") or "0.1.0",
    )
    if validate:
        out.validation = validate_eye_dataset(out)
    return out


def export_canonical(*args, **kwargs):
    """Alias of :func:`write_eye_dataset`."""
    return write_eye_dataset(*args, **kwargs)


def import_canonical(*args, **kwargs):
    """Alias of :func:`read_eye_dataset`."""
    return read_eye_dataset(*args, **kwargs)


def write_provenance(x, path, format=("csv", "json", "rds")):
    """Write provenance as CSV or JSON; native RDS is explicitly gated."""
    _assert_eye_dataset(x)
    if isinstance(format, (list, tuple)):
        format = format[0]
    if format not in {"csv", "json", "rds"}:
        raise EyeProcessValidationError("Invalid provenance format.")
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    if format == "rds":
        raise EyeProcessBackendError("RDS provenance serialization is R-specific.")
    if format == "csv":
        x["provenance"].to_csv(p, index=False, lineterminator="\n")
    else:
        _write_json(p, provenance_manifest(x))
    return str(p.resolve())


def _markdown_table(d: pd.DataFrame, max_rows: int = 50) -> str:
    if not isinstance(d, pd.DataFrame) or d.empty:
        return ""
    z = d.head(max_rows).copy()
    for col in z.columns:
        z[col] = z[col].map(lambda v: "" if pd.isna(v) else str(v).replace("|", r"\|"))
    header = "| " + " | ".join(map(str, z.columns)) + " |"
    rule = "| " + " | ".join(["---"] * len(z.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in z.astype(str).to_numpy()]
    return "\n".join([header, rule, *rows])


def _save_plot(path: Path, fn) -> None:
    obj = fn()
    fig = getattr(obj, "figure", None)
    if fig is None and isinstance(obj, tuple) and obj:
        fig = getattr(obj[-1], "figure", None)
    if fig is None:
        import matplotlib.pyplot as plt

        fig = plt.gcf()
    fig.savefig(path, dpi=120, bbox_inches="tight")


def report_eye_dataset(
    x,
    path="eyeprocess-report.md",
    title="eyeprocess data and analysis report",
    include_plots=False,
    plot_directory=None,
):
    """Write the frozen-R dataset/readiness/quality/provenance Markdown report."""
    _assert_eye_dataset(x)
    readiness = analysis_readiness(x)
    validation = validate_eye_dataset(x)
    sampling = audit_sampling_rate(x)
    quality = audit_signal_quality(x)
    trials = audit_trial_coverage(x)
    warnings_table = interpretive_warnings()
    recordings = x["recordings"]
    participants = recordings["participant_id"].dropna().astype(str).nunique() if "participant_id" in recordings else 0
    lines = [
        f"# {title}",
        "",
        f"Generated: {_now_utc()}",
        "",
        "## Dataset",
        "",
        f"- Schema version: `{x.schema_version}`",
        f"- Recordings: {len(x['recordings'])}",
        f"- Participants: {participants}",
        f"- Gaze samples: {len(x['gaze_samples'])}",
        f"- Eye samples: {len(x['eye_samples'])}",
        f"- Ocular episodes: {len(x['episodes'])}",
        f"- Trials: {int(x['intervals']['interval_type'].eq('trial').sum())}",
        f"- Responses: {len(x['responses'])}",
        f"- Biometric observations: {len(x['biometrics'])}",
        f"- Derived features: {len(x['features'])}",
        "",
        "## Readiness",
        "",
        _markdown_table(readiness),
        "",
        "## Validation",
        "",
        _markdown_table(validation) if len(validation) else "No validation issues detected.",
        "",
        "## Sampling-rate audit",
        "",
        _markdown_table(sampling) if len(sampling) else "No gaze sampling-rate data available.",
        "",
        "## Signal quality",
        "",
        _markdown_table(quality) if len(quality) else "No signal-quality data available.",
        "",
        "## Trial coverage",
        "",
        _markdown_table(trials) if len(trials) else "No trials available.",
        "",
        "## Responsible interpretation",
        "",
        _markdown_table(warnings_table),
        "",
        "## Provenance",
        "",
        _markdown_table(x["provenance"]) if len(x["provenance"]) else "No provenance records available.",
    ]
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if include_plots:
        import eyeprocesspy as ep

        plot_dir = (
            Path(plot_directory) if plot_directory is not None else p.with_suffix("").with_name(p.stem + "-figures")
        )
        plot_dir.mkdir(parents=True, exist_ok=True)
        for filename, name in [
            ("overview.png", "plot_eye_overview"),
            ("signal-quality.png", "plot_signal_quality"),
            ("sampling-rate.png", "plot_sampling_rate"),
        ]:
            fn = getattr(ep, name, None)
            if callable(fn):
                _save_plot(plot_dir / filename, lambda fn=fn: fn(x))
        if len(x["eye_samples"]):
            fn = getattr(ep, "plot_pupil_timeseries", None)
            if callable(fn):
                _save_plot(plot_dir / "pupil.png", lambda: fn(x))
    return str(p.resolve())


def report_processirt(*args, **kwargs):
    """Compatibility alias for :func:`report_eye_dataset`."""
    return report_eye_dataset(*args, **kwargs)


def as_eye_biometrics(x, mapping=None, time_unit="seconds", **kwargs):
    """Coerce an EyeDataset/DataFrame to the canonical biometrics table."""
    if is_eye_dataset(x):
        return x["biometrics"].copy()
    if not isinstance(x, pd.DataFrame):
        raise TypeError("`x` must be an EyeDataset or pandas DataFrame.")
    if mapping is None:
        raise EyeProcessValidationError("A biometric channel mapping is required.")
    mapping = dict(mapping)
    dummy = x.copy()
    if ".eye_x" not in dummy:
        dummy[".eye_x"] = np.nan
    if ".eye_y" not in dummy:
        dummy[".eye_y"] = np.nan
    mapping.setdefault("x", ".eye_x")
    mapping.setdefault("y", ".eye_y")
    return read_eye_generic(
        dummy,
        mapping=mapping,
        time_unit=time_unit,
        **kwargs,
    )["biometrics"].copy()


# R/017-format-validation.R ------------------------------------------------


@dataclass(slots=True)
class EyeFormatValidationSpec:
    require_gaze: bool = True
    require_native_time: bool = True
    require_coordinate_space: bool = True
    require_provenance: bool = True
    require_raw_retention: bool = False
    run_roundtrip: bool = True
    strict: bool = False
    min_detection_confidence: float = 0.55
    numeric_tolerance: float = 1e-8


@dataclass
class EyeRoundtripValidation:
    status: str
    comparison: pd.DataFrame
    original_fingerprint: pd.DataFrame
    restored_fingerprint: pd.DataFrame
    path: str
    numeric_tolerance: float
    eyeprocess_class: str = field(default="eye_roundtrip_validation", init=False)


@dataclass
class EyeFormatValidation:
    case_id: str
    path: str
    vendor: str
    status: str
    started: str
    completed: str
    spec: EyeFormatValidationSpec
    source: pd.DataFrame
    detection: pd.DataFrame
    adapter_issues: pd.DataFrame
    checks: pd.DataFrame
    validation: pd.DataFrame
    coverage: pd.DataFrame
    preservation: pd.DataFrame
    audits: dict[str, pd.DataFrame]
    roundtrip: EyeRoundtripValidation | None
    import_error: Any
    dataset: EyeDataset | None
    eyeprocess_class: str = field(default="eye_format_validation", init=False)


@dataclass
class EyeCorpusValidation:
    manifest: pd.DataFrame
    results: dict[str, EyeFormatValidation]
    summary: pd.DataFrame
    status: str
    completed: str
    eyeprocess_class: str = field(default="eye_corpus_validation", init=False)


def _flag(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise EyeProcessValidationError(f"`{name}` must be boolean.")
    return bool(value)


def format_validation_spec(
    min_detection_confidence=0.55,
    require_gaze=True,
    require_native_time=True,
    require_coordinate_space=True,
    require_provenance=True,
    require_raw_retention=False,
    run_roundtrip=True,
    numeric_tolerance=1e-8,
    strict=False,
):
    """Construct the frozen-R empirical source-format validation specification."""
    confidence = float(min_detection_confidence)
    tolerance = float(numeric_tolerance)
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise EyeProcessValidationError("`min_detection_confidence` must be between zero and one.")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise EyeProcessValidationError("`numeric_tolerance` must be a non-negative number.")
    return EyeFormatValidationSpec(
        require_gaze=_flag(require_gaze, "require_gaze"),
        require_native_time=_flag(require_native_time, "require_native_time"),
        require_coordinate_space=_flag(require_coordinate_space, "require_coordinate_space"),
        require_provenance=_flag(require_provenance, "require_provenance"),
        require_raw_retention=_flag(require_raw_retention, "require_raw_retention"),
        run_roundtrip=_flag(run_roundtrip, "run_roundtrip"),
        strict=_flag(strict, "strict"),
        min_detection_confidence=confidence,
        numeric_tolerance=tolerance,
    )


def eye_format_profiles():
    """Return the frozen 12-row source-format support/validation profile."""
    rows = [
        (
            "gazepoint_analysis",
            "Gazepoint",
            "gazepoint",
            "Analysis sample export",
            "file_or_folder",
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            "seconds",
            "display_normalized_top_left",
            "synthetic_fixture",
            "First-class adapter; native columns and out-of-range normalized coordinates are retained.",
        ),
        (
            "gazepoint_fixations",
            "Gazepoint",
            "gazepoint",
            "Analysis fixation export",
            "file",
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            "seconds",
            "display_normalized_top_left",
            "synthetic_fixture",
            "Vendor-derived fixations remain labelled as vendor-derived.",
        ),
        (
            "gazepoint_aoi",
            "Gazepoint",
            "gazepoint",
            "AOI statistics",
            "file",
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            "seconds",
            "display_normalized_top_left",
            "declared",
            "AOI summaries are retained separately from sample-level AOI assignment.",
        ),
        (
            "gazepoint_biometrics",
            "Gazepoint Biometrics",
            "gazepoint",
            "Combined or paired biometrics",
            "file_or_folder",
            True,
            True,
            True,
            False,
            True,
            False,
            True,
            False,
            "seconds",
            "display_normalized_top_left",
            "synthetic_fixture",
            "Supports pupil, EDA/GSR, heart-rate, IBI, and engagement-style channels when identifiable.",
        ),
        (
            "tobii_pro_lab",
            "Tobii",
            "tobii",
            "Pro Lab tabular export",
            "file",
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            "microseconds",
            "display_pixels_top_left",
            "synthetic_fixture",
            "Handles heterogeneous gaze/event rows and separate eye validity.",
        ),
        (
            "pupillabs_neon",
            "Pupil Labs",
            "pupillabs",
            "Neon folder export",
            "folder_or_gaze_file",
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            "nanoseconds",
            "world_camera_pixels",
            "synthetic_fixture",
            "Uses gaze.csv plus optional fixation, event, and 3D eye-state companions.",
        ),
        (
            "pupillabs_core",
            "Pupil Labs",
            "pupillabs",
            "Core Player folder export",
            "folder_or_gaze_file",
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            "seconds",
            "surface_normalized_bottom_left",
            "synthetic_fixture",
            "Uses gaze_positions.csv plus optional pupil and fixation companions.",
        ),
        (
            "eyelink_asc",
            "SR Research",
            "eyelink",
            "ASC event stream",
            "file",
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            "milliseconds",
            "display_pixels_top_left",
            "synthetic_fixture",
            "Parses line-oriented sample, message, fixation, saccade, and blink records.",
        ),
        (
            "eyelink_data_viewer",
            "SR Research",
            "eyelink",
            "Data Viewer report",
            "file",
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            "vendor_defined",
            "vendor_defined",
            "declared",
            "Imports delimited reports; available fields depend on the report configuration.",
        ),
        (
            "eyelink_edf",
            "SR Research",
            "eyelink",
            "EDF binary via EDF2ASC",
            "binary_file",
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            "milliseconds_after_conversion",
            "display_pixels_top_left_after_conversion",
            "declared",
            "Requires a locally available EDF2ASC converter; the package does not decode EDF directly.",
        ),
        (
            "smi_begaze_text",
            "SMI",
            "smi",
            "BeGaze textual export",
            "file",
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            "vendor_defined",
            "vendor_defined",
            "synthetic_fixture",
            "Supports textual BeGaze exports; proprietary IDF is not decoded directly.",
        ),
        (
            "generic_delimited",
            "Generic",
            "generic",
            "Mapped CSV/TSV/text",
            "file",
            False,
            True,
            True,
            False,
            True,
            True,
            True,
            False,
            "user_declared",
            "user_declared",
            "generic_mapping",
            "Requires an explicit or inferred mapping and declared units.",
        ),
    ]
    columns = [
        "format_id",
        "vendor",
        "adapter",
        "export_family",
        "input_kind",
        "dedicated_adapter",
        "gaze",
        "pupil",
        "episodes",
        "events",
        "aoi",
        "biometrics",
        "calibration",
        "default_time_unit",
        "default_coordinate_space",
        "validation_level",
        "notes",
    ]
    return pd.DataFrame(rows, columns=columns)


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def format_compatibility_matrix(validation=None):
    """Return declared format capabilities plus empirical corpus tallies."""
    out = eye_format_profiles()
    for col in [
        "empirical_cases",
        "empirical_passes",
        "empirical_warnings",
        "empirical_failures",
    ]:
        out[col] = 0
    if isinstance(validation, EyeCorpusValidation):
        summary = validation.summary
        for _, row in summary.iterrows():
            target = None
            if "format_family" in summary and pd.notna(row.get("format_family")):
                key = _normalize_key(row.get("format_family"))
                hit = out[
                    out["format_id"].map(_normalize_key).eq(key) | out["export_family"].map(_normalize_key).eq(key)
                ]
                if not hit.empty:
                    target = hit.index[0]
            if target is None:
                key = _normalize_key(row.get("vendor"))
                hit = out[out["adapter"].map(_normalize_key).eq(key)]
                if not hit.empty:
                    target = hit.index[0]
            if target is None:
                continue
            out.loc[target, "empirical_cases"] += 1
            status = row.get("status")
            if status == "pass":
                out.loc[target, "empirical_passes"] += 1
            elif status == "warning":
                out.loc[target, "empirical_warnings"] += 1
            elif status == "fail":
                out.loc[target, "empirical_failures"] += 1
    return out


def _source_delimiter(path: Path):
    try:
        first = path.open("r", encoding="utf-8", errors="replace").readline()
    except OSError:
        return pd.NA
    candidates = ["\t", ",", ";", "|"]
    counts = [first.count(item) for item in candidates]
    if not any(counts):
        return pd.NA
    return candidates[int(np.argmax(counts))]


def inspect_eye_source(path, recursive=True, inspect_rows=10, include_hash=True):
    """Inspect file/folder structure, delimiters, columns, hashes, and format detection."""
    p = Path(path).expanduser()
    if not p.exists():
        raise EyeProcessValidationError(f"Path does not exist: {path}")
    recursive = _flag(recursive, "recursive")
    include_hash = _flag(include_hash, "include_hash")
    inspect_rows = int(inspect_rows)
    if inspect_rows < 1:
        raise EyeProcessValidationError("`inspect_rows` must be a positive integer.")
    root = p if p.is_dir() else p.parent
    files = [q for q in (p.rglob("*") if recursive else p.glob("*")) if q.is_file()] if p.is_dir() else [p]
    rows = []
    for f in files:
        ext = f.suffix.lower().lstrip(".")
        tabular = ext in {"csv", "tsv", "txt", "asc"}
        sample = None
        if tabular:
            try:
                sample = _read_delimited(f, nrows=inspect_rows)
            except Exception:
                sample = None
        try:
            detection = detect_eye_format(f)
        except Exception:
            detection = pd.DataFrame()
        stat = f.stat()
        rows.append(
            {
                "source_path": str(f.resolve()),
                "relative_path": str(f.resolve().relative_to(root.resolve())) if f != root else f.name,
                "file_name": f.name,
                "extension": ext,
                "size_bytes": float(stat.st_size),
                "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "md5": hashlib.md5(f.read_bytes()).hexdigest() if include_hash else pd.NA,
                "tabular": tabular,
                "readable": sample is not None if tabular else True,
                "delimiter": _source_delimiter(f) if tabular else pd.NA,
                "inspected_rows": 0 if sample is None else len(sample),
                "n_columns": pd.NA if sample is None else sample.shape[1],
                "columns": pd.NA if sample is None else "|".join(map(str, sample.columns)),
                "detected_format": pd.NA if detection.empty else detection.iloc[0]["format"],
                "detection_confidence": np.nan if detection.empty else float(detection.iloc[0]["confidence"]),
            }
        )
    return pd.DataFrame(rows)


_CRITICAL_FIELDS = {
    "recordings": ["recording_id", "vendor"],
    "streams": ["stream_id", "recording_id", "stream_type", "timestamp_unit"],
    "gaze_samples": [
        "recording_id",
        "stream_id",
        "sample_id",
        "timestamp_native",
        "timestamp_seconds",
        "gaze_x",
        "gaze_y",
        "coordinate_space_id",
    ],
    "eye_samples": [
        "recording_id",
        "sample_id",
        "timestamp_native",
        "timestamp_seconds",
        "eye",
        "pupil_diameter",
        "pupil_unit",
    ],
    "episodes": ["episode_id", "recording_id", "episode_type", "start_time", "end_time"],
    "events": ["event_id", "recording_id", "timestamp_seconds", "event_type", "event_name"],
    "intervals": ["interval_id", "recording_id", "interval_type", "start_time", "end_time"],
    "responses": ["response_id", "recording_id", "trial_id", "item_id", "response"],
    "coordinate_spaces": ["coordinate_space_id", "space_type", "origin", "x_unit", "y_unit"],
    "biometrics": ["recording_id", "stream_id", "timestamp_seconds", "channel", "value", "unit"],
    "provenance": ["provenance_id", "timestamp", "action", "component"],
}


def _nonmissing(series: pd.Series) -> pd.Series:
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        return series.map(
            lambda v: (
                False if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)) else bool(str(v).strip())
            )
        )
    return series.notna()


def schema_coverage(x, require_gaze=True):
    """Audit population of every canonical field with frozen critical-field semantics."""
    _assert_eye_dataset(x)
    require_gaze = _flag(require_gaze, "require_gaze")
    rows = []
    for table in canonical_table_names():
        d = x[table]
        for col in schema_table(table):
            present = col in d
            populated = int(_nonmissing(d[col]).sum()) if present and len(d) else 0
            fraction = populated / len(d) if present and len(d) else np.nan
            critical = col in _CRITICAL_FIELDS.get(table, [])
            if not present:
                status = "fail" if critical else "warning"
            elif not len(d):
                status = (
                    "fail"
                    if critical and (table == "recordings" or (table == "gaze_samples" and require_gaze))
                    else "not_applicable"
                )
            elif critical and populated == 0:
                status = "fail"
            elif critical and fraction < 1:
                status = "warning"
            elif populated == 0:
                status = "empty"
            else:
                status = "pass"
            rows.append(
                {
                    "table": table,
                    "field": col,
                    "rows": len(d),
                    "present": present,
                    "populated_n": populated,
                    "populated_fraction": fraction,
                    "critical": critical,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def schema_coverage_summary(x):
    """Summarize detailed canonical schema coverage by table."""
    coverage = schema_coverage(x) if is_eye_dataset(x) else x
    if not isinstance(coverage, pd.DataFrame):
        raise TypeError("`x` must be an EyeDataset or pandas DataFrame.")
    rows = []
    for table, d in coverage.groupby("table", sort=False):
        rows.append(
            {
                "table": table,
                "rows": int(pd.to_numeric(d["rows"], errors="coerce").max()),
                "canonical_fields": len(d),
                "populated_fields": int((d["populated_n"] > 0).sum()),
                "critical_fields": int(d["critical"].astype(bool).sum()),
                "critical_pass": int((d["critical"].astype(bool) & d["status"].eq("pass")).sum()),
                "failures": int(d["status"].eq("fail").sum()),
                "warnings": int(d["status"].eq("warning").sum()),
                "populated_fraction": float((d["populated_n"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _usable_text(values: Any) -> pd.Series:
    s = pd.Series(values, dtype="string")
    z = s.str.strip()
    return z.notna() & z.ne("") & ~z.isin(["NA", "<data.frame>", "<redacted>"])


def _safe_numeric(values: Any) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float, copy=True)


def source_preservation_audit(x, require_raw=False):
    """Audit provenance, native/normalized time, coordinate, unit, metadata, and raw retention."""
    _assert_eye_dataset(x)
    require_raw = _flag(require_raw, "require_raw")
    gaze, eye, bio = x["gaze_samples"], x["eye_samples"], x["biometrics"]
    native = pd.Series(
        [
            *gaze["timestamp_native"].tolist(),
            *eye["timestamp_native"].tolist(),
            *bio["timestamp_native"].tolist(),
        ],
        dtype="object",
    )
    normalized = pd.Series(
        [
            *gaze["timestamp_seconds"].tolist(),
            *eye["timestamp_seconds"].tolist(),
            *bio["timestamp_seconds"].tolist(),
        ],
        dtype="object",
    )
    pupil = pd.to_numeric(eye["pupil_diameter"], errors="coerce")
    pupil_units = eye.loc[np.isfinite(pupil), "pupil_unit"].dropna().astype(str).unique()
    stream_units = x["streams"]["timestamp_unit"].dropna().astype(str)
    checks = [
        (
            "source_provenance",
            len(x["provenance"]) > 0,
            len(x["provenance"]),
            "At least one provenance row is retained.",
        ),
        (
            "source_file_reference",
            bool(_usable_text(x["provenance"]["source_files"]).any()),
            int(_usable_text(x["provenance"]["source_files"]).sum()),
            "Source-file references are recorded.",
        ),
        (
            "source_file_hash",
            bool(_usable_text(x["provenance"]["file_hashes"]).any()),
            int(_usable_text(x["provenance"]["file_hashes"]).sum()),
            "Source-file hashes are recorded when files are available.",
        ),
        (
            "native_timestamps",
            bool(np.isfinite(_safe_numeric(native)).any()),
            int(np.isfinite(_safe_numeric(native)).sum()),
            "Native timestamps are retained.",
        ),
        (
            "normalized_timestamps",
            bool(np.isfinite(_safe_numeric(normalized)).any()),
            int(np.isfinite(_safe_numeric(normalized)).sum()),
            "Seconds-based timestamps are available.",
        ),
        (
            "coordinate_registry",
            len(x["coordinate_spaces"]) > 0,
            len(x["coordinate_spaces"]),
            "Coordinate spaces are explicit.",
        ),
        (
            "stream_units",
            len(x["streams"]) == 0 or bool(stream_units.str.len().gt(0).any()),
            int(stream_units.str.len().gt(0).sum()),
            "Stream timestamp units are explicit.",
        ),
        (
            "pupil_units",
            len(eye) == 0 or not np.isfinite(pupil).any() or len(pupil_units) > 0,
            len(pupil_units),
            "Pupil units are explicit when pupil values exist.",
        ),
        (
            "vendor_metadata",
            bool(x.vendor_metadata),
            len(x.vendor_metadata) if hasattr(x.vendor_metadata, "__len__") else 0,
            "Vendor-specific metadata are retained.",
        ),
        (
            "raw_retention",
            not require_raw or bool(x.raw),
            len(x.raw) if hasattr(x.raw, "__len__") else 0,
            "Raw source data are required for this validation."
            if require_raw
            else "Raw retention is optional for this validation.",
        ),
    ]
    return pd.DataFrame(
        [
            {"check": name, "status": "pass" if ok else "fail", "value": float(value), "message": msg}
            for name, ok, value, msg in checks
        ]
    )


def _stable_cell(value: Any) -> str:
    if value is None or value is pd.NA:
        return "<NA>"
    try:
        if pd.isna(value):
            return "<NA>"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if bool(value) else "FALSE"
    if isinstance(value, (float, np.floating)):
        if math.isinf(float(value)):
            return "Inf" if float(value) > 0 else "-Inf"
        return format(float(value), ".17g")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _stable_table(d: pd.DataFrame, ignore_columns=()):
    keep = [c for c in d.columns if c not in set(ignore_columns)]
    if not keep:
        return pd.DataFrame()
    out = pd.DataFrame({c: d[c].map(_stable_cell) for c in keep})
    if len(out):
        order = out.astype(str).agg("\r".join, axis=1).sort_values(kind="stable").index
        out = out.loc[order].reset_index(drop=True)
    return out


def _table_md5(d: pd.DataFrame) -> str:
    text = d.to_csv(index=False, sep="\t", lineterminator="\n", na_rep="<NA>")
    return hashlib.md5(text.encode("utf-8")).hexdigest()


_VOLATILE = {
    "provenance": ["provenance_id", "timestamp"],
    "quality": ["quality_id", "computed_at"],
    "features": ["feature_id", "derived_at"],
}


def fingerprint_eye_dataset(x, tables=None, ignore_volatile=True):
    """Compute deterministic per-table content fingerprints."""
    _assert_eye_dataset(x)
    if tables is None:
        tables = canonical_table_names()
    tables = [t for t in tables if t in canonical_table_names()]
    rows = []
    for table in tables:
        ignore = _VOLATILE.get(table, []) if ignore_volatile else []
        stable = _stable_table(x[table], ignore)
        rows.append(
            {
                "table": table,
                "rows": len(x[table]),
                "columns": x[table].shape[1],
                "compared_columns": stable.shape[1],
                "md5": _table_md5(stable),
            }
        )
    return pd.DataFrame(rows)


_KEY_FIELDS = {
    "recordings": ["recording_id"],
    "streams": ["stream_id"],
    "gaze_samples": ["sample_id"],
    "eye_samples": ["recording_id", "sample_id", "eye"],
    "episodes": ["episode_id"],
    "events": ["event_id"],
    "intervals": ["interval_id"],
    "responses": ["response_id"],
    "coordinate_spaces": ["coordinate_space_id"],
    "aoi_definitions": ["aoi_id"],
    "aoi_geometry": ["aoi_id", "valid_from", "valid_to", "frame_id"],
    "biometrics": ["recording_id", "stream_id", "timestamp_native", "channel", "trial_id"],
    "calibrations": ["calibration_id"],
    "features": ["feature_id"],
    "quality": ["quality_id"],
    "provenance": ["provenance_id"],
}


def _unsorted_stable_keys(d: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    columns = list(columns)
    if not columns:
        return pd.Series([""] * len(d), index=d.index, dtype="string")
    stable = pd.DataFrame({c: d[c].map(_stable_cell) for c in columns}, index=d.index)
    return stable.astype(str).agg("\r".join, axis=1)


def _sorted_table(d: pd.DataFrame, table: str, common: list[str]) -> pd.DataFrame:
    if not len(d):
        return d.reset_index(drop=True)
    keys = [k for k in _KEY_FIELDS.get(table, []) if k in common]
    if keys:
        key = _unsorted_stable_keys(d, keys)
        if not key.duplicated().any():
            order = np.argsort(key.to_numpy(), kind="stable")
            return d.iloc[order].reset_index(drop=True)
    key = _unsorted_stable_keys(d, common)
    order = np.argsort(key.to_numpy(), kind="stable")
    return d.iloc[order].reset_index(drop=True)


def _compare_columns(a: pd.Series, b: pd.Series, tolerance: float):
    if len(a) != len(b):
        return max(len(a), len(b)), math.inf
    na_a, na_b = a.isna(), b.isna()
    differences = int((na_a != na_b).sum())
    keep = ~(na_a | na_b)
    if not keep.any():
        return differences, 0.0
    aa_num = pd.to_numeric(a[keep], errors="coerce")
    bb_num = pd.to_numeric(b[keep], errors="coerce")
    numeric_mask = aa_num.notna() & bb_num.notna()
    nonnumeric_mask = ~numeric_mask
    max_delta = 0.0
    if numeric_mask.any():
        av = aa_num[numeric_mask].to_numpy(dtype=float)
        bv = bb_num[numeric_mask].to_numpy(dtype=float)
        same_inf = np.isinf(av) & np.isinf(bv) & (np.sign(av) == np.sign(bv))
        finite = np.isfinite(av) & np.isfinite(bv)
        delta = np.full(len(av), np.inf)
        delta[same_inf] = 0.0
        delta[finite] = np.abs(av[finite] - bv[finite])
        differences += int((delta > tolerance).sum())
        finite_delta = delta[np.isfinite(delta)]
        max_delta = float(finite_delta.max()) if finite_delta.size else (math.inf if len(delta) else 0.0)
    if nonnumeric_mask.any():
        av = a[keep][nonnumeric_mask].map(_stable_cell)
        bv = b[keep][nonnumeric_mask].map(_stable_cell)
        differences += int((av.to_numpy() != bv.to_numpy()).sum())
    return differences, max_delta


def compare_eye_datasets(
    x,
    y,
    tables=None,
    numeric_tolerance=1e-8,
    ignore_volatile=True,
):
    """Compare two canonical datasets table-by-table with numeric tolerance."""
    _assert_eye_dataset(x)
    _assert_eye_dataset(y)
    tolerance = float(numeric_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise EyeProcessValidationError("`numeric_tolerance` must be a non-negative number.")
    if tables is None:
        tables = canonical_table_names()
    tables = [t for t in tables if t in canonical_table_names()]
    rows = []
    for table in tables:
        dx, dy = x[table].copy(), y[table].copy()
        missing = [c for c in dx.columns if c not in dy.columns]
        added = [c for c in dy.columns if c not in dx.columns]
        common = [c for c in dx.columns if c in dy.columns]
        if ignore_volatile:
            common = [c for c in common if c not in _VOLATILE.get(table, [])]
        if len(dx) == len(dy) and len(dx):
            dx = _sorted_table(dx, table, common)
            dy = _sorted_table(dy, table, common)
        diffs, numeric_deltas = 0, []
        if len(dx) != len(dy):
            diffs += abs(len(dx) - len(dy))
        for col in common:
            n, delta = _compare_columns(dx[col], dy[col], tolerance)
            diffs += n
            if np.isfinite(delta):
                numeric_deltas.append(delta)
        status = "pass" if len(dx) == len(dy) and not missing and not added and diffs == 0 else "fail"
        rows.append(
            {
                "table": table,
                "rows_original": len(dx),
                "rows_restored": len(dy),
                "columns_original": dx.shape[1],
                "columns_restored": dy.shape[1],
                "missing_columns": "|".join(missing),
                "added_columns": "|".join(added),
                "differing_cells": int(diffs),
                "max_numeric_difference": max(numeric_deltas) if numeric_deltas else np.nan,
                "content_equal": diffs == 0,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def roundtrip_eye_dataset(
    x,
    path=None,
    include_raw=False,
    numeric_tolerance=1e-8,
    cleanup=True,
):
    """Write/read the canonical folder representation and compare all tables."""
    _assert_eye_dataset(x)
    include_raw = _flag(include_raw, "include_raw")
    cleanup = _flag(cleanup, "cleanup")
    own_path = path is None
    path = Path(tempfile.mkdtemp(prefix="eyeprocess-roundtrip-")) if path is None else Path(path)
    try:
        write_eye_dataset(x, path, format="folder", include_raw=include_raw, overwrite=True)
        restored = read_eye_dataset(path, validate=True)
        comparison = compare_eye_datasets(x, restored, numeric_tolerance=numeric_tolerance)
        return EyeRoundtripValidation(
            status="pass" if comparison["status"].eq("pass").all() else "fail",
            comparison=comparison,
            original_fingerprint=fingerprint_eye_dataset(x),
            restored_fingerprint=fingerprint_eye_dataset(restored),
            path=str(path.resolve()),
            numeric_tolerance=float(numeric_tolerance),
        )
    finally:
        if cleanup:
            shutil.rmtree(path, ignore_errors=True)
        elif own_path:
            pass


def _validation_check(check, status, value=np.nan, message=pd.NA):
    return pd.DataFrame([{"check": str(check), "status": str(status), "value": value, "message": message}])


def _safe_audit(fn, x):
    try:
        return fn(x)
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "severity": "warning",
                    "code": "audit_failed",
                    "table": pd.NA,
                    "field": pd.NA,
                    "message": str(exc),
                }
            ]
        )


def _empty_vendor_issues():
    return pd.DataFrame(columns=["severity", "code", "file", "message"])


def _vendor_issue(severity, code, file, message):
    return pd.DataFrame(
        [
            {
                "severity": severity,
                "code": code,
                "file": str(Path(file).expanduser().resolve()),
                "message": message,
            }
        ]
    )


def _pupil_labs_format(path) -> str:
    p = Path(path)
    if p.is_dir():
        if (p / "gaze.csv").exists():
            return "neon"
        if (p / "gaze_positions.csv").exists():
            return "core"
    elif p.name.lower() == "gaze.csv":
        return "neon"
    elif p.name.lower() == "gaze_positions.csv":
        return "core"
    return "unknown"


def _smi_confidence(path) -> float:
    p = Path(path)
    if not p.is_file():
        return 0.0
    try:
        text = "\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[:25]).lower()
    except OSError:
        return 0.0
    terms = ["begaze", "smi", "por x", "por y", "pupil diameter", "event info", "tracking ratio"]
    return min(1.0, sum(term in text for term in terms) / 3.0)


def validate_tobii_export(path):
    """Validate basic Tobii Pro Lab delimited-export structure."""
    p = Path(path)
    if p.is_dir():
        return _vendor_issue("error", "expected_file", p, "Tobii Pro Lab validation expects a delimited export file.")
    try:
        d = _read_delimited(p, nrows=5)
    except Exception:
        return _vendor_issue(
            "error", "unreadable_export", p, "The Tobii export could not be read as a delimited table."
        )
    names = [str(c).lower() for c in d.columns]
    has_time = any(re.search(r"recording timestamp|system_time_stamp|device_time_stamp|timestamp", n) for n in names)
    has_x = any(re.search(r"gaze point.*x|gaze2d x|display_area_x", n) for n in names)
    has_y = any(re.search(r"gaze point.*y|gaze2d y|display_area_y", n) for n in names)
    frames = []
    if not has_time:
        frames.append(
            _vendor_issue("error", "missing_timestamp", p, "No supported Tobii timestamp column was identified.")
        )
    if not (has_x and has_y):
        frames.append(
            _vendor_issue(
                "error", "missing_gaze_coordinates", p, "No supported Tobii gaze-coordinate pair was identified."
            )
        )
    if not any(re.search(r"validity|valid", n) for n in names):
        frames.append(
            _vendor_issue("warning", "missing_validity", p, "No explicit Tobii gaze-validity field was identified.")
        )
    return pd.concat(frames, ignore_index=True) if frames else _empty_vendor_issues()


def validate_pupillabs_export(path):
    """Validate Pupil Labs Neon/Core source shape without claiming an unavailable adapter."""
    p = Path(path)
    fmt = _pupil_labs_format(p)
    if fmt == "unknown":
        return _vendor_issue(
            "error", "unknown_pupil_format", p, "The source is not recognizable as Pupil Labs Neon or Core."
        )
    required = "gaze.csv" if fmt == "neon" else "gaze_positions.csv"
    if p.is_dir() and not (p / required).exists():
        return _vendor_issue("error", "missing_gaze_file", p, f"Required `{required}` is absent.")
    return _empty_vendor_issues()


def validate_eyelink_export(path):
    """Validate EyeLink ASC/Data Viewer shape and explicitly flag EDF conversion."""
    p = Path(path)
    if p.is_dir():
        return _vendor_issue(
            "error", "expected_file", p, "EyeLink validation expects EDF, ASC, or a Data Viewer report file."
        )
    if p.suffix.lower() == ".edf":
        return _vendor_issue(
            "warning",
            "external_converter_required",
            p,
            "EDF requires a locally installed EDF2ASC converter before text parsing.",
        )
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[:50]
    except OSError:
        lines = []
    if not lines:
        return _vendor_issue("error", "unreadable_export", p, "The EyeLink source could not be read.")
    if p.suffix.lower() == ".asc":
        known_tokens = {
            "MSG",
            "START",
            "END",
            "EFIX",
            "ESACC",
            "EBLINK",
            "SFIX",
            "SSACC",
            "SBLINK",
            "BUTTON",
            "INPUT",
            "PRESCALER",
            "EVENTS",
            "SAMPLES",
        }
        tokens = [line.strip().split()[0] if line.strip() else "" for line in lines]
        if not any(token in known_tokens or token.isdigit() for token in tokens):
            return _vendor_issue(
                "error", "unknown_asc_records", p, "No recognized EyeLink ASC record types were found."
            )
    return _empty_vendor_issues()


def validate_smi_export(path):
    """Validate SMI/BeGaze textual exports; proprietary IDF remains unsupported."""
    p = Path(path)
    if p.is_dir():
        return _vendor_issue("error", "expected_file", p, "SMI validation expects a textual BeGaze export file.")
    if p.suffix.lower() == ".idf":
        return _vendor_issue(
            "error", "proprietary_idf", p, "Direct IDF decoding is not supported; export text from BeGaze first."
        )
    if _smi_confidence(p) < 0.5:
        return _vendor_issue(
            "warning", "low_smi_confidence", p, "The textual export weakly matches known SMI/BeGaze fields."
        )
    return _empty_vendor_issues()


def validate_generic_export(path):
    """Validate readability and inferability of a generic mapped delimited export."""
    p = Path(path)
    if p.is_dir():
        return _vendor_issue("error", "expected_file", p, "Generic mapping expects one delimited file.")
    try:
        d = _read_delimited(p, nrows=10)
    except Exception:
        return _vendor_issue(
            "error", "unreadable_export", p, "The generic export could not be read as a delimited table."
        )
    mapping = infer_eye_mapping(d)
    missing = [name for name in ["timestamp", "x", "y"] if name not in mapping]
    if missing:
        return _vendor_issue(
            "warning", "mapping_required", p, "Explicit mappings are required for: " + ", ".join(missing) + "."
        )
    return _empty_vendor_issues()


def _format_validation_summary(x: EyeFormatValidation):
    return pd.DataFrame(
        [
            {
                "case_id": x.case_id,
                "path": x.path,
                "vendor": x.vendor,
                "status": x.status,
                "files": len(x.source),
                "detection_confidence": 0.0 if x.detection.empty else float(x.detection.iloc[0]["confidence"]),
                "imported": bool((x.checks["check"].eq("import") & x.checks["status"].eq("pass")).any()),
                "validation_errors": int(x.validation["severity"].eq("error").sum())
                if "severity" in x.validation
                else pd.NA,
                "validation_warnings": int(x.validation["severity"].eq("warning").sum())
                if "severity" in x.validation
                else pd.NA,
                "roundtrip": "not_run" if x.roundtrip is None else x.roundtrip.status,
            }
        ]
    )


def validate_eye_source(
    path,
    vendor="auto",
    spec=None,
    import_args=None,
    retain_dataset=False,
    case_id=None,
):
    """Run frozen-R source inspection, import, canonical audits, and round-trip checks."""
    spec = format_validation_spec() if spec is None else spec
    if not isinstance(spec, EyeFormatValidationSpec):
        raise EyeProcessValidationError("`spec` must be created by `format_validation_spec()`.")
    import_args = {} if import_args is None else import_args
    if not isinstance(import_args, Mapping):
        raise EyeProcessValidationError("`import_args` must be a mapping.")
    retain_dataset = _flag(retain_dataset, "retain_dataset")
    p = Path(path).expanduser()
    if not p.exists():
        raise EyeProcessValidationError(f"Path does not exist: {path}")
    case_id = p.name if case_id is None else str(case_id)
    started = _now_utc()
    source = inspect_eye_source(p)
    try:
        detection = detect_eye_format(p)
    except Exception:
        detection = pd.DataFrame(columns=["format", "confidence", "priority"])
    selected_vendor = str(vendor)
    if selected_vendor == "auto" and not detection.empty:
        selected_vendor = str(detection.iloc[0]["format"])
    if not detection.empty and selected_vendor in set(detection["format"].astype(str)):
        confidence = float(detection.loc[detection["format"].astype(str).eq(selected_vendor), "confidence"].iloc[0])
    else:
        confidence = 0.0

    adapter_issues = _empty_vendor_issues()
    adapter = _ADAPTERS.get(selected_vendor)
    if adapter is not None and callable(adapter.get("validate")):
        try:
            adapter_issues = adapter["validate"](p)
        except Exception as exc:
            adapter_issues = _vendor_issue("error", "adapter_validation_error", p, str(exc))

    try:
        imported = read_eye_export(p, vendor=selected_vendor, **dict(import_args))
        dataset = imported if is_eye_dataset(imported) else None
        import_error = pd.NA if dataset is not None else "Reader did not return an EyeDataset."
    except Exception as exc:
        dataset = None
        import_error = str(exc)

    validation = validate_eye_dataset(dataset, strict=spec.strict) if dataset is not None else pd.DataFrame()
    coverage = schema_coverage(dataset, require_gaze=spec.require_gaze) if dataset is not None else pd.DataFrame()
    preservation = (
        source_preservation_audit(dataset, require_raw=spec.require_raw_retention)
        if dataset is not None
        else pd.DataFrame()
    )
    audits = (
        {
            "timebase": _safe_audit(audit_timebase, dataset),
            "coordinates": _safe_audit(audit_coordinate_spaces, dataset),
            "sampling_rate": _safe_audit(audit_sampling_rate, dataset),
            "signal_quality": _safe_audit(audit_signal_quality, dataset),
            "event_order": _safe_audit(audit_event_order, dataset),
        }
        if dataset is not None
        else {}
    )
    roundtrip = None
    if dataset is not None and spec.run_roundtrip:
        try:
            roundtrip = roundtrip_eye_dataset(
                dataset,
                numeric_tolerance=spec.numeric_tolerance,
            )
        except Exception:
            roundtrip = EyeRoundtripValidation(
                status="fail",
                comparison=pd.DataFrame([{"table": pd.NA, "status": "fail", "content_equal": False}]),
                original_fingerprint=pd.DataFrame(),
                restored_fingerprint=pd.DataFrame(),
                path="",
                numeric_tolerance=spec.numeric_tolerance,
            )

    known_explicit = selected_vendor in _ADAPTERS
    detection_status = (
        "pass"
        if confidence >= spec.min_detection_confidence
        else ("warning" if str(vendor) != "auto" and known_explicit else "fail")
    )
    checks = [
        _validation_check("format_detection", detection_status, confidence, f"Selected adapter: {selected_vendor}."),
        _validation_check(
            "import",
            "pass" if dataset is not None else "fail",
            1 if dataset is not None else 0,
            "Source imported to an eye_dataset." if dataset is not None else import_error,
        ),
    ]
    if dataset is not None:
        errors = int(validation["severity"].eq("error").sum()) if "severity" in validation else 0
        checks.append(
            _validation_check(
                "dataset_validation",
                "fail" if errors else ("warning" if len(validation) else "pass"),
                errors,
                f"{len(validation)} validation issue(s).",
            )
        )
        critical = coverage[coverage["critical"].astype(bool)]
        checks.append(
            _validation_check(
                "critical_schema_coverage",
                "fail"
                if critical["status"].eq("fail").any()
                else ("warning" if critical["status"].eq("warning").any() else "pass"),
                float(critical["status"].eq("pass").mean()) if len(critical) else np.nan,
                "Critical canonical fields were assessed.",
            )
        )
        gaze_ok = len(dataset["gaze_samples"]) > 0
        checks.append(
            _validation_check(
                "gaze_samples",
                "pass" if not spec.require_gaze or gaze_ok else "fail",
                len(dataset["gaze_samples"]),
                "Gaze observations are required." if spec.require_gaze else "Gaze observations are optional.",
            )
        )
        native_ok = bool(((preservation["check"].eq("native_timestamps")) & preservation["status"].eq("pass")).any())
        coord_ok = bool(((preservation["check"].eq("coordinate_registry")) & preservation["status"].eq("pass")).any())
        prov_ok = bool(((preservation["check"].eq("source_provenance")) & preservation["status"].eq("pass")).any())
        raw_ok = bool(dataset.raw)
        checks += [
            _validation_check(
                "native_time_preservation",
                "pass" if not spec.require_native_time or native_ok else "fail",
                1 if native_ok else 0,
                "Native and normalized time should remain distinguishable.",
            ),
            _validation_check(
                "coordinate_space",
                "pass" if not spec.require_coordinate_space or coord_ok else "fail",
                len(dataset["coordinate_spaces"]),
                "Coordinate semantics should be explicit.",
            ),
            _validation_check(
                "provenance",
                "pass" if not spec.require_provenance or prov_ok else "fail",
                len(dataset["provenance"]),
                "Transformations and source references should be auditable.",
            ),
            _validation_check(
                "raw_retention",
                "pass" if not spec.require_raw_retention or raw_ok else "fail",
                len(dataset.raw) if hasattr(dataset.raw, "__len__") else 0,
                "Raw source retention is required."
                if spec.require_raw_retention
                else "Raw source retention is optional.",
            ),
        ]
        if roundtrip is not None:
            checks.append(
                _validation_check(
                    "canonical_roundtrip",
                    roundtrip.status,
                    float(roundtrip.comparison["status"].eq("pass").mean()),
                    "Canonical folder export and re-import were compared table by table.",
                )
            )
    checks_df = pd.concat(checks, ignore_index=True)
    if len(adapter_issues):
        severity = adapter_issues["severity"].astype(str)
        checks_df = pd.concat(
            [
                checks_df,
                _validation_check(
                    "adapter_specific_validation",
                    "fail" if severity.eq("error").any() else ("warning" if severity.eq("warning").any() else "pass"),
                    len(adapter_issues),
                    "The selected adapter completed its source-format checks.",
                ),
            ],
            ignore_index=True,
        )
    overall = (
        "fail"
        if checks_df["status"].eq("fail").any()
        else ("warning" if checks_df["status"].eq("warning").any() else "pass")
    )
    return EyeFormatValidation(
        case_id=case_id,
        path=str(p.resolve()),
        vendor=selected_vendor,
        status=overall,
        started=started,
        completed=_now_utc(),
        spec=spec,
        source=source,
        detection=detection,
        adapter_issues=adapter_issues,
        checks=checks_df,
        validation=validation,
        coverage=coverage,
        preservation=preservation,
        audits=audits,
        roundtrip=roundtrip,
        import_error=import_error,
        dataset=dataset if retain_dataset else None,
    )


_MANIFEST_COLUMNS = [
    "case_id",
    "path",
    "vendor",
    "format_family",
    "software_version",
    "device_model",
    "expected_import",
    "require_gaze",
    "require_native_time",
    "require_coordinate_space",
    "require_provenance",
    "require_raw_retention",
    "run_roundtrip",
    "notes",
]
_FLAG_COLUMNS = [
    "expected_import",
    "require_gaze",
    "require_native_time",
    "require_coordinate_space",
    "require_provenance",
    "require_raw_retention",
    "run_roundtrip",
]


def _recycle(value: Any, n: int, name: str):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return [value] * n
    values = list(value)
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise EyeProcessValidationError(f"`{name}` must have length one or match `paths`.")
    return values


def _coerce_manifest_flag(series: pd.Series, field: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    for i, value in series.items():
        if pd.isna(value) or str(value).strip().lower() in {"", "na"}:
            continue
        if isinstance(value, (bool, np.bool_)):
            out.loc[i] = bool(value)
            continue
        if isinstance(value, (int, float, np.number)) and value in {0, 1}:
            out.loc[i] = bool(value)
            continue
        z = str(value).strip().lower()
        if z in {"true", "t", "yes", "y", "1"}:
            out.loc[i] = True
        elif z in {"false", "f", "no", "n", "0"}:
            out.loc[i] = False
        else:
            raise EyeProcessValidationError(f"Manifest field `{field}` contains invalid logical value: {value!r}.")
    return out


def _validate_manifest_rows(x, name="validation manifest"):
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"`{name}` must be a pandas DataFrame.")
    required = ["case_id", "path", "vendor"]
    missing = [c for c in required if c not in x]
    if missing:
        raise EyeProcessValidationError(f"{name} is missing: {', '.join(missing)}.")
    out = x.copy()
    for col in required:
        out[col] = out[col].astype("string").str.strip()
        if out[col].isna().any() or out[col].eq("").any():
            raise EyeProcessValidationError(f"Every validation case requires a non-empty `{col}`.")
    if out["case_id"].duplicated().any():
        duplicates = ", ".join(out.loc[out["case_id"].duplicated(False), "case_id"].unique())
        raise EyeProcessValidationError(f"Validation `case_id` values must be unique: {duplicates}.")
    for col in [c for c in _FLAG_COLUMNS if c in out]:
        out[col] = _coerce_manifest_flag(out[col], col)
    return out


def validation_manifest(
    paths,
    vendor="auto",
    format_family=pd.NA,
    software_version=pd.NA,
    device_model=pd.NA,
    expected_import=True,
    require_gaze=True,
    require_native_time=True,
    require_coordinate_space=True,
    require_provenance=True,
    require_raw_retention=False,
    run_roundtrip=True,
    case_id=None,
    notes=pd.NA,
):
    """Construct an empirical validation-corpus manifest."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not paths:
        raise EyeProcessValidationError("At least one source path is required.")
    n = len(paths)
    resolved = [str(Path(p).expanduser().resolve()) for p in paths]
    if case_id is None:
        base = [Path(p).name for p in resolved]
        counts: dict[str, int] = {}
        ids = []
        for value in base:
            counts[value] = counts.get(value, 0) + 1
            ids.append(value if counts[value] == 1 else f"{value}.{counts[value] - 1}")
    else:
        ids = _recycle(case_id, n, "case_id")
    data = {
        "case_id": ids,
        "path": resolved,
        "vendor": _recycle(vendor, n, "vendor"),
        "format_family": _recycle(format_family, n, "format_family"),
        "software_version": _recycle(software_version, n, "software_version"),
        "device_model": _recycle(device_model, n, "device_model"),
        "expected_import": _recycle(expected_import, n, "expected_import"),
        "require_gaze": _recycle(require_gaze, n, "require_gaze"),
        "require_native_time": _recycle(require_native_time, n, "require_native_time"),
        "require_coordinate_space": _recycle(require_coordinate_space, n, "require_coordinate_space"),
        "require_provenance": _recycle(require_provenance, n, "require_provenance"),
        "require_raw_retention": _recycle(require_raw_retention, n, "require_raw_retention"),
        "run_roundtrip": _recycle(run_roundtrip, n, "run_roundtrip"),
        "notes": _recycle(notes, n, "notes"),
    }
    return _validate_manifest_rows(pd.DataFrame(data))


def write_validation_manifest(x, path):
    """Write a validation manifest to CSV."""
    d = _validate_manifest_rows(x, "x")
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(p, index=False, na_rep="", lineterminator="\n")
    return str(p.resolve())


def read_validation_manifest(path):
    """Read and validate a corpus manifest, resolving relative source paths."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise EyeProcessValidationError(f"Manifest does not exist: {path}")
    d = pd.read_csv(p, dtype=object, keep_default_na=False, na_values=[""])
    d = _validate_manifest_rows(d)
    resolved = []
    for value in d["path"].astype(str):
        q = Path(value)
        resolved.append(str((q if q.is_absolute() else p.parent / q).resolve()))
    d["path"] = resolved
    return d


def init_validation_corpus(path, overwrite=False):
    """Initialize the private real-export validation-corpus template."""
    p = Path(path).expanduser()
    overwrite = _flag(overwrite, "overwrite")
    p.mkdir(parents=True, exist_ok=True)
    manifest_path = p / "validation-manifest.csv"
    readme_path = p / "README.txt"
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in _MANIFEST_COLUMNS})
    if overwrite or not manifest_path.exists():
        empty.to_csv(manifest_path, index=False, lineterminator="\n")
    if overwrite or not readme_path.exists():
        readme_path.write_text(
            "\n".join(
                [
                    "eyeprocess real-export validation corpus",
                    "",
                    "1. Create one subdirectory or add one source file per de-identified export case.",
                    "2. Add one unique row per case to validation-manifest.csv.",
                    "3. Use adapter names returned by supported_eye_formats().",
                    "4. Prefer format_id values returned by eye_format_profiles().",
                    "5. Keep this corpus outside public version control.",
                    "6. Run validate_eye_corpus() and review every warning.",
                    "7. Review anonymized bundles manually before sharing.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    return str(p.resolve())


def discover_validation_cases(path, recursive=False):
    """Discover file/directory validation cases under a corpus directory."""
    p = Path(path).expanduser()
    if not p.is_dir():
        raise EyeProcessValidationError(f"Validation corpus directory does not exist: {path}")
    recursive = _flag(recursive, "recursive")
    children = list(p.iterdir())
    cases = [q for q in children if q.name.lower() not in {"manifest.csv", "validation-manifest.csv", "readme.txt"}]
    if not cases and recursive:
        cases = [q for q in p.rglob("*") if q.is_file()]
    if not cases:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in _MANIFEST_COLUMNS})
    return validation_manifest(cases)


def validate_eye_corpus(
    manifest,
    spec=None,
    import_args=None,
    retain_datasets=False,
    stop_on_failure=False,
):
    """Validate every real-export case in a manifest/corpus."""
    spec = format_validation_spec() if spec is None else spec
    import_args = {} if import_args is None else import_args
    if isinstance(manifest, (str, Path)):
        p = Path(manifest)
        if p.is_dir():
            mf = p / "validation-manifest.csv"
            manifest = read_validation_manifest(mf) if mf.exists() else discover_validation_cases(p)
        else:
            manifest = read_validation_manifest(p)
    manifest = _validate_manifest_rows(manifest, "manifest")
    if manifest.empty:
        raise EyeProcessValidationError("The validation corpus contains no cases.")
    retain_datasets = _flag(retain_datasets, "retain_datasets")
    stop_on_failure = _flag(stop_on_failure, "stop_on_failure")
    results = {}
    summaries = []
    override_fields = [
        "require_gaze",
        "require_native_time",
        "require_coordinate_space",
        "require_provenance",
        "require_raw_retention",
        "run_roundtrip",
    ]
    for _, row in manifest.iterrows():
        case_spec = copy.deepcopy(spec)
        for field in override_fields:
            if field in manifest and pd.notna(row[field]):
                setattr(case_spec, field, bool(row[field]))
        if (
            isinstance(import_args, Mapping)
            and str(row["case_id"]) in import_args
            and isinstance(import_args[str(row["case_id"])], Mapping)
        ):
            args = dict(import_args[str(row["case_id"])])
        else:
            args = dict(import_args) if isinstance(import_args, Mapping) else {}
        result = validate_eye_source(
            row["path"],
            vendor=row["vendor"],
            spec=case_spec,
            import_args=args,
            retain_dataset=retain_datasets,
            case_id=row["case_id"],
        )
        results[str(row["case_id"])] = result
        summaries.append(_format_validation_summary(result))
    summary = pd.concat(summaries, ignore_index=True)
    extras = [
        c
        for c in [
            "format_family",
            "software_version",
            "device_model",
            "expected_import",
            "require_gaze",
            "require_native_time",
            "require_coordinate_space",
            "require_provenance",
            "require_raw_retention",
            "run_roundtrip",
            "notes",
        ]
        if c in manifest
    ]
    lookup = manifest.set_index("case_id")
    for field in extras:
        summary[field] = summary["case_id"].map(lookup[field])
    summary["validation_status"] = summary["status"]
    expected = summary.get("expected_import", pd.Series(True, index=summary.index)).fillna(True).astype(bool)
    summary["expectation_met"] = np.where(expected, summary["imported"], ~summary["imported"])
    summary["status"] = np.where(
        ~summary["expectation_met"],
        "fail",
        np.where(expected, summary["validation_status"], "pass"),
    )
    status = (
        "fail"
        if summary["status"].eq("fail").any()
        else ("warning" if summary["status"].eq("warning").any() else "pass")
    )
    out = EyeCorpusValidation(manifest, results, summary, status, _now_utc())
    if stop_on_failure and status == "fail":
        raise EyeProcessValidationError(
            f"The validation corpus contains {int(summary['status'].eq('fail').sum())} failed case(s)."
        )
    return out


def _remap_field(out, field, tables, prefix, linked_fields=None):
    linked_fields = [field] if linked_fields is None else list(linked_fields)
    values = []
    for table in tables:
        d = out[table]
        if field in d:
            values.extend(d[field].dropna().astype(str).tolist())
    values = list(dict.fromkeys(v for v in values if v))
    mapping = {value: f"{prefix}{i:07d}" for i, value in enumerate(values, 1)}
    if not mapping:
        return mapping
    for table in tables:
        d = out[table].copy()
        if d.empty:
            continue
        for target in linked_fields:
            if target in d:
                d[target] = d[target].map(lambda v: mapping.get(str(v), v) if pd.notna(v) else v)
        out[table] = d
    return mapping


def anonymize_eye_dataset(
    x,
    drop_raw=True,
    strip_source_paths=True,
    redact_free_text=True,
    anonymize_aois=True,
    retain_map=False,
    participant_prefix="P",
    recording_prefix="R",
    session_prefix="S",
):
    """Linked-identifier anonymization matching the frozen R table/linkage policy."""
    _assert_eye_dataset(x)
    for name, value in {
        "drop_raw": drop_raw,
        "strip_source_paths": strip_source_paths,
        "redact_free_text": redact_free_text,
        "anonymize_aois": anonymize_aois,
        "retain_map": retain_map,
    }.items():
        _flag(value, name)
    out = x.copy()
    maps = {}
    maps["participants"] = _remap_field(
        out, "participant_id", ["recordings", "intervals", "responses", "features"], participant_prefix
    )
    maps["recordings"] = _remap_field(out, "recording_id", canonical_table_names(), recording_prefix)
    maps["sessions"] = _remap_field(out, "session_id", ["recordings"], session_prefix)
    maps["streams"] = _remap_field(out, "stream_id", ["streams", "gaze_samples", "biometrics", "quality"], "ST")
    maps["gaze_samples"] = _remap_field(out, "sample_id", ["gaze_samples"], "GS")
    maps["eye_samples"] = _remap_field(out, "sample_id", ["eye_samples"], "ES")
    maps["episodes"] = _remap_field(out, "episode_id", ["episodes"], "EP")
    maps["events"] = _remap_field(out, "event_id", ["events"], "EV")
    maps["intervals"] = _remap_field(out, "interval_id", ["intervals"], "IN", ["interval_id", "parent_interval_id"])
    maps["responses"] = _remap_field(out, "response_id", ["responses"], "RS")
    maps["calibrations"] = _remap_field(out, "calibration_id", ["calibrations"], "CA")
    maps["features"] = _remap_field(out, "feature_id", ["features"], "FT")
    maps["quality"] = _remap_field(out, "quality_id", ["quality"], "QL")
    maps["provenance"] = _remap_field(out, "provenance_id", ["provenance"], "PV")
    maps["trials"] = _remap_field(
        out,
        "trial_id",
        [
            "gaze_samples",
            "eye_samples",
            "episodes",
            "events",
            "intervals",
            "responses",
            "biometrics",
            "features",
            "quality",
        ],
        "TR",
    )
    maps["items"] = _remap_field(out, "item_id", ["intervals", "responses", "features"], "IT")
    maps["stimuli"] = _remap_field(
        out,
        "stimulus_id",
        ["gaze_samples", "eye_samples", "episodes", "events", "intervals", "aoi_definitions", "features", "biometrics"],
        "SM",
    )
    if anonymize_aois:
        maps["aois"] = _remap_field(
            out,
            "aoi_id",
            ["episodes", "aoi_definitions", "aoi_geometry", "features"],
            "AO",
            ["aoi_id", "parent_aoi_id"],
        )
    if redact_free_text:
        maps["conditions"] = _remap_field(out, "condition_id", ["intervals"], "CO")
        if len(out["events"]):
            d = out["events"].copy()
            names = list(dict.fromkeys(d["event_name"].dropna().astype(str)))
            event_map = {v: f"event_{i:05d}" for i, v in enumerate(names, 1)}
            d["event_name"] = d["event_name"].map(lambda v: event_map.get(str(v), v) if pd.notna(v) else v)
            d["event_value"] = pd.NA
            d["native_record"] = pd.NA
            out["events"] = d
            maps["event_names"] = event_map
        if len(out["aoi_definitions"]):
            d = out["aoi_definitions"].copy()
            d["aoi_name"] = [pd.NA if pd.isna(v) else f"aoi_{i:05d}" for i, v in enumerate(d["aoi_name"], 1)]
            out["aoi_definitions"] = d
        if len(out["responses"]):
            d = out["responses"].copy()
            values = list(dict.fromkeys(d["response"].dropna().astype(str)))
            response_map = {v: f"response_{i:05d}" for i, v in enumerate(values, 1)}
            d["response"] = d["response"].map(lambda v: response_map.get(str(v), v) if pd.notna(v) else v)
            out["responses"] = d
            maps["response_values"] = response_map
        if len(out["recordings"]):
            out["recordings"]["experiment_type"] = pd.NA
        if len(out["coordinate_spaces"]):
            out["coordinate_spaces"]["reference_object"] = pd.NA
        if len(out["provenance"]):
            out["provenance"]["details"] = "<redacted>"
            mask = out["provenance"]["warnings"].notna()
            out["provenance"].loc[mask, "warnings"] = "<redacted>"
    if strip_source_paths:
        if "source_file_set" in out["recordings"]:
            out["recordings"]["source_file_set"] = pd.NA
        out["provenance"]["source_files"] = pd.NA
        out["provenance"]["file_hashes"] = pd.NA
        out.vendor_metadata = {"redacted": True} if redact_free_text else {"redacted": True}
    if drop_raw:
        out.raw = []
    out = add_provenance(
        out,
        "anonymize_dataset",
        "dataset",
        "Linked identifiers and potentially identifying source metadata were transformed.",
        source_files=(),
        reversible=False,
        warnings="Automated anonymization requires study-specific human review.",
    )
    if retain_map:
        out.anonymization_map = maps
    out.validation = validate_eye_dataset(out)
    return out


def write_format_validation_report(x, path="eyeprocess-format-validation.md"):
    """Write a Markdown report for one validation result or a validation corpus."""
    if isinstance(x, EyeFormatValidation):
        lines = [
            f"# eyeprocess source-format validation: {x.case_id}",
            "",
            f"- Status: **{x.status.upper()}**",
            f"- Adapter: `{x.vendor}`",
            f"- Source: `{x.path}`",
            f"- Completed: {x.completed}",
            "",
            "## Checks",
            "",
            _markdown_table(x.checks),
            "",
            "## Source files",
            "",
            _markdown_table(x.source) if len(x.source) else "No source files were inspected.",
            "",
            "## Format detection",
            "",
            _markdown_table(x.detection) if len(x.detection) else "No format was detected.",
            "",
            "## Adapter-specific findings",
            "",
            _markdown_table(x.adapter_issues) if len(x.adapter_issues) else "No adapter-specific findings.",
            "",
            "## Canonical validation",
            "",
            _markdown_table(x.validation) if len(x.validation) else "No canonical validation issues.",
            "",
            "## Schema coverage",
            "",
            _markdown_table(schema_coverage_summary(x.coverage))
            if len(x.coverage)
            else "No imported dataset was available.",
            "",
            "## Source preservation",
            "",
            _markdown_table(x.preservation) if len(x.preservation) else "No imported dataset was available.",
            "",
            "## Canonical round-trip",
            "",
            "Round-trip validation was not run." if x.roundtrip is None else _markdown_table(x.roundtrip.comparison),
        ]
    elif isinstance(x, EyeCorpusValidation):
        lines = [
            "# eyeprocess export-validation corpus",
            "",
            f"- Status: **{x.status.upper()}**",
            f"- Cases: {len(x.summary)}",
            f"- Completed: {x.completed}",
            "",
            "## Corpus summary",
            "",
            _markdown_table(x.summary),
            "",
            "## Compatibility matrix",
            "",
            _markdown_table(format_compatibility_matrix(x)),
        ]
        for name, result in x.results.items():
            lines += ["", f"## {name}", "", f"Status: **{result.status.upper()}**", "", _markdown_table(result.checks)]
    else:
        raise EyeProcessValidationError("`x` must be an EyeFormatValidation or EyeCorpusValidation object.")
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p.resolve())


def _redact_validation_result(x: EyeFormatValidation):
    out = copy.deepcopy(x)
    out.case_id = "case_00001"
    out.path = "<redacted>"
    if len(out.source):
        source = out.source.copy()
        for i in range(len(source)):
            ext = str(source.iloc[i].get("extension") or "")
            safe = f"source_{i + 1:05d}" + (f".{ext}" if ext else "")
            for col in ["source_path"]:
                if col in source:
                    source.loc[source.index[i], col] = "<redacted>"
            for col in ["relative_path", "file_name"]:
                if col in source:
                    source.loc[source.index[i], col] = safe
            for col in ["md5", "modified"]:
                if col in source:
                    source.loc[source.index[i], col] = pd.NA
        out.source = source
    if len(out.adapter_issues) and "file" in out.adapter_issues:
        out.adapter_issues["file"] = "<redacted>"
    if out.roundtrip is not None:
        out.roundtrip.path = "<redacted>"
    out.dataset = None
    return out


def create_validation_bundle(
    x,
    path="eyeprocess-validation-bundle.zip",
    include_dataset=True,
    anonymize=True,
    overwrite=False,
):
    """Create the frozen validation evidence bundle as a ZIP archive."""
    if not isinstance(x, EyeFormatValidation):
        raise EyeProcessValidationError("`x` must be an EyeFormatValidation object.")
    include_dataset = _flag(include_dataset, "include_dataset")
    anonymize = _flag(anonymize, "anonymize")
    overwrite = _flag(overwrite, "overwrite")
    p = Path(path).expanduser()
    if p.exists() and not overwrite:
        raise EyeProcessValidationError("Output file exists; use `overwrite=True`.")
    if include_dataset and x.dataset is None:
        raise EyeProcessValidationError("The validation did not retain its dataset. Re-run with `retain_dataset=True`.")
    temp = Path(tempfile.mkdtemp(prefix="eyeprocess-validation-bundle-"))
    try:
        evidence = _redact_validation_result(x) if anonymize else copy.deepcopy(x)
        write_format_validation_report(evidence, temp / "validation-report.md")
        for frame, filename in [
            (evidence.checks, "checks.csv"),
            (evidence.source, "source-manifest.csv"),
            (evidence.detection, "format-detection.csv"),
            (evidence.adapter_issues, "adapter-findings.csv"),
            (evidence.validation, "canonical-validation.csv"),
            (evidence.coverage, "schema-coverage.csv"),
            (evidence.preservation, "source-preservation.csv"),
        ]:
            frame.to_csv(temp / filename, index=False, na_rep="", lineterminator="\n")
        if evidence.roundtrip is not None:
            evidence.roundtrip.comparison.to_csv(
                temp / "roundtrip-comparison.csv", index=False, na_rep="", lineterminator="\n"
            )
        if include_dataset:
            dataset = anonymize_eye_dataset(x.dataset) if anonymize else x.dataset
            write_eye_dataset(dataset, temp / "canonical-dataset", include_raw=False, overwrite=True)
        (temp / "BUNDLE").write_text(
            "\n".join(
                [
                    "Bundle-Version: 1",
                    "Package: eyeprocesspy",
                    f"Case-ID: {evidence.case_id}",
                    f"Adapter: {evidence.vendor}",
                    f"Validation-Status: {x.status}",
                    f"Created: {_now_utc()}",
                    f"Anonymized: {str(anonymize).upper()}",
                    "Raw-Source-Included: FALSE",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            p.unlink()
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as archive:
            for item in sorted(temp.rglob("*")):
                if item.is_file():
                    archive.write(item, item.relative_to(temp))
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return str(p.resolve())


__all__ = [
    "anonymize_eye_dataset",
    "as_eye_biometrics",
    "compare_eye_datasets",
    "create_validation_bundle",
    "discover_validation_cases",
    "export_canonical",
    "eye_format_profiles",
    "fingerprint_eye_dataset",
    "format_compatibility_matrix",
    "format_validation_spec",
    "import_canonical",
    "init_validation_corpus",
    "inspect_eye_source",
    "read_eye_dataset",
    "read_validation_manifest",
    "report_eye_dataset",
    "report_processirt",
    "roundtrip_eye_dataset",
    "schema_coverage",
    "schema_coverage_summary",
    "source_preservation_audit",
    "validate_eye_corpus",
    "validate_eye_source",
    "validate_eyelink_export",
    "validate_generic_export",
    "validate_pupillabs_export",
    "validate_smi_export",
    "validate_tobii_export",
    "validation_manifest",
    "write_eye_dataset",
    "write_format_validation_report",
    "write_provenance",
    "write_validation_manifest",
]
