"""Frozen R/078 reproducibility and provenance utilities.

Source reference:
``R/078-reproducibility-provenance-0-9.R`` from eyeprocess 0.11.1.

R-native serialization (RDS/dput) remains an explicit backend boundary.
Python fingerprints use a deterministic canonical Python serialization and
therefore do not claim byte-identical hashes with R serialization.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import locale
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError

__all__ = [
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

_MISSING = None
_FINGERPRINT_CLASS = "eye_reproducibility_fingerprint"
_GRAPH_CLASS = "eye_prov_graph"
_COMPARISON_CLASS = "eye_reproducibility_comparison"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Path)):
        return [value]
    if isinstance(value, pd.Series):
        return value.tolist()
    try:
        return list(value)
    except TypeError:
        return [value]


def _as_frame(value: Any, *, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return pd.DataFrame(value)
    except Exception as exc:
        raise EyeProcessValidationError(f"`{name}` must be coercible to a data frame.") from exc


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise EyeProcessValidationError(f"`{name}` is missing required column(s): " + ", ".join(missing) + ".")


def _recycle(value: Any, n: int) -> list[Any]:
    values = _as_list(value)
    if not values:
        return [None] * n
    return [values[index % len(values)] for index in range(n)]


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _canonicalize(value: Any) -> Any:
    """Convert supported Python objects to deterministic JSON-safe content."""
    if isinstance(value, pd.DataFrame):
        return {
            "__type__": "dataframe",
            "columns": [str(column) for column in value.columns],
            "index": [_clean_scalar(item) for item in value.index.tolist()],
            "data": [[_canonicalize(item) for item in row] for row in value.astype(object).to_numpy().tolist()],
            "attrs": _canonicalize(dict(value.attrs)),
        }

    if isinstance(value, pd.Series):
        return {
            "__type__": "series",
            "name": _clean_scalar(value.name),
            "index": [_clean_scalar(item) for item in value.index.tolist()],
            "data": [_canonicalize(item) for item in value.astype(object).tolist()],
        }

    if isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": _canonicalize(value.tolist()),
        }

    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
            if str(key) != "_python_class_marker"
        }

    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]

    if isinstance(value, (set, frozenset)):
        canonical = [_canonicalize(item) for item in value]
        return sorted(
            canonical,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        )

    if isinstance(value, (datetime, date, Path, np.generic)):
        return _clean_scalar(value)

    if value is pd.NA:
        return None

    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return {
        "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _tag(value: dict[str, Any], class_name: str) -> dict[str, Any]:
    value["eyeprocess_class"] = class_name
    return value


def _class_is(value: Any, class_name: str) -> bool:
    return isinstance(value, Mapping) and value.get("eyeprocess_class") == class_name


def object_hash(x):
    """Hash a Python object deterministically within eyeprocesspy."""
    payload = _canonical_json(x).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash_manifest(
    paths,
    algorithm=("md5", "sha256"),
):
    """Build a file hash manifest."""
    algorithms = _as_list(algorithm)
    algorithm_value = str(algorithms[0]).lower() if algorithms else "md5"
    if algorithm_value not in {"md5", "sha256"}:
        raise EyeProcessValidationError("algorithm must be one of: md5, sha256.")

    path_values = _as_list(paths)
    if not path_values:
        return pd.DataFrame(
            columns=[
                "path",
                "algorithm",
                "hash",
                "size_bytes",
                "modified",
            ]
        )

    rows = []
    for raw_path in path_values:
        path = Path(raw_path).expanduser().resolve(strict=True)
        digest = hashlib.new(algorithm_value)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        stat = path.stat()
        modified = datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
        rows.append(
            {
                "path": path.as_posix(),
                "algorithm": algorithm_value,
                "hash": digest.hexdigest(),
                "size_bytes": stat.st_size,
                "modified": modified,
            }
        )
    return pd.DataFrame(rows)


def analysis_environment_snapshot(packages=None):
    """Snapshot the active Python analysis environment."""
    if packages is None:
        package_names = sorted({name.split(".", 1)[0] for name in sys.modules if name and not name.startswith("_")})
    else:
        package_names = sorted({str(value) for value in _as_list(packages) if value is not None and str(value)})

    rows = []
    for name in package_names:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        rows.append({"package": name, "version": version})

    try:
        locale_value = locale.setlocale(locale.LC_ALL, None)
    except Exception:
        locale_value = None

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "os": {
            "sysname": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "locale": locale_value,
        "timezone": time_zone_name(),
        "packages": pd.DataFrame(rows),
    }


def time_zone_name() -> str | None:
    try:
        return datetime.now().astimezone().tzname()
    except Exception:
        return None


def eye_session_manifest(
    data=None,
    files=None,
    adapter=None,
    decisions=None,
    pipeline=None,
    seeds=None,
    notes=None,
):
    """Create a session-level provenance manifest."""
    return {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "eyeprocess_version": "0.11.1",
        "data_hash": None if data is None else object_hash(data),
        "files": (pd.DataFrame() if files is None else file_hash_manifest(files)),
        "adapter": None if adapter is None else str(_as_list(adapter)[0]),
        "decisions_hash": (None if decisions is None else object_hash(decisions)),
        "pipeline_hash": (None if pipeline is None else object_hash(pipeline)),
        "seeds": seeds,
        "environment": analysis_environment_snapshot(),
        "notes": notes,
    }


def eye_reproducibility_fingerprint(
    data=None,
    analysis_spec=None,
    model_spec=None,
    decisions=None,
    result=None,
    files=None,
    seeds=None,
    label="eyeprocess_analysis",
):
    """Construct a reproducibility fingerprint."""
    labels = _as_list(label)
    label_value = str(labels[0]) if labels else ""
    if not label_value:
        raise EyeProcessValidationError("label must be a non-empty scalar.")

    core = {
        "schema_version": "eyeprocess-reproducibility-0.9",
        "label": label_value,
        "eyeprocess_version": "0.11.1",
        "data_hash": None if data is None else object_hash(data),
        "analysis_spec_hash": (None if analysis_spec is None else object_hash(analysis_spec)),
        "model_spec_hash": (None if model_spec is None else object_hash(model_spec)),
        "decisions_hash": (None if decisions is None else object_hash(decisions)),
        "result_hash": None if result is None else object_hash(result),
        "file_manifest": (pd.DataFrame() if files is None else file_hash_manifest(files)),
        "seeds": seeds,
        "environment": analysis_environment_snapshot(),
    }
    core["fingerprint_hash"] = object_hash(core)
    return _tag(core, _FINGERPRINT_CLASS)


def compare_reproducibility_fingerprints(old, new):
    """Compare two reproducibility fingerprints."""
    fields = [
        "eyeprocess_version",
        "data_hash",
        "analysis_spec_hash",
        "model_spec_hash",
        "decisions_hash",
        "result_hash",
    ]
    rows = []
    for field in fields:
        old_value = old.get(field)
        new_value = new.get(field)
        rows.append(
            {
                "field": field,
                "old": None if old_value is None else str(old_value),
                "new": None if new_value is None else str(new_value),
                "identical": old_value == new_value,
            }
        )

    output = {
        "detail": pd.DataFrame(rows),
        "identical": (old.get("fingerprint_hash") == new.get("fingerprint_hash")),
        "old_hash": old.get("fingerprint_hash"),
        "new_hash": new.get("fingerprint_hash"),
    }
    return _tag(output, _COMPARISON_CLASS)


def verify_reproducibility_fingerprint(x):
    """Verify an internally stored fingerprint hash."""
    if not _class_is(x, _FINGERPRINT_CLASS):
        raise EyeProcessValidationError("x must be an eye_reproducibility_fingerprint.")

    payload = {key: value for key, value in x.items() if key not in {"fingerprint_hash", "eyeprocess_class"}}
    return x.get("fingerprint_hash") == object_hash(payload)


def _jsonify(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [{str(key): _jsonify(item) for key, item in row.items()} for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_jsonify(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(item) for item in value.tolist()]
    value = _clean_scalar(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_reproducibility_fingerprint(
    x,
    path,
    format=("rds", "dput", "json"),
):
    """Write a fingerprint; JSON is native, RDS/dput remain R-specific."""
    formats = _as_list(format)
    format_value = str(formats[0]).lower() if formats else "rds"

    if format_value in {"rds", "dput", "r"}:
        raise EyeProcessValidationError(
            "RDS/dput serialization is R-specific and is intentionally not emulated by eyeprocesspy. Use format='json'."
        )
    if format_value != "json":
        raise EyeProcessValidationError("format must be one of: rds, dput, json.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            _jsonify(x),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_path.resolve().as_posix()


def _restore_fingerprint_payload(value: Any) -> Any:
    if not isinstance(value, Mapping):
        raise EyeProcessValidationError("Fingerprint JSON must contain an object.")
    restored = dict(value)
    if isinstance(restored.get("file_manifest"), list):
        restored["file_manifest"] = pd.DataFrame(restored["file_manifest"])
    environment = restored.get("environment")
    if isinstance(environment, Mapping) and isinstance(environment.get("packages"), list):
        environment = dict(environment)
        environment["packages"] = pd.DataFrame(environment["packages"])
        restored["environment"] = environment
    return _tag(restored, _FINGERPRINT_CLASS)


def read_reproducibility_fingerprint(
    path,
    format=None,
):
    """Read a JSON fingerprint; RDS/dput inputs remain R-specific."""
    input_path = Path(path)
    format_value = input_path.suffix.lower().lstrip(".") if format is None else str(_as_list(format)[0]).lower()

    if format_value in {"rds", "dput", "r"}:
        raise EyeProcessValidationError(
            "RDS/dput deserialization is R-specific and is intentionally not emulated by eyeprocesspy."
        )
    if format_value != "json":
        raise EyeProcessValidationError("format must be one of: rds, dput, json.")

    value = json.loads(input_path.read_text(encoding="utf-8"))
    fingerprint = _restore_fingerprint_payload(value)
    if not verify_reproducibility_fingerprint(fingerprint):
        import warnings

        warnings.warn(
            "Stored fingerprint hash does not match the imported object representation.",
            RuntimeWarning,
            stacklevel=2,
        )
    return fingerprint


def provenance_lineage_table(
    id,
    type="entity",
    label=None,
    value=None,
):
    """Build a provenance lineage node table."""
    ids = _as_list(id)
    types = _as_list(type)
    labels = ids if label is None else _as_list(label)
    values = [None] if value is None else _as_list(value)

    n = max(len(ids), len(types), len(labels), len(values))
    if n < 1:
        raise EyeProcessValidationError("At least one provenance node is required.")

    output = pd.DataFrame(
        {
            "id": [None if item is None else str(item) for item in _recycle(ids, n)],
            "type": [None if item is None else str(item) for item in _recycle(types, n)],
            "label": [None if item is None else str(item) for item in _recycle(labels, n)],
            "value": [None if item is None else str(item) for item in _recycle(values, n)],
        }
    )

    if output["id"].isna().any() or (output["id"] == "").any():
        raise EyeProcessValidationError("provenance node ids cannot be missing or empty.")
    if output["type"].isna().any() or (output["type"] == "").any():
        raise EyeProcessValidationError("provenance node types cannot be missing or empty.")
    if output["id"].duplicated().any():
        raise EyeProcessValidationError("provenance node ids must be unique.")
    return output


def provenance_edge_table(
    from_,
    to,
    relation="wasDerivedFrom",
):
    """Build a provenance edge table."""
    from_values = _as_list(from_)
    to_values = _as_list(to)
    relations = _as_list(relation)
    n = max(len(from_values), len(to_values), len(relations))

    if n < 1:
        return pd.DataFrame(columns=["from", "to", "relation"])

    output = pd.DataFrame(
        {
            "from": [None if item is None else str(item) for item in _recycle(from_values, n)],
            "to": [None if item is None else str(item) for item in _recycle(to_values, n)],
            "relation": [None if item is None else str(item) for item in _recycle(relations, n)],
        }
    )

    if (
        output["from"].isna().any()
        or output["to"].isna().any()
        or (output["from"] == "").any()
        or (output["to"] == "").any()
    ):
        raise EyeProcessValidationError("provenance edge endpoints cannot be missing or empty.")
    if output["relation"].isna().any() or (output["relation"] == "").any():
        raise EyeProcessValidationError("provenance relations cannot be missing or empty.")
    return output


def eye_prov_graph(
    nodes,
    edges=None,
    metadata=None,
):
    """Construct a lightweight provenance graph."""
    nodes_frame = _as_frame(nodes, name="nodes")
    if edges is None:
        edges_frame = pd.DataFrame(columns=["from", "to", "relation"])
    else:
        edges_frame = _as_frame(edges, name="edges")

    _require_columns(
        nodes_frame,
        ["id", "type", "label"],
        name="nodes",
    )
    _require_columns(
        edges_frame,
        ["from", "to", "relation"],
        name="edges",
    )

    graph = _tag(
        {
            "nodes": nodes_frame,
            "edges": edges_frame,
            "metadata": {} if metadata is None else metadata,
        },
        _GRAPH_CLASS,
    )
    validate_eye_prov_graph(graph)
    return graph


def validate_eye_prov_graph(x):
    """Validate a provenance graph."""
    if not _class_is(x, _GRAPH_CLASS):
        raise EyeProcessValidationError("x must be an eye_prov_graph.")

    nodes = _as_frame(x["nodes"], name="nodes")
    edges = _as_frame(x["edges"], name="edges")
    _require_columns(nodes, ["id", "type", "label"], name="nodes")
    _require_columns(edges, ["from", "to", "relation"], name="edges")

    ids = nodes["id"].astype("string")
    if ids.isna().any() or (ids == "").any():
        raise EyeProcessValidationError("provenance node ids cannot be missing or empty.")
    if ids.duplicated().any():
        raise EyeProcessValidationError("duplicate provenance node ids.")

    known = set(ids.astype(str))
    endpoints = set(edges["from"].astype(str)) | set(edges["to"].astype(str))
    bad = sorted(endpoints - known)
    if bad:
        raise EyeProcessValidationError("edges reference unknown node(s): " + ", ".join(bad))
    return True


def export_prov_json(
    x,
    path,
):
    """Export compact PROV-oriented JSON."""
    validate_eye_prov_graph(x)
    payload = {
        "schema": "eyeprocess-prov-0.9",
        "nodes": _jsonify(x["nodes"]),
        "edges": _jsonify(x["edges"]),
        "metadata": _jsonify(x.get("metadata", {})),
    }

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_path.as_posix()


def export_ro_crate_metadata(
    path="ro-crate-metadata.json",
    name="eyeprocess analysis",
    description="Reproducible eyeprocess analysis crate",
    files=None,
    creator=None,
    license=None,
    doi=None,
):
    """Export minimal RO-Crate 1.3 metadata."""
    graph = [
        {
            "@id": "ro-crate-metadata.json",
            "@type": "CreativeWork",
            "about": {"@id": "./"},
            "conformsTo": {"@id": "https://w3id.org/ro/crate/1.3"},
        },
        {
            "@id": "./",
            "@type": "Dataset",
            "name": name,
            "description": description,
            "datePublished": date.today().isoformat(),
        },
    ]

    dataset = graph[1]
    if license is not None:
        dataset["license"] = license
    if doi is not None:
        dataset["identifier"] = doi
    if creator is not None:
        dataset["creator"] = {"@id": "#creator"}
        graph.append(
            {
                "@id": "#creator",
                "@type": "Person",
                "name": creator,
            }
        )

    file_values = _as_list(files)
    if file_values:
        resolved = [Path(item).expanduser().resolve(strict=True) for item in file_values]
        basenames = [item.name for item in resolved]
        if len(set(basenames)) != len(basenames):
            raise EyeProcessValidationError("RO-Crate file basenames must be unique in this minimal exporter.")
        dataset["hasPart"] = [{"@id": basename} for basename in basenames]
        for item in resolved:
            graph.append(
                {
                    "@id": item.name,
                    "@type": "File",
                    "contentSize": item.stat().st_size,
                }
            )

    payload = {
        "@context": "https://w3id.org/ro/crate/1.3/context",
        "@graph": graph,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_path.as_posix()


def _dot_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def write_prov_dot(x):
    """Return Graphviz DOT for a provenance graph."""
    validate_eye_prov_graph(x)

    lines = ["digraph eyeprocess_provenance {"]
    for _, row in x["nodes"].iterrows():
        lines.append(
            f'  "{_dot_escape(row["id"])}" [label="{_dot_escape(row["label"])}\\n({_dot_escape(row["type"])})"];'
        )
    for _, row in x["edges"].iterrows():
        lines.append(
            f'  "{_dot_escape(row["from"])}" -> "{_dot_escape(row["to"])}" [label="{_dot_escape(row["relation"])}"];'
        )
    lines.append("}")
    return "\n".join(lines)
