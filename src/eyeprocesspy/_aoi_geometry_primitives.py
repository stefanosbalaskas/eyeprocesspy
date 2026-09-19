"""AOI geometry perturbation and uncertainty analysis.

Vendor-neutral scientific core for testing whether conclusions depend on small,
defensible changes to AOI geometry.  The module deliberately treats AOI
geometry as an analytical specification rather than a fixed truth.

Scientific safeguards
---------------------
* overlapping AOIs are labelled ``__ambiguous__`` by default;
* missing coordinates remain missing and are never converted to zero;
* unit conversions require explicit screen/viewing geometry;
* random jitter is seeded and its resolved seed is preserved;
* model callback errors and non-convergence are recorded, never interpreted as
  successful results;
* stability frequencies are descriptive sensitivity summaries, not
  probabilities that a scientific conclusion is true.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError
from .irt import EyeResult

OUTSIDE = "__outside__"
AMBIGUOUS = "__ambiguous__"


def _result(cls: str, **kwargs: Any) -> EyeResult:
    return EyeResult(kwargs, eyeprocess_class=cls)


def _frame(x: Any, name: str) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        raise EyeProcessValidationError(f"`{name}` must be a pandas DataFrame.")
    return x.copy()


def _finite_scalar(x: Any, name: str, *, nonnegative: bool = False) -> float:
    try:
        out = float(x)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError(f"`{name}` must be a finite numeric scalar.") from exc
    if not np.isfinite(out):
        raise EyeProcessValidationError(f"`{name}` must be a finite numeric scalar.")
    if nonnegative and out < 0:
        raise EyeProcessValidationError(f"`{name}` must be non-negative.")
    return out


def _normalise_pair(value: Any, name: str, *, nonnegative: bool = False) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise EyeProcessValidationError(f"`{name}` must be numeric.")
    if np.isscalar(value):
        v = _finite_scalar(value, name, nonnegative=nonnegative)
        return (v, v)
    values = list(value)
    if len(values) != 2:
        raise EyeProcessValidationError(f"`{name}` must be a scalar or length-2 numeric sequence.")
    return (
        _finite_scalar(values[0], f"{name}[0]", nonnegative=nonnegative),
        _finite_scalar(values[1], f"{name}[1]", nonnegative=nonnegative),
    )


def _software_provenance() -> dict[str, Any]:
    try:
        from . import __version__  # local import avoids import-cycle at module load
    except Exception:  # pragma: no cover - defensive when module is executed standalone
        __version__ = None
    return {"package": "eyeprocesspy", "version": __version__, "module": __name__}


def _stable_frame_hash(frame: pd.DataFrame) -> str:
    def normalise(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (list, tuple)):
            return [normalise(z) for z in v]
        if isinstance(v, Mapping):
            return {str(k): normalise(val) for k, val in sorted(v.items(), key=lambda z: str(z[0]))}
        if pd.isna(v) if np.isscalar(v) else False:
            return None
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        return v

    records = []
    for row in frame.to_dict(orient="records"):
        records.append({str(k): normalise(v) for k, v in sorted(row.items())})
    blob = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(blob).hexdigest()


def _polygon_array(value: Any) -> np.ndarray:
    try:
        polygon = np.asarray(value, dtype=float)
    except Exception as exc:
        raise EyeProcessValidationError(
            "Polygon geometry must be coercible to an n x 2 numeric array."
        ) from exc
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        raise EyeProcessValidationError(
            "Polygon geometry must contain at least three x/y vertices."
        )
    if not np.all(np.isfinite(polygon)):
        raise EyeProcessValidationError("Polygon vertices must be finite.")
    if np.allclose(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    if polygon.shape[0] < 3:
        raise EyeProcessValidationError(
            "Polygon geometry must contain at least three unique vertices."
        )
    return polygon


def _signed_polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, tol: float = 1e-12) -> bool:
    return (
        min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol
        and min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol
        and abs(_orientation(a, b, p)) <= tol
    )


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    o1, o2, o3, o4 = (
        _orientation(a, b, c),
        _orientation(a, b, d),
        _orientation(c, d, a),
        _orientation(c, d, b),
    )
    tol = 1e-12
    if (o1 > tol and o2 < -tol or o1 < -tol and o2 > tol) and (
        o3 > tol and o4 < -tol or o3 < -tol and o4 > tol
    ):
        return True
    if abs(o1) <= tol and _on_segment(a, b, c):
        return True
    if abs(o2) <= tol and _on_segment(a, b, d):
        return True
    if abs(o3) <= tol and _on_segment(c, d, a):
        return True
    if abs(o4) <= tol and _on_segment(c, d, b):
        return True
    return False


def _polygon_self_intersects(poly: np.ndarray) -> bool:
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            if j in {i, (i + 1) % n} or (j + 1) % n in {i, (i + 1) % n}:
                continue
            if i == 0 and (j + 1) % n == 0:
                continue
            c, d = poly[j], poly[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                return True
    return False


def _polygon_is_convex(poly: np.ndarray) -> bool:
    signs: list[int] = []
    n = len(poly)
    for i in range(n):
        cross = _orientation(poly[i], poly[(i + 1) % n], poly[(i + 2) % n])
        if abs(cross) <= 1e-12:
            continue
        signs.append(1 if cross > 0 else -1)
    return bool(signs) and len(set(signs)) == 1


def _line_intersection(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray
) -> np.ndarray:
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) <= 1e-12:
        raise EyeProcessValidationError(
            "Polygon offset produced parallel adjacent edges; simplify the polygon or use translation/anisotropic expansion instead."
        )
    q = p2 - p1
    t = (q[0] * d2[1] - q[1] * d2[0]) / cross
    return p1 + t * d1


def _offset_convex_polygon(poly: np.ndarray, distance: float) -> np.ndarray:
    """Offset a simple convex polygon by signed perpendicular distance.

    Positive ``distance`` dilates; negative distance erodes.  Concave polygons
    are rejected instead of being silently convexified or approximately scaled.
    """
    if not _polygon_is_convex(poly):
        raise EyeProcessValidationError(
            "True dilation/erosion is supported for convex polygons only. "
            "Concave polygons are not silently convexified; use translation, jitter, or anisotropic expansion, "
            "or preprocess geometry with a dedicated geometry engine."
        )
    area = _signed_polygon_area(poly)
    if abs(area) <= 1e-12:
        raise EyeProcessValidationError("Polygon area must be greater than zero.")
    ccw = area > 0
    shifted: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        edge = b - a
        length = float(np.hypot(edge[0], edge[1]))
        if length <= 1e-12:
            raise EyeProcessValidationError("Polygon contains a zero-length edge.")
        if ccw:
            outward = np.array([edge[1], -edge[0]]) / length
        else:
            outward = np.array([-edge[1], edge[0]]) / length
        shifted.append((a + distance * outward, edge))
    vertices = []
    for i in range(len(poly)):
        prev_point, prev_dir = shifted[(i - 1) % len(poly)]
        cur_point, cur_dir = shifted[i]
        vertices.append(_line_intersection(prev_point, prev_dir, cur_point, cur_dir))
    out = np.vstack(vertices)
    if abs(_signed_polygon_area(out)) <= 1e-12 or _polygon_self_intersects(out):
        raise EyeProcessValidationError("Erosion collapsed or invalidated polygon geometry.")
    original_sign = math.copysign(1.0, area)
    new_sign = math.copysign(1.0, _signed_polygon_area(out))
    if original_sign != new_sign:
        raise EyeProcessValidationError(
            "Erosion crossed the polygon interior and inverted geometry."
        )
    return out
