"""Build the pure-Python validation wheel without external build dependencies.

This is intentionally a local validation fallback for sandboxed/offline development.
GitHub CI remains authoritative for the normal PEP 517 wheel/sdist build.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "eyeprocesspy"
DIST = ROOT / "dist"
VERSION = "0.1.0.dev0"
DIST_INFO = f"eyeprocesspy-{VERSION}.dist-info"
WHEEL = DIST / f"eyeprocesspy-{VERSION}-py3-none-any.whl"

METADATA = """Metadata-Version: 2.4
Name: eyeprocesspy
Version: 0.1.0.dev0
Summary: Python parity implementation of eyeprocess: vendor-neutral eye-tracking and multimodal process-data infrastructure.
Requires-Python: >=3.11
License: MIT
Requires-Dist: numpy>=1.26
Requires-Dist: pandas>=2.2
Requires-Dist: scipy>=1.13

"""
WHEEL_TEXT = """Wheel-Version: 1.0
Generator: eyeprocesspy-offline-validation
Root-Is-Purelib: true
Tag: py3-none-any

"""


def digest(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")


def main() -> None:
    DIST.mkdir(exist_ok=True)
    files: dict[str, bytes] = {}
    for path in sorted(PKG.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        arc = Path("eyeprocesspy") / path.relative_to(PKG)
        files[arc.as_posix()] = path.read_bytes()
    files[f"{DIST_INFO}/METADATA"] = METADATA.encode()
    files[f"{DIST_INFO}/WHEEL"] = WHEEL_TEXT.encode()

    rows = [[name, digest(data), str(len(data))] for name, data in files.items()]
    rows.append([f"{DIST_INFO}/RECORD", "", ""])
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    files[f"{DIST_INFO}/RECORD"] = buf.getvalue().encode()

    with zipfile.ZipFile(WHEEL, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    print(WHEEL)


if __name__ == "__main__":
    main()
