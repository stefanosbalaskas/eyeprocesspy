from __future__ import annotations

import csv
from pathlib import Path

import eyeprocesspy as ep


MATRIX = Path(__file__).parents[1] / "parity" / "PARITY_MATRIX.csv"


def test_all_frozen_r_exports_resolve_from_package_front_door() -> None:
    with MATRIX.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    names = [row["python_name"] or row["r_name"] for row in rows]
    missing = [name for name in names if getattr(ep, name, None) is None]

    assert len(rows) == 1182
    assert missing == []
