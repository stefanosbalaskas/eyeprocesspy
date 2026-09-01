from __future__ import annotations

import csv
import inspect
from pathlib import Path

import eyeprocesspy as ep

MATRIX = Path("parity/PARITY_MATRIX.csv")


def main() -> None:
    with MATRIX.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    missing_symbols: list[str] = []
    missing_docs: list[str] = []
    short_docs: list[str] = []

    for row in rows:
        name = row["python_name"] or row["r_name"]
        obj = getattr(ep, name, None)
        if obj is None:
            missing_symbols.append(name)
            continue
        doc = inspect.getdoc(obj) or ""
        if not doc.strip():
            missing_docs.append(name)
        elif len(doc.split()) < 4:
            short_docs.append(name)

    print(f"API_ROWS={len(rows)}")
    print(f"API_MISSING_SYMBOLS={len(missing_symbols)}")
    print(f"API_MISSING_DOCSTRINGS={len(missing_docs)}")
    print(f"API_SHORT_DOCSTRINGS={len(short_docs)}")

    for name in missing_symbols:
        print(f"API_MISSING_SYMBOL={name}")
    for name in missing_docs:
        print(f"API_MISSING_DOCSTRING={name}")
    for name in short_docs:
        print(f"API_SHORT_DOCSTRING={name}")

    if missing_symbols or missing_docs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
