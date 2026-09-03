from __future__ import annotations

import csv
import inspect
from pathlib import Path

import eyeprocesspy as ep

MATRIX = Path("parity/PARITY_MATRIX.csv")
REFERENCE = Path("docs/reference/api.md")


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

    reference_text = REFERENCE.read_text(encoding="utf-8") if REFERENCE.exists() else ""
    reference_ok = "::: eyeprocesspy" in reference_text and "members: true" in reference_text
    resolved = len(rows) - len(missing_symbols)
    documented = resolved - len(missing_docs)
    doc_pct = (100.0 * documented / resolved) if resolved else 100.0

    print(f"API_ROWS={len(rows)}")
    print(f"API_RESOLVED_SYMBOLS={resolved}")
    print(f"API_MISSING_SYMBOLS={len(missing_symbols)}")
    print(f"API_DOCUMENTED_SYMBOLS={documented}")
    print(f"API_MISSING_DOCSTRINGS={len(missing_docs)}")
    print(f"API_SHORT_DOCSTRINGS={len(short_docs)}")
    print(f"API_DOCSTRING_COVERAGE={doc_pct:.2f}")
    print(f"API_REFERENCE_PAGE_OK={int(reference_ok)}")

    for name in missing_symbols:
        print(f"API_MISSING_SYMBOL={name}")
    for name in missing_docs:
        print(f"API_MISSING_DOCSTRING={name}")
    for name in short_docs:
        print(f"API_SHORT_DOCSTRING={name}")

    # Release-critical documentation invariants are structural: every frozen
    # public API must resolve from the package front door and the website must
    # expose the complete package through mkdocstrings. Function-level docstring
    # debt remains visible above and can be hardened without suppressing the
    # coverage/parity audits that identify scientific test debt.
    if missing_symbols or not reference_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
