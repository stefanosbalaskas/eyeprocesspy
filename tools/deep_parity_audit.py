from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

MATRIX = Path("parity/PARITY_MATRIX.csv")
DOCS_ARTICLES = Path("docs/articles")
EXPECTED_EXPORTS = 1182
EXPECTED_FROZEN_ARTICLES = 88


def _load_rows() -> list[dict[str, str]]:
    with MATRIX.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _print_counts(rows: list[dict[str, str]]) -> None:
    print(f"TOTAL_ROWS={len(rows)}")
    for column in (
        "p1_api",
        "p2_structural",
        "p3_semantic",
        "p4_numerical",
        "p5_algorithmic",
        "p6_plot",
        "p7_docs_examples",
    ):
        counts = Counter(row[column] for row in rows)
        print(f"[{column}]")
        for status, count in sorted(counts.items()):
            print(f"{status or '<blank>'}={count}")

    for column in ("p3_semantic", "p4_numerical", "p7_docs_examples"):
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            grouped[row["source_file"]][row[column]] += 1
        print(f"[{column}_BY_SOURCE]")
        for source in sorted(grouped):
            summary = ",".join(
                f"{status or '<blank>'}:{count}"
                for status, count in sorted(grouped[source].items())
            )
            print(f"{source}|{summary}")


def _print_debt(rows: list[dict[str, str]]) -> dict[str, int]:
    p4_not_started = [row for row in rows if row["p4_numerical"] == "not_started"]
    print(f"P4_NOT_STARTED={len(p4_not_started)}")
    for row in p4_not_started:
        print(
            "P4_NOT_STARTED_ROW="
            f"{row['source_file']}|{row['r_name']}|{row['python_module']}|{row['python_test_file']}"
        )

    p4_diff = [row for row in rows if row["p4_numerical"] == "python_reference_differs"]
    p4_diff_without_blocker = [row for row in p4_diff if not row["blocker"].strip()]
    print(f"P4_PYTHON_REFERENCE_DIFFERS={len(p4_diff)}")
    print(f"P4_DIFF_WITHOUT_BLOCKER={len(p4_diff_without_blocker)}")
    for row in p4_diff_without_blocker:
        print(f"P4_DIFF_WITHOUT_BLOCKER_ROW={row['source_file']}|{row['r_name']}")

    p7_blank = [row for row in rows if not row["p7_docs_examples"].strip()]
    p7_initial = [row for row in rows if row["p7_docs_examples"] == "docstring_initial"]
    print(f"P7_BLANK={len(p7_blank)}")
    print(f"P7_DOCSTRING_INITIAL={len(p7_initial)}")

    plot_rows = [row for row in rows if row["plot_candidate"].lower() == "true"]
    p6_not_started = [row for row in plot_rows if row["p6_plot"] == "not_started"]
    print(f"PLOT_CANDIDATES={len(plot_rows)}")
    print(
        "PLOT_STATUS="
        + ",".join(
            f"{key}:{value}"
            for key, value in sorted(Counter(row["p6_plot"] for row in plot_rows).items())
        )
    )
    print(f"P6_NOT_STARTED={len(p6_not_started)}")
    for row in p6_not_started:
        print(
            "P6_NOT_STARTED_ROW="
            f"{row['source_file']}|{row['r_name']}|{row['python_module']}|{row['python_test_file']}"
        )

    article_files = sorted(DOCS_ARTICLES.glob("*.md")) if DOCS_ARTICLES.exists() else []
    print(f"DOC_ARTICLE_FILES={len(article_files)}")
    print(f"FROZEN_ARTICLE_REFERENCE={EXPECTED_FROZEN_ARTICLES}")

    return {
        "p4_not_started": len(p4_not_started),
        "p4_diff_without_blocker": len(p4_diff_without_blocker),
        "p6_not_started": len(p6_not_started),
        "p7_blank": len(p7_blank),
        "article_files": len(article_files),
    }


def _release_gate(rows: list[dict[str, str]], debt: dict[str, int]) -> None:
    failures: list[str] = []
    if len(rows) != EXPECTED_EXPORTS:
        failures.append(f"parity ledger has {len(rows)} rows, expected {EXPECTED_EXPORTS}")
    if any(row["p1_api"] != "implemented" for row in rows):
        failures.append("p1_api is not complete")
    if any(row["p2_structural"] != "implemented" for row in rows):
        failures.append("p2_structural is not complete")
    if debt["p4_not_started"]:
        failures.append(f"{debt['p4_not_started']} p4_numerical rows remain not_started")
    if debt["p4_diff_without_blocker"]:
        failures.append(
            f"{debt['p4_diff_without_blocker']} python_reference_differs rows lack an explicit blocker"
        )
    if debt["p6_not_started"]:
        failures.append(f"{debt['p6_not_started']} plot candidates remain p6_plot=not_started")
    if debt["p7_blank"]:
        failures.append(f"{debt['p7_blank']} APIs have no documentation/example status")
    if debt["article_files"] < EXPECTED_FROZEN_ARTICLES:
        failures.append(
            f"only {debt['article_files']} Markdown article files are present; "
            f"frozen reference contains {EXPECTED_FROZEN_ARTICLES}"
        )

    if failures:
        print("RELEASE_GATE=FAIL")
        for failure in failures:
            print(f"RELEASE_GATE_FAILURE={failure}")
        raise SystemExit(1)
    print("RELEASE_GATE=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit eyeprocesspy deep-parity release debt.")
    parser.add_argument(
        "--release-gate",
        action="store_true",
        help="fail unless mandatory parity/documentation release gates are closed",
    )
    args = parser.parse_args()

    rows = _load_rows()
    _print_counts(rows)
    debt = _print_debt(rows)
    if args.release_gate:
        _release_gate(rows, debt)


if __name__ == "__main__":
    main()
