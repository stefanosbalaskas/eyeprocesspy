#!/usr/bin/env python3
"""Report stable word-count metrics for the assembled CHI manuscript."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

WORD_RE = re.compile(r"\b[\w’'-]+\b", flags=re.UNICODE)


def _slice(text: str, start: str, end: str) -> str:
    if start not in text or end not in text:
        raise ValueError(f"Missing manuscript marker: {start!r} or {end!r}.")
    return text.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]


def _strip_nonprose(text: str) -> str:
    lines: list[str] = []
    in_fence = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("**Figure "):
            continue
        lines.append(line)
    return "\n".join(lines)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def manuscript_metrics(text: str) -> dict[str, int]:
    abstract = _slice(text, "## Abstract", "## Keywords")
    main = _slice(text, "## 1 Introduction", "## References")
    return {
        "abstract_words": word_count(_strip_nonprose(abstract)),
        "main_text_words_excluding_headings_code_and_figure_captions": word_count(
            _strip_nonprose(main)
        ),
        "assembled_words_all_markdown": word_count(text),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manuscript", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--abstract-limit", type=int, default=150)
    parser.add_argument("--fail-over-abstract", action="store_true")
    args = parser.parse_args()

    text = args.manuscript.read_text(encoding="utf-8")
    metrics = manuscript_metrics(text)
    payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8", newline="\n")

    if args.fail_over_abstract and metrics["abstract_words"] > args.abstract_limit:
        raise SystemExit(
            f"Abstract has {metrics['abstract_words']} words; limit is {args.abstract_limit}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
