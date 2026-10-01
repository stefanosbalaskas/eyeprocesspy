#!/usr/bin/env python3
"""Assemble the CHI manuscript from section-level source files.

This is a writing/presentation utility only. It does not read empirical result
artifacts, refit models, rerun measurement pipelines, or modify frozen research
contracts. The section files remain the editable source of truth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

SECTION_FILES = (
    "introduction_related_work.md",
    "methods_draft.md",
    "results_draft.md",
    "discussion_draft.md",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()


def _drop_draft_h1(text: str) -> str:
    """Remove one draft-only H1 while preserving numbered manuscript headings."""
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


def assemble(paper_dir: Path) -> str:
    frontmatter = _read(paper_dir / "frontmatter_conclusion.md")
    marker = "\n## 6 Conclusion\n"
    if marker not in frontmatter:
        raise ValueError("frontmatter_conclusion.md is missing the Conclusion marker.")
    front, conclusion_body = frontmatter.split(marker, maxsplit=1)
    conclusion = "## 6 Conclusion\n\n" + conclusion_body.strip()

    sections: list[str] = [front.strip()]
    for name in SECTION_FILES:
        path = paper_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        sections.append(_drop_draft_h1(_read(path)))
    sections.append(conclusion)
    sections.append(
        "## References\n\n"
        "Bibliography source: `literature_references.bib`. "
        "The final ACM build should render this locked bibliography rather than "
        "maintain a second hand-edited reference list."
    )
    return "\n\n".join(section.strip() for section in sections if section.strip()) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <paper-dir>/manuscript_draft.md.",
    )
    args = parser.parse_args()

    output = args.output or args.paper_dir / "manuscript_draft.md"
    content = assemble(args.paper_dir)
    output.write_text(content, encoding="utf-8", newline="\n")
    print(f"wrote {output} ({len(content.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
