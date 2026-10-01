#!/usr/bin/env python3
"""Build an anonymous ACM/CHI review manuscript from the Markdown sources.

The Markdown section files remain the editable source of truth. This utility is
presentation-only: it assembles those sources, converts known author-year
references into BibTeX-backed citations, emits an anonymous single-column
``acmart`` review manuscript, and optionally compiles a PDF.

It never reads empirical workflow artifacts or changes the frozen analyses.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from assemble_manuscript import assemble

CITATION_REPLACEMENTS = {
    "(Andersson et al., 2017)": "[@andersson2017algorithm]",
    "Andersson et al. (2017)": "@andersson2017algorithm",
    "(Hessels et al., 2016)": "[@hessels2016aoi]",
    "Hessels et al. (2016)": "@hessels2016aoi",
    "(Feit et al., 2017)": "[@feit2017everyday]",
    "Feit et al. (2017)": "@feit2017everyday",
    "(Steegen et al., 2016)": "[@steegen2016multiverse]",
    "Steegen et al. (2016)": "@steegen2016multiverse",
    "(Simonsohn et al., 2020)": "[@simonsohn2020specification]",
    "Simonsohn et al. (2020)": "@simonsohn2020specification",
    "(Silberzahn et al., 2018)": "[@silberzahn2018many]",
    "Silberzahn et al. (2018)": "@silberzahn2018many",
    "(Liu et al., 2021; Sarma et al., 2023)": (
        "[@liu2021boba; @sarma2023multiverse]"
    ),
    "Liu et al. (2021)": "@liu2021boba",
    "Sarma et al. (2023)": "@sarma2023multiverse",
    "Peelle and Van Engen (2021)": "@peelle2021time",
    "Peelle & Van Engen (2021)": "@peelle2021time",
    "Godwin, Lee, and Drieghe (2025)": "@godwin2025multiverse",
    "Godwin et al. (2025)": "@godwin2025multiverse",
    "(Dar et al., 2021)": "[@dar2021remodnav]",
    "Dar et al. (2021)": "@dar2021remodnav",
    "Nyström and Holmqvist (2010)": "@nystrom2010adaptive",
    "Holmqvist, Nyström, and Mulvey (2012)": "@holmqvist2012quality",
    "Orquin and Holmqvist (2018)": "@orquin2018threats",
    "Dunn et al. (2024; 2023 edition)": "@dunn2024reporting",
    "Dunn et al. (2024)": "@dunn2024reporting",
    "(Steegen et al., 2016; Simonsohn et al., 2020)": (
        "[@steegen2016multiverse; @simonsohn2020specification]"
    ),
    "(Peelle & Van Engen, 2021; Godwin et al., 2025)": (
        "[@peelle2021time; @godwin2025multiverse]"
    ),
    (
        "(Andersson et al., 2017; Hessels et al., 2016; "
        "Holmqvist et al., 2012; Dunn et al., 2024)"
    ): (
        "[@andersson2017algorithm; @hessels2016aoi; "
        "@holmqvist2012quality; @dunn2024reporting]"
    ),
    "(Juřík et al., 2025)": "[@jurik2025srl]",
    "Juřík et al. (2025)": "@jurik2025srl",
}

FIGURE_FILES = {
    1: "fig1_srl_specification_curve.pdf",
    2: "fig2_srl_detector_distance.pdf",
    3: "fig3_mcfw_context_jaccard.pdf",
}

FIGURE_DESCRIPTIONS = {
    1: (
        "Specification curve for 144 SRL measurement specifications. Point "
        "estimates span both sides of the no-effect line while every 95 percent "
        "confidence interval crosses the line. Marker shape distinguishes the "
        "60, 65, and 70 cm viewing-distance assumptions."
    ),
    2: (
        "Line chart of median SRL Prompt rate ratios by event detector and "
        "assumed viewing distance. The three detector trajectories separate as "
        "viewing distance changes, illustrating detector-by-geometry sensitivity."
    ),
    3: (
        "Line chart of median fixation-state Jaccard overlap across six MCFW-Gaze "
        "contexts. I-VT 40 degrees per second and I-DT have consistently greater "
        "overlap than detector pairs involving I-VT 30 degrees per second."
    ),
}


def _between(text: str, start: str, end: str) -> str:
    if start not in text or end not in text:
        raise ValueError(f"Missing manuscript marker: {start!r} or {end!r}.")
    return text.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0].strip()


def _latex_escape(text: str) -> str:
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("×", r"$\times$"),
        ("°", r"$^\circ$"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _citationize(text: str) -> str:
    for source, target in sorted(
        CITATION_REPLACEMENTS.items(), key=lambda item: -len(item[0])
    ):
        text = text.replace(source, target)

    text = text.replace(
        "MCFW-Gaze provides raw binocular gaze",
        "MCFW-Gaze [@nagasawa2026mcfw] provides raw binocular gaze",
    )
    text = text.replace(
        "Tobii documentation records device timestamps in microseconds",
        (
            "Tobii Pro SDK documentation [@tobii2026timestamps] records device "
            "timestamps in microseconds"
        ),
    )
    return text


def _ascii_safe_code_fences(text: str) -> str:
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        elif in_fence:
            line = (
                line.replace("×", "x")
                .replace("°", " deg")
                .replace("≥", ">=")
                .replace("≤", "<=")
            )
        lines.append(line)
    return "\n".join(lines)


def _figure_block(number: int, caption: str, figure_available: bool) -> str:
    filename = FIGURE_FILES[number]
    escaped_caption = _latex_escape(caption)
    description = _latex_escape(FIGURE_DESCRIPTIONS[number])
    if figure_available:
        visual = rf"\includegraphics[width=\linewidth]{{figures/{filename}}}"
    else:
        visual = (
            rf"\fbox{{\parbox[c][1.65in][c]{{0.92\linewidth}}"
            rf"{{\centering Figure {number} placeholder\\"
            r"generated from frozen artifact}}}"
        )
    return f"""```{{=latex}}
\begin{{figure}}[t]
\centering
{visual}
\caption{{{escaped_caption}}}
\Description{{{description}}}
\label{{fig:figure{number}}}
\end{{figure}}
```"""


def _replace_figure_caption_section(
    body: str,
    *,
    available_figures: set[int],
) -> str:
    marker = "## Figure captions"
    discussion = "## 5 Discussion"
    if marker not in body:
        raise ValueError("Assembled manuscript is missing the Figure captions section.")
    before, rest = body.split(marker, maxsplit=1)
    if discussion not in rest:
        raise ValueError("Figure captions are not followed by the Discussion section.")
    captions, after = rest.split(discussion, maxsplit=1)
    lines = [line.strip() for line in captions.splitlines() if line.strip()]
    blocks: list[str] = []
    for number in FIGURE_FILES:
        prefix = f"**Figure {number}."
        matches = [line for line in lines if line.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one caption for Figure {number}.")
        caption = re.sub(
            rf"^\*\*Figure {number}\.\s*",
            "",
            matches[0],
        ).replace("**", "")
        blocks.append(
            _figure_block(
                number,
                caption,
                number in available_figures,
            )
        )
    return (
        before.rstrip()
        + "\n\n"
        + "\n\n".join(blocks)
        + "\n\n"
        + discussion
        + "\n"
        + after.lstrip()
    )


def _pandoc_body(markdown: str, *, pandoc: str) -> str:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "body.md"
        output = root / "body.tex"
        source.write_text(markdown, encoding="utf-8", newline="\n")
        subprocess.run(
            [
                pandoc,
                str(source),
                "--from=markdown+citations+raw_tex",
                "--to=latex",
                "--natbib",
                "--listings",
                "--top-level-division=section",
                f"--output={output}",
            ],
            check=True,
        )
        latex = output.read_text(encoding="utf-8")
    return re.sub(
        r"(\\(?:section|subsection|subsubsection)\{)(\d+(?:\.\d+)*\s+)",
        lambda match: match.group(1),
        latex,
    )


def _copy_available_figures(figure_dir: Path | None, output_dir: Path) -> set[int]:
    available: set[int] = set()
    target = output_dir / "figures"
    if figure_dir is None:
        return available
    for number, filename in FIGURE_FILES.items():
        source = figure_dir / filename
        if source.is_file():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / filename)
            available.add(number)
    return available


def _compiler(name: str, fallback: str | None = None) -> str:
    executable = shutil.which(name)
    if executable and Path(executable).exists():
        return executable
    if fallback:
        executable = shutil.which(fallback)
        if executable and Path(executable).exists():
            return executable
    raise RuntimeError(f"Required executable not found: {name}")


def _compile_pdf(output_dir: Path, tex_name: str) -> None:
    pdflatex = _compiler("pdflatex")
    try:
        bibtex = _compiler("bibtex")
    except RuntimeError:
        bibtex = _compiler("bibtex.original")
    stem = Path(tex_name).stem
    command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_name]
    subprocess.run(command, cwd=output_dir, check=True)
    subprocess.run([bibtex, stem], cwd=output_dir, check=True)
    subprocess.run(command, cwd=output_dir, check=True)
    subprocess.run(command, cwd=output_dir, check=True)


def build(
    paper_dir: Path,
    output_dir: Path,
    *,
    figure_dir: Path | None,
    compile_pdf: bool,
) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manuscript = assemble(paper_dir)
    title = manuscript.splitlines()[0].removeprefix("# ").strip()
    abstract = _between(manuscript, "## Abstract", "## Keywords")
    keywords = _between(manuscript, "## Keywords", "## 1 Introduction")
    body = "## 1 Introduction\n" + manuscript.split("## 1 Introduction", maxsplit=1)[1]
    body = body.split("## References", maxsplit=1)[0].rstrip()

    available = _copy_available_figures(figure_dir, output_dir)
    body = _replace_figure_caption_section(body, available_figures=available)
    body = _citationize(body)
    body = _ascii_safe_code_fences(body)

    pandoc = _compiler("pandoc")
    body_latex = _pandoc_body(body, pandoc=pandoc)

    bibliography = paper_dir / "literature_references.bib"
    if not bibliography.is_file():
        raise FileNotFoundError(bibliography)
    shutil.copy2(bibliography, output_dir / bibliography.name)

    keyword_text = ", ".join(
        part.strip() for part in re.split(r"[;,]", keywords) if part.strip()
    )
    tex = rf"""\documentclass[manuscript,review,anonymous]{{acmart}}
\usepackage{{listings}}
\providecommand{{\passthrough}}[1]{{#1}}
\providecommand{{\tightlist}}{{\setlength{{\itemsep}}{{0pt}}\setlength{{\parskip}}{{0pt}}}}
\setcopyright{{none}}
\settopmatter{{printacmref=false,printccs=false,printfolios=true}}
\renewcommand\footnotetextcopyrightpermission[1]{{}}
\begin{{document}}
\title{{{_latex_escape(title)}}}
\author{{Anonymous Author(s)}}
\begin{{abstract}}
{abstract}
\end{{abstract}}
\keywords{{{_latex_escape(keyword_text)}}}
\maketitle
{body_latex}
\bibliographystyle{{ACM-Reference-Format}}
\bibliography{{literature_references}}
\end{{document}}
"""
    tex_path = output_dir / "manuscript_acm_review.tex"
    tex_path.write_text(tex, encoding="utf-8", newline="\n")

    pdf_path: Path | None = None
    if compile_pdf:
        _compile_pdf(output_dir, tex_path.name)
        candidate = output_dir / "manuscript_acm_review.pdf"
        if not candidate.is_file():
            raise RuntimeError("LaTeX compilation completed without a PDF output.")
        pdf_path = candidate
    return tex_path, pdf_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <paper-dir>/build/acm-review.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        help="Optional directory containing the three PDF publication figures.",
    )
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or args.paper_dir / "build" / "acm-review"
    tex_path, pdf_path = build(
        args.paper_dir,
        output_dir,
        figure_dir=args.figure_dir,
        compile_pdf=args.compile,
    )
    print(f"wrote {tex_path}")
    if pdf_path is not None:
        print(f"wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
