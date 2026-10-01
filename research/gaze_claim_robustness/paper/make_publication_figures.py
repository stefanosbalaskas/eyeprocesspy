#!/usr/bin/env python3
"""Generate the frozen CHI paper figures from canonical frozen result inputs.

This script is presentation-only. It does not refit models, rerun detectors, or
change either frozen empirical universe. It supports two equivalent input modes:

1. the canonical GitHub Actions ZIP artifacts, whose full SHA-256 values are
   verified before reading their result tables; or
2. compact, committed paper snapshots extracted verbatim from those artifacts,
   whose file SHA-256 values are likewise verified.

The figures use marker/line-style encodings in addition to color so their
scientific distinctions remain legible in grayscale and for readers with color-
vision deficiencies. The three SVG outputs are checked against frozen hashes
under the explicit renderer versions that produced those bytes. PDF sidecars
are emitted only for manuscript typesetting.
"""
from __future__ import annotations

import argparse
import hashlib
import tempfile
import zipfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["svg.hashsalt"] = "gaze-claim-robustness-chi"
plt.rcParams["svg.fonttype"] = "none"

SRL_ARTIFACT_SHA256 = "b125b071a9a1bf68f5ba3c11f1b8fdeedca1195de94a64e27155af79d7ae342d"
MCFW_ARTIFACT_SHA256 = "ed2af7a036d1fb9a62b5226b71b6f02d007f87a9d23b1970828f9c9742b93020"
SRL_SNAPSHOT_SHA256 = "706b57051efe074700d3b3dbc693bece3d3747da2d5940565f4c93f6a9dc6193"
MCFW_SNAPSHOT_SHA256 = "d0c7471199e5a6242c1f51f872f4f7a3a915adb8a91749adea7fe0ff60ee235f"

RENDERER_VERSIONS = {
    "matplotlib": "3.10.8",
    "numpy": "2.3.5",
    "pandas": "2.2.3",
}

SVG_SHA256 = {
    "fig1_srl_specification_curve.svg": (
        "4c2531bdc716d51edc91a71196df61383a046c06a27ac04fb73301b002a4bcbc"
    ),
    "fig2_srl_detector_distance.svg": (
        "a6ad1b378f624f652765fefbdb6aa14174691e67009562348ba54dc83f1d9666"
    ),
    "fig3_mcfw_context_jaccard.svg": (
        "8123b2352966e5eca4773bac6a525bc16786685f1a3c052a820c88208b05cabe"
    ),
}

DETECTOR_LABELS = {
    "ivt_30_100_simple": "I-VT 30°/s, 100 ms",
    "ivt_40_50_simple": "I-VT 40°/s, 50 ms",
    "idt_1_100": "I-DT 1°, 100 ms",
}
DETECTOR_STYLES = {
    "idt_1_100": {"marker": "o", "linestyle": "-"},
    "ivt_30_100_simple": {"marker": "s", "linestyle": "--"},
    "ivt_40_50_simple": {"marker": "^", "linestyle": "-."},
}
PAIR_LABELS = {
    ("ivt_30_100_simple", "ivt_40_50_simple"): "I-VT 30 vs I-VT 40",
    ("ivt_30_100_simple", "idt_1_100"): "I-VT 30 vs I-DT",
    ("ivt_40_50_simple", "idt_1_100"): "I-VT 40 vs I-DT",
}
PAIR_LINESTYLES = {
    ("ivt_30_100_simple", "ivt_40_50_simple"): "-",
    ("ivt_30_100_simple", "idt_1_100"): "--",
    ("ivt_40_50_simple", "idt_1_100"): "-.",
}
EYE_MARKERS = {"left": "o", "right": "s"}
CONTEXT_ORDER = [
    "natural_image",
    "gaze_pattern_auth",
    "password",
    "web_shopping",
    "web_news",
    "web_video",
]
CONTEXT_LABELS = {
    "natural_image": "Natural image",
    "gaze_pattern_auth": "Gaze auth",
    "password": "Password",
    "web_shopping": "Shopping",
    "web_news": "News",
    "web_video": "Video",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual}")


def _verify_renderer_versions() -> None:
    actual = {
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    if actual != RENDERER_VERSIONS:
        raise RuntimeError(
            "Frozen SVG byte verification requires the declared renderer "
            f"environment {RENDERER_VERSIONS}; observed {actual}."
        )


def _extract(zip_path: Path, target: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)


def _save_figure(fig: plt.Figure, svg_out: Path) -> None:
    fig.savefig(svg_out, format="svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(
        svg_out.with_suffix(".pdf"),
        format="pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def figure_srl_specification_curve(srl: pd.DataFrame, out: Path) -> None:
    order = srl.sort_values("estimate").reset_index(drop=True)
    x = np.arange(1, len(order) + 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.vlines(x, order["CI_lower"], order["CI_upper"], linewidth=0.45, alpha=0.45)
    for distance, marker in ((60.0, "o"), (65.0, "s"), (70.0, "^")):
        mask = order["viewing_distance_cm"].eq(distance)
        ax.scatter(
            x[mask],
            order.loc[mask, "estimate"],
            s=18,
            marker=marker,
            label=f"{int(distance)} cm",
            zorder=3,
        )
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_xlabel("Specification (ordered by Prompt log-rate ratio)")
    ax.set_ylabel("Prompt log rate ratio")
    ax.set_title("SRL: substantive estimate across 144 frozen measurement specifications")
    ax.legend(title="Viewing distance", ncol=3, frameon=False, loc="upper left")
    ax.text(
        0.99,
        0.02,
        "All 144 Wald 95% CIs include 0",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout()
    _save_figure(fig, out)


def figure_srl_detector_distance(srl: pd.DataFrame, out: Path) -> None:
    med = (
        srl.groupby(["viewing_distance_cm", "detector_id"], as_index=False)
        .agg(
            median_rr=("rate_ratio", "median"),
            q25_rr=("rate_ratio", lambda x: x.quantile(0.25)),
            q75_rr=("rate_ratio", lambda x: x.quantile(0.75)),
        )
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for detector, group in med.groupby("detector_id"):
        group = group.sort_values("viewing_distance_cm")
        style = DETECTOR_STYLES[detector]
        yerr = np.vstack(
            [
                group["median_rr"] - group["q25_rr"],
                group["q75_rr"] - group["median_rr"],
            ]
        )
        ax.errorbar(
            group["viewing_distance_cm"],
            group["median_rr"],
            yerr=yerr,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            elinewidth=1.2,
            capsize=3,
            label=DETECTOR_LABELS[detector],
        )
    ax.axhline(1, linewidth=1, linestyle="--")
    ax.set_xticks([60, 65, 70])
    ax.set_xlabel("Assumed viewing distance (cm)")
    ax.set_ylabel("Median Prompt rate ratio")
    ax.set_title("SRL: detector and visual-angle geometry jointly shift the effect estimate")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, out)


def figure_mcfw_context_jaccard(context: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    positions = np.arange(len(CONTEXT_ORDER))
    offsets = {"left": -0.10, "right": 0.10}
    for pair, pair_frame in context.groupby(["detector_a", "detector_b"]):
        pair_label = PAIR_LABELS[pair]
        for eye in ("left", "right"):
            group = (
                pair_frame.loc[pair_frame["eye"].eq(eye)]
                .set_index("trial_family")
                .reindex(CONTEXT_ORDER)
            )
            ax.plot(
                positions + offsets[eye],
                group["median_fixation_jaccard"],
                marker=EYE_MARKERS[eye],
                linestyle=PAIR_LINESTYLES[pair],
                linewidth=1.3,
                label=f"{pair_label} — {eye}",
            )
    ax.set_xticks(
        positions,
        [CONTEXT_LABELS[value] for value in CONTEXT_ORDER],
        rotation=20,
        ha="right",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Median fixation-state Jaccard")
    ax.set_xlabel("Source-defined interaction context")
    ax.set_title("MCFW-Gaze: detector overlap remains structured across independent contexts")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, out)


def _load_artifact_inputs(
    srl_artifact: Path,
    mcfw_artifact: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _verify(srl_artifact, SRL_ARTIFACT_SHA256, "SRL artifact")
    _verify(mcfw_artifact, MCFW_ARTIFACT_SHA256, "MCFW artifact")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        srl_root = tmp_root / "srl"
        mcfw_root = tmp_root / "mcfw"
        srl_root.mkdir()
        mcfw_root.mkdir()
        _extract(srl_artifact, srl_root)
        _extract(mcfw_artifact, mcfw_root)
        srl = pd.read_csv(srl_root / "model_results" / "srl_multiverse_model_results.csv")
        context = pd.read_csv(mcfw_root / "summary" / "mcfw_context_profile.csv")
    if len(srl) != 144 or not srl["status"].eq("ok").all():
        raise ValueError("SRL frozen model universe is not the expected 144-row complete result.")
    return srl, context


def _load_snapshot_inputs(snapshot_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    srl_path = snapshot_dir / "srl_paper_figure_snapshot.csv"
    mcfw_path = snapshot_dir / "mcfw_context_jaccard_snapshot.csv"
    _verify(srl_path, SRL_SNAPSHOT_SHA256, "SRL paper snapshot")
    _verify(mcfw_path, MCFW_SNAPSHOT_SHA256, "MCFW paper snapshot")
    srl = pd.read_csv(srl_path)
    context = pd.read_csv(mcfw_path)
    if len(srl) != 144:
        raise ValueError("SRL paper snapshot must contain exactly 144 rows.")
    if set(srl["viewing_distance_cm"]) != {60.0, 65.0, 70.0}:
        raise ValueError("SRL paper snapshot viewing-distance set differs from the frozen universe.")
    if set(srl["detector_id"]) != set(DETECTOR_LABELS):
        raise ValueError("SRL paper snapshot detector set differs from the frozen universe.")
    return srl, context


def _validate_context(context: pd.DataFrame) -> None:
    if set(context["trial_family"]) != set(CONTEXT_ORDER):
        raise ValueError("MCFW context-family set differs from the frozen validation.")
    pairs = set(zip(context["detector_a"], context["detector_b"]))
    if pairs != set(PAIR_LABELS):
        raise ValueError("MCFW detector-pair set differs from the frozen validation.")
    if set(context["eye"]) != {"left", "right"}:
        raise ValueError("MCFW eye set differs from the frozen validation.")


def _verify_frozen_svgs(output_dir: Path) -> None:
    mismatches: list[str] = []
    for filename, expected in SVG_SHA256.items():
        actual = _sha256(output_dir / filename)
        if actual != expected:
            mismatches.append(f"{filename}: observed {actual}; expected {expected}")
    if mismatches:
        raise ValueError("Frozen SVG hash mismatch:\n" + "\n".join(mismatches))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--srl-artifact", type=Path)
    parser.add_argument("--mcfw-artifact", type=Path)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        help="Directory containing the two committed compact paper snapshots.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    _verify_renderer_versions()

    artifact_mode = args.srl_artifact is not None or args.mcfw_artifact is not None
    snapshot_mode = args.snapshot_dir is not None
    if artifact_mode == snapshot_mode:
        parser.error(
            "Choose exactly one input mode: both --srl-artifact/--mcfw-artifact, "
            "or --snapshot-dir."
        )
    if artifact_mode:
        if args.srl_artifact is None or args.mcfw_artifact is None:
            parser.error("Artifact mode requires both --srl-artifact and --mcfw-artifact.")
        srl, context = _load_artifact_inputs(args.srl_artifact, args.mcfw_artifact)
    else:
        assert args.snapshot_dir is not None
        srl, context = _load_snapshot_inputs(args.snapshot_dir)

    _validate_context(context)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure_srl_specification_curve(
        srl,
        args.output_dir / "fig1_srl_specification_curve.svg",
    )
    figure_srl_detector_distance(
        srl,
        args.output_dir / "fig2_srl_detector_distance.svg",
    )
    figure_mcfw_context_jaccard(
        context,
        args.output_dir / "fig3_mcfw_context_jaccard.svg",
    )
    _verify_frozen_svgs(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
