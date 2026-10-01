#!/usr/bin/env python3
"""Generate the frozen CHI paper figures from canonical workflow artifacts.

This script is presentation-only. It does not refit models, rerun detectors, or
change either frozen empirical universe. It verifies the canonical GitHub
Actions artifact SHA-256 values before reading their CSV outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["svg.hashsalt"] = "gaze-claim-robustness-chi"

SRL_ARTIFACT_SHA256 = "b125b071a9a1bf68f5ba3c11f1b8fdeedca1195de94a64e27155af79d7ae342d"
MCFW_ARTIFACT_SHA256 = "ed2af7a036d1fb9a62b5226b71b6f02d007f87a9d23b1970828f9c9742b93020"

DETECTOR_LABELS = {
    "ivt_30_100_simple": "I-VT 30°/s, 100 ms",
    "ivt_40_50_simple": "I-VT 40°/s, 50 ms",
    "idt_1_100": "I-DT 1°, 100 ms",
}
PAIR_LABELS = {
    ("ivt_30_100_simple", "ivt_40_50_simple"): "I-VT 30 vs I-VT 40",
    ("ivt_30_100_simple", "idt_1_100"): "I-VT 30 vs I-DT",
    ("ivt_40_50_simple", "idt_1_100"): "I-VT 40 vs I-DT",
}
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
        raise ValueError(f"{label} artifact SHA-256 mismatch: {actual}")


def _extract(zip_path: Path, target: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)


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
    fig.savefig(out, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


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
        ax.plot(
            group["viewing_distance_cm"],
            group["median_rr"],
            marker="o",
            linewidth=1.8,
            label=DETECTOR_LABELS[detector],
        )
        ax.vlines(
            group["viewing_distance_cm"],
            group["q25_rr"],
            group["q75_rr"],
            linewidth=1.2,
            alpha=0.8,
        )
    ax.axhline(1, linewidth=1, linestyle="--")
    ax.set_xticks([60, 65, 70])
    ax.set_xlabel("Assumed viewing distance (cm)")
    ax.set_ylabel("Median Prompt rate ratio")
    ax.set_title("SRL: detector and visual-angle geometry jointly shift the effect estimate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def figure_mcfw_context_jaccard(context: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    positions = np.arange(len(CONTEXT_ORDER))
    offsets = {"left": -0.10, "right": 0.10}
    markers = {"left": "o", "right": "s"}
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
                marker=markers[eye],
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
    fig.savefig(out, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--srl-artifact", required=True, type=Path)
    parser.add_argument("--mcfw-artifact", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    _verify(args.srl_artifact, SRL_ARTIFACT_SHA256, "SRL")
    _verify(args.mcfw_artifact, MCFW_ARTIFACT_SHA256, "MCFW")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        srl_root = tmp_root / "srl"
        mcfw_root = tmp_root / "mcfw"
        srl_root.mkdir()
        mcfw_root.mkdir()
        _extract(args.srl_artifact, srl_root)
        _extract(args.mcfw_artifact, mcfw_root)

        srl = pd.read_csv(srl_root / "model_results" / "srl_multiverse_model_results.csv")
        context = pd.read_csv(mcfw_root / "summary" / "mcfw_context_profile.csv")

        if len(srl) != 144 or not srl["status"].eq("ok").all():
            raise ValueError("SRL frozen model universe is not the expected 144-row complete result.")
        if set(context["trial_family"]) != set(CONTEXT_ORDER):
            raise ValueError("MCFW context-family set differs from the frozen validation.")

        figure_srl_specification_curve(srl, args.output_dir / "fig1_srl_specification_curve.svg")
        figure_srl_detector_distance(srl, args.output_dir / "fig2_srl_detector_distance.svg")
        figure_mcfw_context_jaccard(context, args.output_dir / "fig3_mcfw_context_jaccard.svg")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
