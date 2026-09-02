# API and plotting reference

The `eyeprocesspy` public surface follows the frozen eyeprocess 0.11.1 API ledger and exposes a broad Python namespace for reproducible behavioral-process research.

[Browse the generated API](api.md){ .md-button .md-button--primary }
[Plotting reference](plotting.md){ .md-button }
[Visual gallery](../gallery.md){ .md-button }

## Scientific families

| Family | Representative surface |
| --- | --- |
| **Core data contracts** | `EyeDataset`, schemas, canonical tables, provenance, coordinate spaces, timebase |
| **Import and adapters** | generic readers, adapter registry, folder workflows, vendor detection |
| **Gazepoint** | gaze/fixation/event/biometric readers, pairing, real-export validation, workflow helpers |
| **Preprocessing and features** | fixations, saccades, AOIs, dwell, scanpaths, transitions, entropy, wide/trial feature tables |
| **Pupil and multimodal** | pupil preprocessing, functional pupil, missingness, synchronized/staged multimodal workflows |
| **Process quality** | reliability, Bland–Altman, sampling irregularity, calibration error, gaze precision, quality profiles |
| **AOI uncertainty** | calibration propagation, probabilistic AOIs, compositional AOIs, sensitivity/boundary uncertainty |
| **Psychometrics and IRT** | information, scoring, fit, Q3, DIF/DTF, dynamic/process-informed/advanced IRT |
| **Measurement intelligence** | measure registry, guardrails, linking, norms, fairness, item-bank optimization |
| **Validation** | recovery, SBC-style evidence, stress tests, negative controls, grouped validation, atlases |
| **Reproducibility** | benchmarks, provenance manifests, software-paper evidence, parity/release validation |
| **Plots and reporting** | Matplotlib graphics, diagnostics, evidence visualizations, publication-facing reporting helpers |

## Plotting surface

The plotting reference documents the major Matplotlib families and how to save, combine, and audit figures. Core plots return standard Matplotlib axes and commonly expose the underlying plotted data through `ax.eyeprocess_plot_data`; matrix plots can also expose `ax.eyeprocess_plot_matrix`.

[Open plotting reference](plotting.md){ .md-button .md-button--primary }

## How to find the right function

- Start with [Getting started](../getting-started.md) for the canonical workflow.
- Use [Python-native guides](../guides/index.md) when you know the research problem but not the function names.
- Use [Runnable examples](../examples/index.md) for compact scripts.
- Use the [88-article library](../articles/index.md) for full workflow/parity context.
- Use [Parity and validation](../parity-and-validation.md) for scientific fidelity and documented R/Python differences.

## Reference breadth

The frozen parity surface contains **1,182 resolved APIs** spanning import, data contracts, preprocessing, gaze/AOI analysis, pupil workflows, process measurement, IRT, validation, reproducibility, interoperability, governance, storage, scientific plots, and reporting.

The generated API page is intentionally comprehensive; the curated guides and plotting reference provide the higher-level navigation layer.
