# CHI paper-facing evidence package

This directory is the manuscript workspace for the gaze-claim robustness study. It contains **presentation and writing artifacts only**. Nothing in this layer refits the SRL models, reruns MCFW detectors, changes a frozen measurement decision, or selects a preferred branch after result inspection.

## Source hierarchy

The manuscript should be edited from the section-level sources below rather than by copying prose back into analysis scripts:

1. `paper_outline.md` — working title, thesis, RQs, contribution claims, paper structure, and explicit claims to avoid.
2. `literature_positioning.csv` — evidence matrix recording what each positioning source establishes, what it does **not** establish, and its role in the novelty argument.
3. `literature_references.bib` — locked bibliography for the paper-facing sources.
4. `chi_format_notes.md` — dated current-CHI format/word-limit benchmark and submission checks.
5. `frontmatter_conclusion.md` — authoritative title, abstract, keywords, and Conclusion source.
6. `introduction_related_work.md` — complete Introduction and Related Work draft.
7. `methods_draft.md` — claim-centered robustness framework plus frozen SRL and MCFW methods/provenance.
8. `results_draft.md` — frozen empirical Results and figure captions.
9. `discussion_draft.md` — interpretation, HCI implications, limitations, and scope boundaries.
10. `claims_audit.md` — evidence-to-claim and overclaim-control matrix for submission editing.
11. `chi_reviewer_simulation.md` — author-facing CHI review simulation, including the 2026 novelty stress-test and likely reviewer objections.
12. `assemble_manuscript.py` — deterministic Markdown assembler; `manuscript_draft.md` is a build product, not a second hand-edited source.
13. `manuscript_metrics.py` — mechanical abstract/main-text word-count gate.
14. `make_publication_figures.py` — presentation-only renderer for the three frozen figures.
15. `build_acm_review.py` — anonymous one-column `acmart` review-source generator with BibTeX citations, accessibility descriptions, real figures, ACM running-title handling, and CCS concepts.

The GitHub Actions workflow `.github/workflows/gaze-claim-paper.yml` exercises the full paper path: manuscript assembly, length checks, frozen-figure regeneration, Pandoc conversion, anonymous ACM TeX generation, PDF compilation, and artifact upload.

## Canonical empirical inputs

### SRL claim-propagation universe

- workflow run: `36708103076`
- analysis head: `5fad0f192067ca74277c4b5f2bec462514395124`
- workflow artifact: `srl-frozen-model-universe-5fad0f192067ca74277c4b5f2bec462514395124`
- artifact SHA-256: `b125b071a9a1bf68f5ba3c11f1b8fdeedca1195de94a64e27155af79d7ae342d`
- planned and successful model specifications: 144/144

### MCFW-Gaze measurement-generalization universe

- workflow run: `36776432517`
- analysis head: `878e351a82a48665990955853c55b2315b10fd27`
- workflow artifact: `mcfw-detector-validation-878e351a82a48665990955853c55b2315b10fd27`
- artifact SHA-256: `ed2af7a036d1fb9a62b5226b71b6f02d007f87a9d23b1970828f9c9742b93020`
- planned detector-summary rows: 28,170
- planned detector-pair rows: 28,170

## Literature-positioning and claim contract

The novelty claim is deliberately narrower than "multiverse analysis for eye tracking." Existing work already establishes multiverse/specification analysis, HCI multiverse tooling, temporal-window sensitivity, eye-movement cleaning/analysis multiverses, detector disagreement, AOI uncertainty, gaze-data quality, reporting standards, and downstream gaze-based relationship sensitivity under alternative preprocessing pipelines. The September 2026 *Workload Multiverse* study by Schindler and Onnasch is treated explicitly as a close downstream precedent rather than omitted from the novelty argument.

The paper's intended methodological extension is the **mapping from a declared cross-stage eye-tracking measurement-decision space to the stability of one prespecified downstream HCI claim**, with explicit branch-type distinctions, planned-denominator accountability, multidimensional interpretation, and an independent raw-signal validation case.

The manuscript distinguishes three branch classes explicitly:

1. **measurement-definition alternatives** — detector, analyzed eye, visual-angle geometry, and AOI convention;
2. **sample-definition sensitivities** — quality rule and cohort, which alter empirical support and can change the formal target population;
3. **estimand-changing analyses** — analyses that alter the scientific quantity itself and therefore remain outside the primary claim-stability denominator.

A branch enters the declared universe only when its rationale is source- or literature-grounded, fully specified before focal-result inspection, executable without outcome-dependent tuning or silent repair, and comparable under the same focal claim contract. Specification counts describe this frozen grid and are not probabilities or invariant robustness percentages.

Every source in `literature_positioning.csv` includes a `what_it_does_not_establish` field so that detector, AOI, quality, reporting, or prior multiverse evidence is not silently promoted into support for a stronger novelty claim. `claims_audit.md` applies the same discipline to the manuscript's own headline claims, and `chi_reviewer_simulation.md` stress-tests those claims against likely CHI objections.

## Current CHI-format benchmark

As verified on 2026-10-01, CHI 2027 uses single-column anonymous review submissions, encourages approximately 5,000–8,000 words, and caps abstracts at 150 words. The current mechanical manuscript build reports:

- abstract: **140 words**;
- main text excluding headings, code blocks, and figure captions: **7,505 words**;
- assembled Markdown including front matter and other counted material: **8,139 words**.

The CHI 2027 initial deadline (2026-09-10 AoE) has passed, so these constraints are treated as the current formatting benchmark unless the work corresponds to an already-submitted 2027 paper. The actual target-cycle call must be re-verified before submission. See `chi_format_notes.md` for the dated format check.

## Manuscript assembly and ACM review build

Markdown assembly:

```bash
python research/gaze_claim_robustness/paper/assemble_manuscript.py
```

The assembler reads the section-level Markdown sources in fixed order, moves the Conclusion from the shared front-matter source to the end, and writes `manuscript_draft.md`. It performs no empirical computation and reads no result artifact directly.

The automated paper workflow then builds an anonymous review manuscript using:

```latex
\documentclass[manuscript,review,anonymous]{acmart}
```

`build_acm_review.py` converts the assembled manuscript to BibTeX-backed ACM LaTeX, keeps the full paper title on page 1 while using `From Gaze Signals to HCI Claims` as the shorter running title, inserts the real publication figures with `\Description{...}` accessibility text, emits the HCI design/evaluation and empirical-studies CCS concepts, and compiles the review PDF in CI. The ACM reference strip remains suppressed in the anonymous pre-eRights build so the local manuscript does not manufacture conference/DOI metadata that have not yet been assigned.

## Durable frozen figure inputs

The publication figures can still be reproduced directly from the two canonical workflow ZIP artifacts. For durable paper builds after those Actions artifacts expire, the repository also stores exact compact presentation snapshots extracted from the frozen results:

- `results/srl_paper_figure_snapshot.csv` — SHA-256 `706b57051efe074700d3b3dbc693bece3d3747da2d5940565f4c93f6a9dc6193`;
- `results/mcfw_context_jaccard_snapshot.csv` — SHA-256 `d0c7471199e5a6242c1f51f872f4f7a3a915adb8a91749adea7fe0ff60ee235f`.

These snapshots introduce no new statistical computation; they are exact paper-facing extracts of the already-frozen empirical outputs.

Canonical-artifact mode:

```bash
python research/gaze_claim_robustness/paper/make_publication_figures.py \
  --srl-artifact /path/to/srl-frozen-model-universe-5fad0f192067ca74277c4b5f2bec462514395124.zip \
  --mcfw-artifact /path/to/mcfw-detector-validation-878e351a82a48665990955853c55b2315b10fd27.zip \
  --output-dir paper-figures
```

Durable snapshot mode:

```bash
python research/gaze_claim_robustness/paper/make_publication_figures.py \
  --snapshot-dir research/gaze_claim_robustness/results \
  --output-dir paper-figures
```

## Frozen renderer and generated-figure hashes

Byte-identical SVG verification is explicitly tied to the renderer environment that produced the frozen figure bytes:

- Matplotlib `3.10.8`;
- pandas `2.2.3`;
- NumPy `2.3.5`;
- Matplotlib SVG hash salt `gaze-claim-robustness-chi`;
- SVG text retained as editable text;
- generation-date metadata suppressed.

The renderer refuses byte-hash verification under a different declared renderer version. This keeps scientific input provenance separate from presentation-library versioning.

Frozen SVG SHA-256 values:

- `fig1_srl_specification_curve.svg`: `4c2531bdc716d51edc91a71196df61383a046c06a27ac04fb73301b002a4bcbc`;
- `fig2_srl_detector_distance.svg`: `a6ad1b378f624f652765fefbdb6aa14174691e67009562348ba54dc83f1d9666`;
- `fig3_mcfw_context_jaccard.svg`: `8123b2352966e5eca4773bac6a525bc16786685f1a3c052a820c88208b05cabe`.

PDF sidecars are generated from the same figure objects solely for `acmart` typesetting. The SVG hashes remain the frozen presentation identity.

## Manuscript interpretation boundary

The manuscript intentionally separates:

1. **SRL claim propagation** — measurement choices alter the direction and magnitude of a substantive Prompt effect estimate while all 95% intervals include the null; and
2. **MCFW-Gaze measurement generalization** — detector-dependent event representations persist across independent hardware, sampling rate, and interaction contexts.

The paper does not interpret specification frequencies as probabilities that an effect is true or as invariant robustness percentages, does not designate a detector/viewing distance/eye/AOI/quality branch as scientifically correct, does not claim strict formal estimand identity across sample-definition sensitivities, does not describe MCFW-Gaze as a replication of the SRL Prompt effect, and does not claim to be the first demonstration that preprocessing can alter a downstream gaze-based relationship.
