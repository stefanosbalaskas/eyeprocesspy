# CHI paper-facing evidence package

This directory is the manuscript workspace for the gaze-claim robustness study. It contains **presentation, writing, and review-submission artifacts only**. Nothing in this layer refits the SRL models, reruns MCFW detectors, changes a frozen measurement decision, or selects a preferred branch after result inspection.

## Source hierarchy

Edit the section-level sources rather than copying prose back into analysis scripts:

1. `paper_outline.md` — working title, thesis, RQs, contribution claims, paper structure, and explicit claims to avoid.
2. `literature_positioning.csv` — evidence matrix recording what each positioning source establishes, what it does **not** establish, and its role in the novelty argument.
3. `literature_references.bib` — locked bibliography for paper-facing sources.
4. `chi_format_notes.md` — dated CHI 2027 format/submission benchmark.
5. `chi2027_submission_compliance.md` — requirement-by-requirement CHI 2027 compliance ledger, separating build-verifiable requirements from author/PCS actions.
6. `chi2027_pcs_author_checks.md` — author-only PCS checklist, including ORCID/DBLP, frozen author/title metadata, conflicts, descriptors, review-responsibility slots, ethics note, and exact-upload checks.
7. `frontmatter_conclusion.md` — authoritative title, abstract, keywords, and Conclusion source.
8. `introduction_related_work.md` — complete Introduction and Related Work draft.
9. `methods_draft.md` — claim-centered robustness framework plus frozen SRL/MCFW methods, secondary-data ethics context, and material AI-use disclosure.
10. `results_draft.md` — frozen empirical Results and figure captions.
11. `discussion_draft.md` — interpretation, HCI implications, limitations, and scope boundaries.
12. `claims_audit.md` — evidence-to-claim and overclaim-control matrix.
13. `chi_reviewer_simulation.md` — author-facing CHI review simulation including the 2026 novelty stress-test.
14. `assemble_manuscript.py` — deterministic Markdown assembler; `manuscript_draft.md` is a build product, not a second editable source.
15. `manuscript_metrics.py` — mechanical abstract/main-text word-count gate.
16. `make_publication_figures.py` — presentation-only renderer for the three frozen figures.
17. `build_acm_review.py` — official anonymous one-column `acmart` review-source generator with BibTeX citations, real figures, accessibility descriptions, ACM running-title handling, and CCS concepts.
18. `build_anonymized_supplement.py` — deterministic builder for the single anonymized non-video review ZIP with README and manifest.

The workflow `.github/workflows/gaze-claim-paper.yml` exercises the paper path: manuscript assembly, length checks, frozen-figure regeneration, supplement construction, Pandoc conversion, official anonymous ACM TeX generation, PDF compilation, anonymity/content checks, and artifact upload.

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

The intended methodological extension is the **mapping from a declared cross-stage eye-tracking measurement-decision space to the stability of one prespecified downstream HCI claim**, with explicit branch-type distinctions, planned-denominator accountability, multidimensional interpretation, and an independent raw-signal validation case.

The manuscript distinguishes:

1. **measurement-definition alternatives** — detector, analyzed eye, visual-angle geometry, and AOI convention;
2. **sample-definition sensitivities** — quality rule and cohort, which alter empirical support and can change the formal target population;
3. **estimand-changing analyses** — analyses that alter the scientific quantity itself and remain outside the primary claim-stability denominator.

A branch enters the declared universe only when its rationale is source- or literature-grounded, fully specified before focal-result inspection, executable without outcome-dependent tuning or silent repair, and comparable under the same focal claim contract. Specification counts describe this frozen grid and are not probabilities or invariant robustness percentages.

## Current CHI 2027 benchmark and calendar boundary

As verified on 2026-10-01, CHI 2027 Papers uses an anonymous, single-column ACM review submission and the official LaTeX review source is:

```latex
\documentclass[manuscript,review,anonymous]{acmart}
```

CHI encourages approximately 5,000–8,000 words and caps abstracts at 150 words. The current mechanical manuscript build reports:

- abstract: **140 words**;
- main text excluding headings, code blocks, and figure captions: **7,578 words**;
- assembled Markdown including front matter and other counted material: **8,213 words**.

The CHI 2027 initial Papers deadline was **2026-09-10 AoE** and has passed. These rules therefore apply to an already-existing CHI 2027 Papers submission/resubmission; they cannot create a new initial CHI 2027 Papers submission on 2026-10-01. The live PCS stage and official instructions override this repository if they differ.

## Review-build and anonymization boundary

`build_acm_review.py` converts the assembled manuscript to BibTeX-backed ACM LaTeX, inserts the real publication figures with explicit `\Description{...}` text, emits CCS concepts, keeps the long page-one title with a short running title, and strips paper-facing workflow/commit identifiers from the review narrative. The Actions workflow scans extracted PDF text for known author/repository/workflow identifiers and verifies that the material ChatGPT-use disclosure remains present.

`build_anonymized_supplement.py` emits one deterministic non-video review ZIP with `README.txt` and `MANIFEST.tsv`. It contains selected decision contracts, core analysis mechanics, and compact frozen result snapshots, but no raw participant archives. The builder rejects known author/repository/workflow identifiers before writing the archive.

The repository can verify source-level accessibility measures (semantic structure, figure descriptions, marker/line-style redundancy, language/PDF metadata intent), but **full PDF accessibility remains an author final-action gate**. The exact PDF uploaded to PCS should receive an accessibility Full Check/remediation; the official `acmart` review class must not be replaced merely to obtain automatic tagging.

## Durable frozen figure inputs

The figures can be regenerated directly from the canonical workflow ZIP artifacts. For durable builds after Actions artifacts expire, the repository stores exact compact presentation snapshots extracted from the frozen results:

- `results/srl_paper_figure_snapshot.csv` — SHA-256 `706b57051efe074700d3b3dbc693bece3d3747da2d5940565f4c93f6a9dc6193`;
- `results/mcfw_context_jaccard_snapshot.csv` — SHA-256 `d0c7471199e5a6242c1f51f872f4f7a3a915adb8a91749adea7fe0ff60ee235f`.

These snapshots introduce no new statistical computation; they are exact paper-facing extracts of the already-frozen empirical outputs.

## Frozen renderer and generated-figure hashes

Byte-identical SVG verification is tied to the renderer environment that produced the frozen presentation bytes:

- Matplotlib `3.10.8`;
- pandas `2.2.3`;
- NumPy `2.3.5`;
- SVG hash salt `gaze-claim-robustness-chi`;
- SVG text retained as editable text;
- generation-date metadata suppressed.

The accessibility revision preserves non-color encodings using marker shape and/or line style. Current frozen SVG SHA-256 values are:

- `fig1_srl_specification_curve.svg`: `4c2531bdc716d51edc91a71196df61383a046c06a27ac04fb73301b002a4bcbc`;
- `fig2_srl_detector_distance.svg`: `bc25f54e554dbe9bff20852e1c63e82578c8fa92b69f605c2c6866ccfc21249c`;
- `fig3_mcfw_context_jaccard.svg`: `aaf07ef4cc7e3f72352fed16b90d7d9e340527582ed273adb7427d57c5b24a0b`.

PDF sidecars are emitted from the same figure objects solely for manuscript typesetting.

## Manuscript interpretation boundary

The paper intentionally separates:

1. **SRL claim propagation** — measurement choices alter the direction and magnitude of a substantive Prompt contrast estimate while all 95% intervals include the null; and
2. **MCFW-Gaze measurement generalization** — detector-dependent event representations persist across independent hardware, sampling rate, and interaction contexts.

The paper does not interpret specification frequencies as probabilities that an effect is true or as invariant robustness percentages, does not designate a detector/viewing distance/eye/AOI/quality branch as scientifically correct, does not claim strict formal estimand identity across sample-definition sensitivities, does not describe MCFW-Gaze as a replication of the SRL Prompt contrast, and does not claim to be the first demonstration that preprocessing can alter a downstream gaze-based relationship.

## Final submission boundary

A green repository build is necessary but not sufficient for CHI submission completeness. Before any CHI 2027 upload/resubmission, complete `chi2027_submission_compliance.md` and `chi2027_pcs_author_checks.md` against the live PCS record. In particular, verify the already-submitted paper's stage, frozen title/author list/affiliations, every author's ORCID and DBLP field, conflicts, descriptors, all four review-responsibility slots, related/concurrent-work disclosures, the secondary-data ethics note, and accessibility/anonymity of the exact uploaded PDF/ZIP.
