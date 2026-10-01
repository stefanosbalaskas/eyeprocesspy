# CHI paper-facing evidence package

This directory is the manuscript workspace for the gaze-claim robustness study. It contains **presentation and writing artifacts only**. Nothing in this layer refits the SRL models, reruns MCFW detectors, changes a frozen measurement decision, or selects a preferred branch after result inspection.

## Source hierarchy

The manuscript should be edited from the section-level sources below rather than by copying prose back into analysis scripts:

1. `paper_outline.md` — working title, thesis, RQs, contribution claims, abstract, paper structure, and explicit claims to avoid.
2. `literature_positioning.csv` — evidence matrix recording what each positioning source establishes, what it does **not** establish, and its role in the novelty argument.
3. `literature_references.bib` — locked minimal bibliography corresponding to the positioning matrix.
4. `frontmatter_conclusion.md` — title, abstract, keywords, and Conclusion source.
5. `introduction_related_work.md` — complete Introduction and Related Work draft.
6. `methods_draft.md` — claim-centered robustness framework plus frozen SRL and MCFW methods/provenance.
7. `results_draft.md` — frozen empirical Results and figure captions.
8. `discussion_draft.md` — interpretation, HCI implications, limitations, and scope boundaries.
9. `claims_audit.md` — evidence-to-claim and overclaim-control matrix for submission editing.
10. `assemble_manuscript.py` — deterministic assembler that creates `manuscript_draft.md` from the section sources; the generated manuscript is a build product rather than a second hand-edited source.
11. `make_publication_figures.py` — deterministic presentation-only figure generator from canonical workflow artifacts.

This hierarchy keeps analysis, evidence positioning, manuscript prose, and generated presentation products distinct.

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

## Literature-positioning contract

The novelty claim is deliberately narrower than "multiverse analysis for eye tracking." Existing work already establishes multiverse/specification analysis, HCI multiverse tooling, temporal-window sensitivity, eye-movement cleaning/analysis multiverses, detector disagreement, AOI uncertainty, gaze-data quality, and reporting standards.

The paper's intended methodological extension is the **mapping from a declared cross-stage eye-tracking measurement-decision space to the stability of one prespecified downstream HCI claim**, with planned-denominator accountability and an independent raw-signal validation case.

Every source in `literature_positioning.csv` includes a `what_it_does_not_establish` field so that detector, AOI, quality, or reporting evidence is not silently promoted into evidence for full claim propagation. `claims_audit.md` applies the same discipline to the manuscript's own headline claims.

## Manuscript assembly

Run:

```bash
python research/gaze_claim_robustness/paper/assemble_manuscript.py
```

The assembler reads the section-level Markdown sources in fixed order, moves the Conclusion from the shared front-matter source to the end, and writes `manuscript_draft.md`. It performs no empirical computation and reads no result artifact directly.

## Figure generation

`make_publication_figures.py` requires the two canonical GitHub Actions ZIP artifacts and verifies their exact SHA-256 values before reading any result table.

Example:

```bash
python research/gaze_claim_robustness/paper/make_publication_figures.py \
  --srl-artifact /path/to/srl-frozen-model-universe-5fad0f192067ca74277c4b5f2bec462514395124.zip \
  --mcfw-artifact /path/to/mcfw-detector-validation-878e351a82a48665990955853c55b2315b10fd27.zip \
  --output-dir research/gaze_claim_robustness/paper/figures
```

The SVG renderer uses a fixed Matplotlib SVG hash salt, leaves text editable as SVG text, and suppresses generation-date metadata. Repeated generation from the same canonical artifacts is therefore byte-deterministic.

## Frozen generated-figure hashes

- `fig1_srl_specification_curve.svg`: `4c2531bdc716d51edc91a71196df61383a046c06a27ac04fb73301b002a4bcbc`
- `fig2_srl_detector_distance.svg`: `a6ad1b378f624f652765fefbdb6aa14174691e67009562348ba54dc83f1d9666`
- `fig3_mcfw_context_jaccard.svg`: `8123b2352966e5eca4773bac6a525bc16786685f1a3c052a820c88208b05cabe`

The SVG files are generated products; the source of truth is the figure generator plus the exact canonical workflow artifacts and their hashes.

## Manuscript interpretation boundary

The manuscript intentionally separates:

1. **SRL claim propagation** — measurement choices alter the direction and magnitude of a substantive Prompt effect estimate while all 95% intervals include the null; and
2. **MCFW-Gaze measurement generalization** — detector-dependent event representations persist across independent hardware, sampling rate, and interaction contexts.

The paper does not interpret specification frequencies as probabilities that an effect is true, does not designate a detector/viewing distance/eye/AOI/quality branch as scientifically correct, and does not describe MCFW-Gaze as a replication of the SRL Prompt effect.
