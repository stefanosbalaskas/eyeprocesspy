# CHI paper-facing evidence package

This directory contains **presentation-only** artifacts derived from the two frozen empirical universes. Nothing in this layer refits the SRL models, reruns MCFW detectors, changes a measurement decision, or selects a preferred branch after results inspection.

## Canonical inputs

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

## Manuscript draft

`results_draft.md` is the current paper-facing Results section. It intentionally separates:

1. **SRL claim propagation** — measurement choices alter the direction and magnitude of a substantive Prompt effect estimate while all 95% intervals include the null; and
2. **MCFW-Gaze measurement generalization** — detector-dependent event representations persist across independent hardware, sampling rate, and interaction contexts.

The draft does not interpret specification frequencies as probabilities that an effect is true, and it does not designate a detector, viewing distance, eye, AOI convention, or quality branch as the scientifically correct choice.
