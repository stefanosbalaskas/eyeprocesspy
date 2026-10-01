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

The SVG renderer uses a fixed Matplotlib SVG hash salt and suppresses the generation-date metadata, so repeated generation from the same artifacts is byte-deterministic.

## Frozen generated-figure hashes

- `fig1_srl_specification_curve.svg`: `633c01374b9f4384289a089526b98456fa36b918d0d1dd64d57e0d627e856fec`
- `fig2_srl_detector_distance.svg`: `ec2b5e3614f13cbba9de8971a5ae459e69a3f1a9eb204beeccd39c1e90f427fb`
- `fig3_mcfw_context_jaccard.svg`: `1f5f6ed4541f42b334aef82657ab1025955826d49a0e669a6f252b61b1ca31a6`

The SVG files are generated products; the source of truth is the figure generator plus the exact canonical workflow artifacts and their hashes.

## Manuscript draft

`results_draft.md` is the current paper-facing Results section. It intentionally separates:

1. **SRL claim propagation** — measurement choices alter the direction and magnitude of a substantive Prompt effect estimate while all 95% intervals include the null; and
2. **MCFW-Gaze measurement generalization** — detector-dependent event representations persist across independent hardware, sampling rate, and interaction contexts.

The draft does not interpret specification frequencies as probabilities that an effect is true, and it does not designate a detector, viewing distance, eye, AOI convention, or quality branch as the scientifically correct choice.
