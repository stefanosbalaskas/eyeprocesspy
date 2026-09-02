# eyeprocesspy examples

These scripts are deliberately deterministic and require no private research data. They are intended to be executable reference workflows rather than pseudocode.

## Install the plotting dependencies

```bash
python -m pip install -e ".[plots]"
```

If you are using the manual wheel instead of a checkout, install the wheel first and then install Matplotlib:

```bash
python -m pip install eyeprocesspy-0.1.0-py3-none-any.whl
python -m pip install "matplotlib>=3.9"
```

## End-to-end research workflows

```bash
python examples/complete_workflow.py
python examples/calibration_probabilistic_aoi.py
python examples/process_reliability.py
python examples/irt_diagnostics.py
```

- `complete_workflow.py` — constructs and validates a canonical `EyeDataset`, derives scanpaths, transitions and gaze entropy, renders auditable figures, and inspects provenance.
- `calibration_probabilistic_aoi.py` — estimates empirical calibration error, summarizes the uncertainty ellipse, propagates error to AOI membership and plots probabilistic assignments.
- `process_reliability.py` — demonstrates repeated-measure ICC, Bland–Altman agreement, temporal stability and a reliability plot.
- `irt_diagnostics.py` — generates information/SEM, item-fit and DIF diagnostic plots.

## Gallery generators

```bash
python examples/core_gallery.py
python examples/advanced_gallery.py
```

- `core_gallery.py` constructs a complete synthetic `EyeDataset`, validates it, and demonstrates the core gaze/AOI/pupil plotting surface.
- `advanced_gallery.py` demonstrates repeated-measure reliability, calibration-error modeling, probabilistic AOI assignment, sampling irregularity, and IRT diagnostic plots.

The gallery scripts write SVG figures to `gallery-output/`; the focused workflows write to `workflow-output/`.

## Documentation

- [Runnable examples](../docs/examples/index.md)
- [Practical cookbook](../docs/cookbook.md)
- [Visual gallery](../docs/gallery.md)
- [Python-native guides](../docs/guides/)
- [88-article workflow library](../docs/articles/)

The examples demonstrate computational workflows, not automatic psychological interpretations. Reliability is not construct validity, probabilistic AOI membership is measurement uncertainty rather than probability of attention, and DIF is evidence about measurement invariance rather than automatic proof of unfairness.
