# eyeprocesspy examples

These scripts are deliberately deterministic and require no private research data.

```bash
python -m pip install -e ".[plots]"
python examples/core_gallery.py
python examples/advanced_gallery.py
```

Both scripts write SVG figures to `gallery-output/`.

- `core_gallery.py` constructs a canonical `EyeDataset`, validates it, and demonstrates the core gaze/AOI/pupil plotting surface.
- `advanced_gallery.py` demonstrates repeated-measure reliability, calibration-error modeling, probabilistic AOI assignment, sampling irregularity, and IRT diagnostic plots.

The same figure families are represented in the documentation gallery.
