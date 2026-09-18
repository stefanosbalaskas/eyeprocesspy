# AOI perturbation API

The AOI perturbation surface is vendor-neutral and exported from the `eyeprocesspy` package front door.

::: eyeprocesspy.aoi_perturbation
    options:
      members: true
      inherited_members: false
      show_root_heading: true
      show_root_full_path: false
      show_source: true
      members_order: source
      separate_signature: true
      show_signature_annotations: true

## Core contract

Use `aoi_perturbation_spec()` for one validated branch or `create_aoi_perturbation_grid()` for a declared sensitivity set.

`run_aoi_sensitivity_analysis()` performs geometry perturbation → AOI remapping → feature recomputation → optional fixed model callback → assignment/model stability summaries while retaining failures and provenance.

See [AOI perturbation and uncertainty analysis](../guides/aoi-uncertainty/) for design, interpretation, limitations, and reporting guidance.