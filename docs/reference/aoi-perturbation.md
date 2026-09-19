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

See [AOI perturbation and uncertainty analysis](../guides/aoi-uncertainty/index.md) for design, interpretation, limitations, and reporting guidance.
## Practical guidance

Use the [AOI robustness decision clinic](../guides/aoi-uncertainty/index.mddecision-clinic.md) to interpret assignment/model stability patterns, the [fixation-level worked example](../examples/aoi-perturbation-sensitivity.md) for end-to-end model propagation, and the [sample-level worked example](../examples/sample-level-aoi-sensitivity.md) when geometry sensitivity is evaluated directly on gaze samples.

## Failure handling

Use [AOI perturbation troubleshooting](../guides/aoi-uncertainty/index.mdtroubleshooting.md) for geometry failures, overlap ambiguity, zero-versus-missing denominators, callback exceptions, non-convergence, and model-N changes. The [worked failure clinic](../examples/aoi-perturbation-failure-clinic.md) executes these states on synthetic data.

## Planning

Use the [AOI sensitivity analysis plan](../guides/aoi-uncertainty/index.mdanalysis-plan.md) before running branch-specific models. The [worked filled plan](../examples/aoi-analysis-plan-worked.md) maps each planning decision to the public API.

## Evidence bundle

Use the [AOI robustness reporting bundle](../guides/aoi-uncertainty/index.mdreporting-bundle.md) to package the analysis plan, branch audit, assignment/model summaries, failures, provenance, report, and figures. The [worked example](../examples/aoi-reporting-bundle.md) provides a deterministic helper and SHA-256 manifest without adding a new public API.
