# Governed end-to-end analysis pipelines

The governed pipeline layer links import, measurement quality, preprocessing, feature construction, modelling, diagnostics, sensitivity analysis, and reporting while preserving researcher-declared choices. Pipeline steps are explicit functions with declared dependencies; `eyeprocesspy` does not silently choose preprocessing or statistical specifications.

```python
import eyeprocesspy as ep

spec = ep.eye_analysis_spec(
    blink_correction="linear",
    pupil_baseline=(-500, 0),
    fixation_algorithm="ivt",
    aoi_rule="probabilistic",
)

pipeline = ep.eye_analysis_pipeline([
    ep.eye_pipeline_step("import_data", read_fun),
    ep.eye_pipeline_step("quality", quality_fun, requires="import_data"),
    ep.eye_pipeline_step("model", model_fun, requires="quality"),
], spec=spec)

run = ep.run_eye_pipeline(pipeline, context={"path": "study.csv"})
audit = ep.audit_eye_pipeline(run)
ax = ep.plot_eye_analysis_pipeline(pipeline)
```

`eye_targets_manifest()` and `write_eye_targets_template()` provide explicit interoperability scaffolding. They do not pretend arbitrary Python closures can be translated losslessly into R targets commands or another workflow engine.
