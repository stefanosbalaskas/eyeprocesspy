# Interoperability with targets-style workflows

`eyeprocesspy` governed pipelines and R `targets`-style workflows solve different but complementary problems. The eyeprocess pipeline object records scientific decisions and process lineage; a targets workflow can orchestrate file/function dependencies and incremental execution.

```python
manifest = ep.eye_targets_manifest(pipeline)
ep.write_eye_targets_template(pipeline, "_targets.R")
```

The generated file is intentionally a template with explicit placeholders. `eyeprocesspy` does not silently rewrite arbitrary Python step closures into executable R targets expressions because such translation can change semantics or hide undeclared dependencies.
