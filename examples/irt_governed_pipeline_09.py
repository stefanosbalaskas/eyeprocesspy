"""Governed pipeline dependencies and analysis-decision provenance."""
import eyeprocesspy as ep

spec = ep.eye_analysis_spec(method="declared")
pipeline = ep.eye_analysis_pipeline([
    ep.eye_pipeline_step("source", lambda context, spec: context["value"]),
    ep.eye_pipeline_step("transform", lambda source, context, spec: source + 3, requires="source", decision="method"),
], spec=spec)
run = ep.run_eye_pipeline(pipeline, context={"value": 2})
audit = ep.audit_eye_pipeline(run)
assert ep.pipeline_result(run, "transform") == 5
assert audit.valid is True
assert "digraph" in ep.eye_pipeline_dot(pipeline)
