# Evidence and decision provenance

The evidence graph links raw measurements, transformations, metrics, models, diagnostics, and decisions into an explicit directed graph. `build_evidence_graph()` creates the graph, `trace_item_decision()` retrieves the evidence ancestors of a decision, and `audit_evidence_dependencies()` checks missing endpoints and cycles.

`compare_decision_provenance()` reports nodes and edges added or removed between two decision pipelines. This makes consequential measurement decisions auditable and helps distinguish a changed conclusion caused by changed data, changed preprocessing, changed metrics, or changed models.

The graph records declared dependencies; it does not itself establish causal or construct-validity evidence.
