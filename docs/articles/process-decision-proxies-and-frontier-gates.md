# Process-decision proxies and research-frontier gates

The frozen package distinguishes useful process representations from research-frontier estimators that have not yet met the package's evidence threshold.

`fit_kde_latent_distribution_irt()`, `fit_persistence_gaze_diffusion_irt()`, `fit_nonignorable_missing_irt()`, and `fit_crossclassified_process_irt_mhrm()` therefore return `eye_gated_process_model` contracts when no validated external estimator is supplied. No simpler internal surrogate is fitted in their place.

```python
kde = ep.fit_kde_latent_distribution_irt(response_matrix)
assert kde.status == "gated"
ep.audit_frontier_model_contract(kde)
```

`prepare_structured_unstructured_process_features()` also records the frozen leakage rule: learned scaling, vocabulary, embeddings, feature selection, and other representations must be fitted **inside the training fold only**.
