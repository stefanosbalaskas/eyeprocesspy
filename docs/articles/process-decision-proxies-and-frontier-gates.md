# Process-decision proxies and research-frontier gates

The frozen package distinguishes useful process representations from research-frontier estimators that have not yet met the package's evidence threshold.

`fit_kde_latent_distribution_irt()`, `fit_persistence_gaze_diffusion_irt()`, `fit_nonignorable_missing_irt()`, and `fit_crossclassified_process_irt_mhrm()` therefore return `eye_gated_process_model` contracts when no validated external estimator is supplied. No simpler internal surrogate is fitted in their place.

```python
kde = ep.fit_kde_latent_distribution_irt(response_matrix)
assert kde.status == "gated"
ep.audit_frontier_model_contract(kde)
```

`prepare_structured_unstructured_process_features()` also records the frozen leakage rule: learned scaling, vocabulary, embeddings, feature selection, and other representations must be fitted **inside the training fold only**.

## Pre-action representations

`preaction_process_features()` derives leakage-aware observed features from configurable windows before the response event: AOI entropy/switches, AOI proportions, pupil mean/slope, and blink proportion. These are representations of observed process dynamics, not evidence of latent intention.

```python
pre = ep.preaction_process_features(samples, windows_ms=[500, 1000, 2000])
ax = ep.plot_eye_preaction_process_features(pre, feature="pupil_mean")
```

## aDDM/GLAM-inspired descriptive proxies

`addm_glam_proxy_features()` computes target/distractor/action proportions, evidence slopes, late-minus-early evidence and conservative relative-attention/caution proxies. They are **not** fitted drift rate, gaze-discount, decision-threshold, or causal-attention parameters.

```python
proxy = ep.addm_glam_proxy_features(samples)
ax = ep.plot_eye_decision_process_proxy(proxy)
```

`process_feature_family_registry()` and `assign_process_feature_family()` apply the frozen first-match interpretation registry. `process_feature_stability()` then summarizes top-N selection rates and mean importance across repeated splits. Learned representations and feature selection remain training-fold-local when used predictively.
