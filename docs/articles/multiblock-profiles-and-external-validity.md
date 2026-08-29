# Multiblock process structure, profiles, and external validity

## Conceptual feature blocks

`process_feature_blocks()` defines non-overlapping psychometric, gaze, pupil, quality, or other conceptual blocks. `fit_multiblock_process_map(..., engine="pca_block_scaled")` supplies the frozen package's transparent block-standardized PCA reference: each block is standardized and scaled by the square root of its number of variables before the joint decomposition.

```python
blocks = ep.process_feature_blocks(person_data, {
    "Psychometric": ["theta", "accuracy"],
    "Gaze": ["dwell", "entropy"],
    "Pupil": ["pupil"],
    "Quality": ["validity"],
}, id="person_id")
fit = ep.fit_multiblock_process_map(blocks, engine="pca_block_scaled")
```

This is exploratory block-aware mapping, not a replacement for multimodal IRT.

## Process-profile discovery

`fit_process_profile_mixture(..., engine="kmeans_reference")` is deliberately labelled `descriptive_kmeans_reference_not_finite_mixture`. Membership probabilities are distance-based descriptive proximity values. Profile labels remain neutral until externally validated.

## External validity

`audit_process_external_validity()` separates baseline predictors from added process predictors and reports process-criterion associations and incremental R². Association with an external criterion contributes validity evidence but does not establish a causal mechanism.
