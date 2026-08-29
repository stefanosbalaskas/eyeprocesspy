# Experimental Process-IRT Methods and Evidence Gates

Some process-IRT methods are too estimator-specific or too weakly validated to present as production estimators. `eyeprocesspy` follows the frozen R package's three-way policy: provide a transparent reference estimator, provide an adapter to an established engine, or fail explicitly rather than silently substituting another model.

## Why these functions are gated

Experimental and gated models require deliberate opt-in through `fit_irt_model(..., allow_experimental=True)`. The registry records model status and requirements.

## Process-state HMM + IRT

The registered `process_hmm` model remains experimental. Until its later-tranche Python engine is parity-validated, the registry callback raises a backend gate rather than inventing an HMM estimator.

## Cognitive diagnosis and latent process classes

The current 0.7 tranche provides the revisiting-aware CDM bridge. The broader latent-class/process-state families are implemented in later source families and remain separate parity work.

## Latent-space IRT and flexible IRFs

`latent_space_process`, `gpirt_shape_audit`, and `flow_mirt` are registered with their frozen status. Where a validated Python-equivalent engine has not yet been established, calls fail explicitly. A shape audit or embedding is not relabelled as an exact GPIRT/flow-MIRT estimator.

## Sequence representations

Sequence/HMM/embedding channels can be declared with `irt_sequence_channel()`. Declaration does not imply that a process-state model has passed recovery or external validation.

## Linking, person fit, and process-aware CAT

Those families are implemented in the frozen 0.9 IRT tranche and retain separate evidence contracts. Process inconsistency must not be relabelled as cheating, deception, disengagement, or pathology.

## Promotion rule

Novelty is not evidence. Experimental models should remain experimental until simulation, recovery, calibration, misspecification, preprocessing sensitivity, transportability, and external validation support promotion.
