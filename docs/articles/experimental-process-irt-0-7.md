# Experimental Process-IRT Methods and Evidence Gates

`eyeprocesspy` follows the frozen R `eyeprocess` 0.11.1 policy for advanced process-IRT: use a transparent reference procedure when the source defines one, use an explicit validated external engine where the method is engine-specific, and never silently substitute a convenient estimator while calling it parity.

## Why these functions are gated

Experimental models are process-measurement hypotheses, not automatic psychological inference. HMM states, sequence embeddings, gaze residuals, process classes and latent spaces must not be relabelled as attention, strategy, engagement, effort, guessing, misconduct, comprehension or cognitive load without independent evidence.

## Process-state HMM + IRT

`fit_process_hmm_irt()` ports the frozen two-stage reference architecture: process features are standardized, a diagonal-Gaussian HMM is fitted within ordered sequences by forward/backward EM, posterior state occupancy is retained, and occupancy can enter a response layer. `process_state_occupancy()` and `process_state_transition_summary()` expose the auditable state evidence. This is deliberately distinguished from a fully joint HMM-IRT likelihood.

## Cognitive diagnosis and latent process classes

`fit_cognitive_diagnosis_process()` preserves the GDINA/external-engine boundary. The exact GDINA path is not silently replaced in Python. `fit_latent_class_process_irt()` provides the frozen transparent two-stage process-class reference workflow. Process clusters are statistical groupings, not named cognitive strategies.

## Latent-space IRT

`fit_latent_space_irt()` remains an explicit LSMjml-equivalence gate because the frozen R implementation delegates to LSMjml. `process_residual_map()` and `validate_latent_space_process_similarity()` operate on validated latent-space result contracts; they do not manufacture latent coordinates when the required engine is absent.

## Process-adjusted DIF

`process_dif_nuisance_surrogate()` constructs a person-level process surrogate and `audit_process_adjusted_dif()` contrasts response-model coefficients before and after nuisance adjustment. The Python fixed-effect reference is marked as a backend-different diagnostic where the R design depends on optional mixed-model machinery. A reduction after adjustment is not a causal explanation of DIF.

## Sequence representations

`process_ngram_features()` constructs deterministic n-gram counts. `process_sequence_embedding()` applies TF-IDF weighting followed by truncated SVD. `fit_response_process_embedding_irt()` joins those embeddings to a response layer while retaining the experimental feature-integration status.

## Flexible item-response curves

`fit_gpirt(engine="spline_reference")` remains a **shape-criticism reference**, not a Gaussian-process IRT estimator. `compare_parametric_nonparametric_irf()` and `audit_irf_shape()` quantify departure from conventional logistic curves. Exact dynamic GPIRT and Flow-MIRT remain external-engine gates.

## Linking and person fit

`equate_irt_scales()` implements mean-sigma, mean-mean, Stocking-Lord and Haebara linking. `process_person_fit()` reports joint response/RT/process discrepancy evidence without assigning a psychological or behavioural label to a person.

## Process-aware adaptive testing

`process_item_information()`, `expected_process_information()`, `select_next_item_process()` and `simulate_process_cat()` combine response information with explicitly weighted process/RT information and optional burden penalties. The simulator is a design tool, not a production testing engine.

## Promotion rule

Advanced methods should remain experimental until recovery, calibration, misspecification, preprocessing sensitivity, transportability, negative controls and independent empirical validation support promotion.
