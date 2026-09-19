# Detector multiverse decision guide

Use this guide before looking at detector-specific inferential results. The aim is to define a defensible measurement-sensitivity question, not to enumerate every algorithm or threshold that can be made to run.

## First question: does detector choice reach the estimand?

A detector multiverse is most useful when fixations, saccades, pursuits, or their boundaries feed the scientific outcome: AOI dwell, fixation count, TTFF, revisits, transitions, scanpaths, fixation-locked pupil summaries, or models built from those features.

If the analysis uses only raw sample-level gaze positions and never classifies events, detector sensitivity is not the relevant uncertainty analysis. Consider coordinate quality, AOI geometry, preprocessing, or sampling sensitivity instead.

## Decision table

| Situation | Recommended analysis | Avoid |
| --- | --- | --- |
| One detector is required by a preregistered protocol and alternatives are not scientifically defensible | Run the declared detector; document its parameters and limitations | Inventing a multiverse after seeing the result |
| Several thresholds within one detector family are defensible | Use an explicit parameter grid with a predeclared range | Treating the grid as a universal recommendation |
| More than one detector family is defensible | Include family-level branches and compare events before inference | Comparing coefficients without checking whether event catalogues materially differ |
| Natural viewing/pursuit content motivates REMoDNaV | Use the external REMoDNaV bridge and record its version/status | Relabelling the built-in adaptive reference detector as REMoDNaV |
| Vendor events are scientifically relevant | Include the vendor catalogue as a labelled branch | Treating vendor output as ground truth without external validation |
| A specialist estimator is required downstream | Use the `callback` model engine with the documented tidy contract | Silently substituting OLS or another simpler estimator |
| A detector or model backend is unavailable | Retain the failed branch in the audit and convergence denominator | Replacing it with a different algorithm without disclosure |

## Before you define branches

Write down:

1. the event types that matter to the scientific claim;
2. sampling rate and coordinate units;
3. preprocessing shared by all branches;
4. detector families that are defensible for the task and signal;
5. parameter ranges and the scientific reason for each bound;
6. AOI definitions and overlap rule;
7. quality rules used before modelling;
8. the identical downstream model specification;
9. the coefficient term and any substantive threshold used for interpretation;
10. how failed, missing-term, and non-converged branches will be reported.

## How many specifications?

There is no universal target number. The set should be large enough to cover the **scientifically plausible decision space** and small enough that every branch can be explained. Five carefully justified specifications are more informative than hundreds of arbitrary combinations.

If a parameter grid becomes large, separate questions: first evaluate within-family parameter sensitivity, then compare a smaller number of family-level reference specifications. This keeps the inferential record interpretable.

## What counts as a concerning result?

Detector sensitivity deserves discussion when one or more of the following occur:

- event counts or durations differ enough to alter the process description;
- temporal overlap is low for events central to the claim;
- AOI features change enough to alter substantive interpretation;
- the coefficient changes sign or crosses a predeclared substantive threshold;
- confidence intervals or effect magnitude vary materially;
- convergence or term availability differs across branches;
- row attrition differs across branches in `input_audit`.

None of these patterns identifies which detector is correct by itself. They show that the result is conditional on measurement choices and may require external validation or a narrower claim.

## API path

- Define branches: [`define_event_detector_spec()` and `create_detector_multiverse()`](building-detector-multiverse.md)
- Compare events: [Comparing event catalogues](comparing-event-catalogues.md)
- Propagate to AOIs/features: [Detector choice → AOI features](detector-aoi-features.md)
- Propagate to models: [Detector choice → statistical inference](detector-statistical-inference.md)
- Report: [Reporting detector sensitivity](reporting-detector-sensitivity.md)
- Diagnose failures: [Detector multiverse troubleshooting](detector-multiverse-troubleshooting.md)
- Exact signatures: [Detector multiverse API](../reference/detector-multiverse.md)
