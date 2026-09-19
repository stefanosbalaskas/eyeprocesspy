# Why Event Detection Is an Analytical Choice

A fixation detector makes inferentially consequential decisions about **where one event begins and ends, whether a brief interruption fragments an event, and whether a sample sequence is labelled fixation, saccade, pursuit, PSO, or no event**. Those decisions can alter fixation count, dwell, first-fixation timing, scanpath length, transition structure, and any model using those quantities.

## The wrong comparison

A detector comparison should not stop at statements such as “algorithm A produced 110 fixations and algorithm B produced 125.” Event counts describe structural differences, but they do not reveal whether a scientific effect changes.

The multiverse therefore tracks three levels:

1. **event structure** — counts, durations, onset/offset differences, temporal overlap, matched-event precision/recall/F1 when meaningful;
2. **measurement consequences** — AOI dwell, counts, mean durations, TTFF, revisits, transitions, scanpath inputs, and fixation-locked pupil summaries;
3. **inference consequences** — coefficient distributions, confidence intervals, sign stability, estimate range, convergence, and an explicitly declared substantive threshold when one exists.

## Detector parameters are data- and task-dependent

Velocity, dispersion, minimum duration, gap-merging, smoothing, and filtering settings are not universal constants. Recording frequency and precision, data loss, smooth pursuit, head motion, and stimulus dynamics can all affect appropriate detector behavior. A parameter grid should therefore encode **defensible alternatives for the study**, not a generic “recommended range” presented without context.

## Event agreement is not validity

Agreement between two algorithms does not prove that either algorithm correctly represents ocular events. When manually labelled ground truth or an independently validated benchmark is available, compare against it. When it is not available, detector agreement is a robustness diagnostic rather than a validity criterion.

## Practical decision rule

Use a primary detector justified by the study design, then construct a sensitivity set around genuine analytical uncertainty. If the substantive result changes, report the dependency rather than selecting the detector that produces the preferred conclusion.

Next: [Building a Detector Multiverse](building-detector-multiverse.md).
