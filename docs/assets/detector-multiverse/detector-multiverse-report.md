# Event-detector multiverse report

## Scope

This report evaluates whether events, AOI features, and statistical conclusions change across the supplied defensible detector specifications. The specification set is not evidence that omitted detector choices are valid or irrelevant.

## Detector specifications

| detector_id | algorithm | velocity_threshold | dispersion_threshold | minimum_duration_ms | maximum_gap_ms | merge_rule | sampling_rate | coordinate_unit | implementation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive | adaptive_velocity |  |  | 60 | 75 | none | 60 | degrees | eyeprocesspy |
| idt_A | idt |  | 1.2 | 80 |  | none | 60 | degrees | eyeprocesspy |
| ivt25 | ivt | 25 |  | 60 | 75 | none | 60 | degrees | eyeprocesspy |
| ivt30 | ivt | 30 |  | 60 | 75 | none | 60 | degrees | eyeprocesspy |
| ivt35 | ivt | 35 |  | 60 | 75 | none | 60 | degrees | eyeprocesspy |
| remodnav | remodnav |  |  | 60 |  | none | 60 | degrees | REMoDNaV |

## Event-level sensitivity

| detector_id | number_of_fixations | mean_fixation_duration_ms | total_fixation_duration_ms |
| --- | ---: | ---: | ---: |
| adaptive | 77 | 377.06 | 29033.33 |
| idt_A | 52 | 598.72 | 31133.33 |
| ivt25 | 87 | 325.86 | 28350.00 |
| ivt30 | 80 | 361.67 | 28933.33 |
| ivt35 | 76 | 382.46 | 29066.67 |

## AOI-feature sensitivity

Across trial × AOI units, detector choice changed dwell, fixation counts, mean fixation duration, and TTFF. The synthetic example retained zero-event trials and kept all-invalid trials missing rather than recoding them as zero.

## Inference stability

For the simulated disclosure-condition term, six detector specifications were planned, five returned the requested coefficient, and one REMoDNaV branch failed explicitly because the external backend was unavailable. All five modelled internal branches converged and retained the same effect direction, so the planned-specification convergence rate is 5/6 (0.833). The estimate range was approximately 20.8 ms across converged branches in this seeded example. These values are demonstration output, not empirical evidence.

## Branch failures

The REMoDNaV branch was explicitly recorded as unavailable in the environment used to generate this report. No internal detector was silently substituted.

## Reporting guidance

Report the detector family and parameters, sampling rate and coordinate units, AOI assignment rule, number of successful/failed specifications, event-level agreement, the range of key AOI features, coefficient distributions with uncertainty, convergence failures, and any substantive threshold used. Do not summarize robustness by counting p-values alone.

## Limitations

Detector sensitivity is conditional on the supplied preprocessing, AOIs, quality rules, model specification, and detector set. Agreement between detectors does not establish event validity, and disagreement does not identify which detector is correct without external evidence.
