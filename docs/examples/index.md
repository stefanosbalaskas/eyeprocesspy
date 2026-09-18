# Runnable examples

`eyeprocesspy` ships deterministic examples that require no private participant data. The focused workflows below have been executed against the CI-built `0.1.0` wheel; the gallery generators produce the figures shown throughout this site.

<div class="grid cards" markdown>

-   :material-vector-link: **Trial-level mediation preparation**

    Preserve every repeated-measures trial while separating within- and between-participant exposure/gaze components and auditing zero, missingness, and quality states.

    [Open mediation guide](../guides/multilevel-mediation/index.md)

-   :material-eye: **Core gaze, AOI and provenance**

    Validate a canonical `EyeDataset`, recover scanpaths and transitions, compute gaze entropy, render auditable plots and inspect provenance.

    [Open worked workflow](core-workflow.md)

-   :material-target: **Calibration uncertainty → probabilistic AOIs**

    Fit an empirical calibration-error model, propagate coordinate uncertainty and diagnose boundary-sensitive AOI assignments.

    [Open worked workflow](calibration-probabilistic-aoi.md)

-   :material-chart-timeline-variant: **Process-measure reliability**

    Estimate repeated-measure ICC, Bland–Altman agreement and temporal stability without confusing reliability with construct validity.

    [Open worked workflow](process-reliability.md)

-   :material-timer-sand: **Censored gaze latency**

    Build event/censor rows from trial windows and fixation/AOI events, retain never-inspected valid trials, fit clustered Cox/AFT models, inspect diagnostics, and produce manuscript-ready reporting.

    [Open worked workflow](gaze-survival-analysis.md)

-   :material-source-branch: **Evidence-verification survival**

    Model time to first source/evidence AOI entry under repeated conditions, retain valid never-inspected trials as censored observations, and compare clustered Cox with explicit AFT sensitivity models.

    [Open verification workflow](gaze-verification-survival.md)

-   :material-vector-polygon: **AOI perturbation sensitivity**

    Perturb AOI geometry in pixels or degrees, remap fixations, recompute features, propagate the same model, and inspect robustness visually.

    [Open worked workflow](aoi-perturbation-sensitivity.md)

-   :material-alert-circle-check: **AOI failure clinic**

    Trigger geometry failure, overlap ambiguity, non-convergence, callback failure, and missing coordinates deliberately, then inspect the retained audit trail.

    [Open failure clinic](aoi-perturbation-failure-clinic.md)

-   :material-chart-bell-curve-cumulative: **IRT diagnostics**

    Inspect conditional information, item fit and DIF with publication-ready Matplotlib diagnostics.

    [Open worked workflow](irt-diagnostics.md)

</div>

For short task-oriented snippets, use the [Cookbook](../cookbook.md). For visual output, browse the [15-figure gallery](../gallery.md).

## Verify the installation

```python
import eyeprocesspy as ep

print(ep.__version__)
print(ep.__r_reference_version__)

study = ep.eyeprocess_benchmark_study()
audit = ep.validate_benchmark_study(study)
print(audit["valid"])
```

## Import and validate an export

```python
import eyeprocesspy as ep

eye = ep.read_eye_export("participant_001.csv", vendor="auto")
issues = ep.validate_eye_dataset(eye)

if not issues.empty:
    print(issues)
```

The canonical `EyeDataset` keeps recordings, streams, gaze, eye samples, episodes, events, intervals, responses, coordinate spaces, AOIs, features, quality and provenance in explicit tables.

## Scanpath and transition analysis

```python
sequence = ep.scanpath_sequence(
    eye,
    trial_id="trial-01",
    source="visits",
    collapse_consecutive=True,
)

matrix = ep.transition_matrix(
    eye,
    source="visits",
    normalize="row",
)

entropy = ep.gaze_entropy(
    eye,
    level="trial",
    source="samples",
)
```

## Plot gaze, fixations and pupil data

```python
import matplotlib.pyplot as plt
import eyeprocesspy as ep

ax = ep.plot_eye_trace(eye, trial_id="trial-01")
plt.show()

ax = ep.plot_scanpath(eye, trial_id="trial-01")
plt.show()

ax = ep.plot_pupil_timeseries(eye, trial_id="trial-01")
plt.show()
```

The plotting surface preserves its numerical payload on the returned axes where relevant:

```python
plot_data = ax.eyeprocess_plot_data
```

Matrix plots additionally expose `ax.eyeprocess_plot_matrix`.

## Process-measure reliability

```python
profile = ep.process_reliability_profile(
    repeated,
    person="person",
    session="session",
    measure="dwell_score",
)

print(profile["icc"])
print(profile["bland_altman"]["summary"])
```

Reliability is population- and design-dependent; it does not establish construct validity.

## Calibration uncertainty and probabilistic AOIs

```python
model = ep.calibration_error_model(calibration_validation_data)
uncertainty = ep.gaze_uncertainty_ellipse(model, level=0.95)

assignment = ep.probabilistic_aoi_assignment(
    gaze_points,
    aois,
    model,
    draws=500,
    seed=1,
    min_probability=0.50,
)
```

This workflow quantifies coordinate uncertainty under the fitted calibration-error model; it does **not** estimate a posterior probability of psychological attention.

## IRT diagnostic plotting

```python
ax = ep.plot_eye_irt_information_profile(information_profile)
ax = ep.plot_eye_irt_item_fit(item_fit, statistic="infit")
ax = ep.plot_eye_irt_dif_curve(dif_curve)
```

The wider IRT surface also includes person fit, Q3/local dependence, score uncertainty, adaptive traces, link stability, DTF, recovery/SBC evidence, bank coverage, prior sensitivity and process-alignment diagnostics.

## Provenance and reproducibility

```python
manifest = ep.provenance_manifest(eye)
print(manifest["schema_version"])
print(manifest["sources"])
print(manifest["validation"])
```

For release-level verification, use the deterministic benchmark, validation evidence, reproducibility manifest and software-paper evidence workflows documented in the article library.

## Executable programs

| Script | Purpose | Output |
| --- | --- | --- |
| `examples/complete_workflow.py` | Canonical dataset → validation → scanpath/transitions/entropy → plots → provenance | `workflow-output/*.svg` |
| `examples/calibration_probabilistic_aoi.py` | Calibration error → uncertainty ellipse → probabilistic AOI | `workflow-output/*.svg` |
| `examples/process_reliability.py` | ICC, Bland–Altman and temporal stability | `workflow-output/process-reliability.svg` |
| `examples/irt_diagnostics.py` | Information, item fit and DIF diagnostics | `workflow-output/*.svg` |
| `examples/core_gallery.py` | Eight core gaze/AOI/pupil plots | `gallery-output/*.svg` |
| `examples/advanced_gallery.py` | Reliability, uncertainty, quality and IRT plot families | `gallery-output/*.svg` |
| `examples/worked_gaze_survival_analysis.py` | Raw trial/event inputs → censoring → Kaplan–Meier → clustered Cox/AFT → diagnostics/reporting | console summaries + survival plots |
| `examples/worked_gaze_verification_survival.py` | Source/evidence AOI entry → censoring → clustered Cox/AFT → diagnostics/reporting | CSV/JSON summaries + Kaplan–Meier SVG |
| `examples/aoi_perturbation_sensitivity.py` | AOI geometry → reassignment → feature/model sensitivity → robustness plots | `workflow-output/aoi-*.svg` |
| `examples/aoi_perturbation_failure_clinic.py` | Deliberate AOI geometry/model failures → retained audits → troubleshooting report | `workflow-output/aoi-failure-clinic/*` |

| `examples/multilevel_mediation_preparation.py` | Trial-level gaze-mediator preparation, within/between decomposition, zero/missing/quality audit | console audit tables |

All listed examples are deterministic and use no private participant data.

## AOI uncertainty

- [AOI perturbation sensitivity](aoi-perturbation-sensitivity.md): fixation-level end-to-end geometry, assignment, feature, model, and plotting workflow.
- [AOI perturbation failure clinic](aoi-perturbation-failure-clinic.md): deliberate geometry/model failure, ambiguity, non-convergence, and retained audit trails.
- [Sample-level AOI sensitivity](sample-level-aoi-sensitivity.md): explicit sample semantics, zero-versus-missing cells, and when to separate geometry from detector uncertainty.
