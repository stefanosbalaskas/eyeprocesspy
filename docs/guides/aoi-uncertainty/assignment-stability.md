# Interpreting AOI Assignment Stability

Assignment stability compares each perturbed branch with the nominal AOI assignment. It is a measurement-robustness diagnostic, not an estimate of whether one geometry is scientifically “true.”

## Core quantities

The framework reports proportions unchanged, newly assigned, lost, and reassigned; AOI-to-AOI reassignment matrices; AOI-level stability; and optional participant- and trial-level stability.

## Fixation versus sample level

Declare the row type explicitly with `observation_level="fixation"` or `"sample"`.

Both levels use the same geometry and assignment engine. Recomputed features always include universal `observation_count` and `first_observation` fields. Fixation input additionally populates `fixation_count` and `first_fixation`; sample input populates `sample_count` and leaves fixation-specific fields missing. Sample rows are therefore never relabeled as fixations.

## Zero is not missing

When a complete AOI universe is supplied by the sensitivity runner, participant/trial × AOI cells are retained even when nothing lands in an AOI.

- `0` means the group had valid assignment opportunities but none were assigned to that AOI.
- Missing count/inspection means the group had **no valid assignment opportunity**, for example because all coordinates were missing.
- If assigned observations have missing duration values, dwell remains missing rather than being partially summed.
- First-observation timing is never inferred from row order.

These rules keep denominators auditable across perturbation branches.

## Missing coordinates and overlap

Missing x/y values remain missing and are excluded from comparable-assignment denominators. They are never converted to zero or outside-AOI observations.

Multiple simultaneous memberships default to `__ambiguous__`. This prevents boundary overlap from being silently converted into a unique AOI.

## Assignment frequency

`estimate_fixation_assignment_probability()` reports the empirical frequency with which an observation maps to each AOI across the **declared perturbation set**. Despite the historical function name, this is not a posterior probability of true membership.

Low stability signals sensitivity to geometry; it does not identify which boundary is substantively correct.

Continue to [Propagating AOI Uncertainty Into Statistical Models](model-propagation.md) or use the [decision clinic](decision-clinic.md) to diagnose a stability pattern.
