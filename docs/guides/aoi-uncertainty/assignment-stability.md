# Interpreting AOI Assignment Stability

Assignment stability compares each perturbed branch with the nominal AOI assignment.

## Core quantities

The framework reports proportions unchanged, newly assigned, lost, and reassigned; AOI-to-AOI reassignment matrices; AOI-level stability; and optional participant- and trial-level stability.

## Missing coordinates

Missing x/y values remain missing. They are excluded from the comparable-assignment denominator and are never converted to zero or outside-AOI observations.

## Overlap

Multiple simultaneous memberships default to `__ambiguous__`. This prevents boundary overlap from being silently converted into a unique AOI.

## Assignment frequency

`estimate_fixation_assignment_probability()` reports the empirical frequency with which an observation maps to each AOI across the declared perturbation set. It is **not** a posterior probability of true membership.

Low stability signals sensitivity to geometry; it does not tell you which boundary is substantively correct.

Continue to [Propagating AOI Uncertainty Into Statistical Models](model-propagation.md).