# Why participant-aggregated mediation can be misleading

Participant aggregation removes the trial index before mediation is estimated. That can be useful for a genuinely between-person hypothesis, but it does not estimate the same paths as a repeated-measures mediation model.

Consider a disclosure experiment where every participant sees both disclosed and undisclosed ads. If dwell changes across those trials, the scientifically relevant `a` path is primarily **within participant**. A participant-level average disclosure score can have little or no variance, while a participant-level average dwell score mainly captures stable reading differences. Regressing one participant mean on another therefore mixes levels and may erase the manipulated contrast.

The preparation contract separates:

```text
within exposure:   X_ij - Xbar_i
between exposure:  Xbar_i
within mediator:   M_ij - Mbar_i
between mediator:  Mbar_i
```

The downstream model can then estimate `a_W`, `a_B`, `b_W`, and `b_B` separately. The within indirect effect `a_W × b_W` answers a different question from `a_B × b_B`.

## Interpretation rule

Do not describe the between-participant path as evidence for the within-participant mechanism, or vice versa. When the manipulation is purely within participant, a missing or near-zero between-exposure variance is expected rather than a model defect.

## Practical diagnostic

Run `identify_mediation_levels()` before fitting. If a variable is classified as `within_only`, do not force a between path into the substantive interpretation. If it is `between_only`, a within-person mediation claim is unsupported by the observed design.
