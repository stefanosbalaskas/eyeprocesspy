# EMIP Route B — post-result independent-audit implementation clarification

**Audit date:** 13 November 2026. This addendum **does not modify** the pre-result
31-participant EMIP protocol, its 24-branch planned denominator, the original
derived data, the source inclusion rule, or any fitted effect estimate.

The original declared EMIP I-VT comparison specified an instantaneous
two-sample pixel-displacement velocity under the source-reported 344 × 194-mm,
1920 × 1080-pixel display and nominal 700-mm viewing distance. It did **not**
separately publish a computational formula distinguishing two possible angular
mappings.

The independent audit therefore checked both:
1. **Small-angle pairwise-displacement implementation:** convert physical
   displacement between consecutive valid gaze samples to degrees using
   `(delta_x_mm / 700) × (180 / pi)`, analogously for y, then divide
   Euclidean angular displacement by the intersample elapsed seconds.
2. **Eccentricity-dependent arctangent mapping:** convert each absolute
   screen-centered position to `atan(offset_mm / 700)` in degrees and
   difference successive angles.

In an initial audit of participant ID 1, Vehicle, left-mask, I-VT 40°/s,
the second representation yielded one additional fixation (169 versus 168)
without changing AOI assignments (47) or transition counts (0). Other
detector variants on that test record matched. The small-angle instantaneous
displacement convention matched the frozen I-VT implementation; its later
application to **all 31 public original gaze streams** reproduced all 744
frozen derived records with zero differences in valid support, fixation counts,
AOI assignment and transition counts. An independent re-fit of the 24
Poisson working models then matched all coefficients, clustered SEs,
and Wald confidence intervals to numerical precision.

**Interpretation:** this is an exact independent computational replication
of one now-explicit, source-grounded event-detection implementation.
It does **not** establish that all plausible visual-angle mappings are
numerically equivalent, that one detector is more accurate, or that the
nonrandom 31-person mirror generalizes to the original 216-person EMIP
release. The original preregistered/frozen protocol is not backdated to claim
it explicitly specified the small-angle formula. Future uses should make
the exact angular mapping and rounding conventions part of the frozen
measurement-decision contract.

This addendum is preserved separately to make the validation's dependence
on implementation choice visible to reviewers.
