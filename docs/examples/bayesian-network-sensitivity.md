# Bayesian-network sensitivity example

This workflow asks whether graph conclusions survive alternative defensible upstream feature definitions.

## Detector branches

Each detector is run **before** the BN layer:

~~~text
I-VT ----------> events -> features -> BN input
I-DT ----------> events -> features -> BN input
Adaptive ------> events -> features -> BN input
REMoDNaV ------> events -> features -> BN input
~~~

Then compare the networks under an identical node/constraint specification:

~~~python
from eyeprocesspy.bayesian_networks import compare_bn_across_detectors

comparison = compare_bn_across_detectors(
    {
        "ivt": ivt_data,
        "idt": idt_data,
        "adaptive": adaptive_data,
        "remodnav": remodnav_data,
    },
    constraints=constraints,
    algorithm="hill_climb",
    score="bic",
)

print(comparison.edge_table)
~~~

A relation with `edge_robustness = 1.0` appeared under every planned specification; a relation with `0.25` appeared under one of four. Direction should still be inspected separately and should not be converted into a causal claim.

![Illustrative robustness](../assets/bayesian-networks/bn-detector-robustness.svg)

The same pattern applies to AOI-geometry sensitivity through `compare_bn_across_aoi_specs()`.
