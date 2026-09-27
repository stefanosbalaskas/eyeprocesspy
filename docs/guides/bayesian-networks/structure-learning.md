# Theory-constrained structure learning

Structure learning is a search over dependency structures, not permission to ignore experimental design.

## Encode temporal knowledge

~~~python
from eyeprocesspy.bayesian_networks import define_bn_constraints

constraints = define_bn_constraints(
    temporal_tiers=[
        ["condition"],
        ["target_dwell_fraction", "fixation_count", "transition_entropy"],
        ["pupil_auc", "scr_amplitude"],
        ["trust", "confidence"],
        ["choice"],
    ],
    required_edges=[],
    forbidden_edges=[],
    max_parents=4,
    nodes=bn_data.structure_nodes,
)
~~~

All edges from later tiers back into earlier tiers are forbidden. Directions within a tier remain available unless separately constrained.

## Algorithms

The first pgmpy backend exposes:

- `hill_climb`: score-based DAG search;
- `pc`: stable constraint-based discovery for purely discrete or Gaussian data;
- `ges`: score-based equivalence search returned as a DAG by the adapter.

Use a score appropriate to the node family. Passing `score="bic"` is translated to `bic-d`, `bic-g`, or `bic-cg` for discrete, Gaussian, or mixed structure search.

## Recommended analysis logic

1. Predeclare scientifically impossible directions.
2. Keep required edges sparse; otherwise discovery becomes confirmation by construction.
3. Restrict parent count when the sample size cannot support large local models.
4. Compare more than one defensible learner/score when structure is exploratory.
5. Bootstrap the graph before interpreting individual edges.
6. Discuss Markov-equivalent directions explicitly.

An unconstrained graph containing an experimental node produces an `eyeprocesspy` warning unless `allow_unconstrained=True` is explicitly supplied.
