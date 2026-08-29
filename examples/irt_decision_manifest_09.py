"""Research-decision manifest, lock, blinding snapshot, and decision-space audit."""
import pandas as pd
import eyeprocesspy as ep

manifest = ep.eye_decision_manifest(
    model={"family": "gaussian"},
    provenance={"data_source": "demo", "software_version": ep.__version__, "analysis_commit": "example"},
)
lock = ep.lock_decision_manifest(manifest)
snapshot = ep.outcome_blind_snapshot(pd.DataFrame({"x": [1,2,3], "outcome": [0,1,0]}), outcome="outcome")
entropy = ep.analysis_decision_entropy(fixation=(60,80), pupil=("none","linear"), base=2)
assert ep.verify_decision_manifest_lock(lock)
assert ep.verify_outcome_blind_snapshot(snapshot)
assert entropy.attrs["joint_specifications"] == 4
