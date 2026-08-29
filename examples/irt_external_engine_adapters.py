"""External-engine adapter and equivalence example."""
import numpy as np
import pandas as pd
import eyeprocesspy as ep

registry = ep.external_model_engines()
print(registry[["engine", "domain", "available"]])

responses = pd.DataFrame({"I1": [1, 0, 1, 1], "I2": [0, 1, 1, 0]})
result = ep.fit_mirt_adapter(responses, purpose="demonstrate exact-engine gating")
assert result.status == "not_available"
assert ep.validate_engine_adapter(result).valid

# Engine-comparison infrastructure is backend-neutral and executable in Python.
data = pd.DataFrame({"x": np.arange(10, dtype=float)})
data["y"] = 0.5 + 1.25 * data["x"]
engines = {
    "reference": lambda d: np.polyfit(d.x, d.y, 1),
    "replicate": lambda d: np.polyfit(d.x, d.y + 1e-12, 1),
}
comparison = ep.compare_model_engines(
    data,
    engines=engines,
    extractors=lambda fit: {"slope": float(fit[0])},
    reference="reference",
    tolerance=1e-8,
)
assert comparison.estimates["equivalent"].dropna().all()
ax = ep.plot_eye_engine_comparison(comparison, parameter="slope")
assert len(ax.eyeprocess_plot_data) == 2
