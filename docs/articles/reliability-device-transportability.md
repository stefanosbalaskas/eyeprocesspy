# Reliability and device transportability

Process metrics require reliability evidence before they are used as person-, item-, or condition-level measurements. The frozen 0.11.1 workflow combines Generalizability-Theory-style variance decomposition with prospective decision studies and metric-level reliability auditing.

`fit_process_gstudy()` decomposes observed variance across declared facets such as person, item, session, and device. `design_process_dstudy()` evaluates how changing the number of items, sessions, or devices changes relative and absolute dependability. `audit_process_reliability()` provides ICC, G-theory, split-half, or bootstrap reference summaries.

These coefficients describe measurement stability under the declared design. They do not establish construct validity and should not be used as automatic participant classifications.
