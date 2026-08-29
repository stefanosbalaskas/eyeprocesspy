import eyeprocesspy as ep
g=ep.build_evidence_graph(raw_data=['gaze','pupil'],transformations=['clean'],metrics=['dwell','pupil_auc'],models=['process_model'],diagnostics=['calibration'],decisions=['item_I01_revise'])
print(ep.audit_evidence_dependencies(g).summary)
print(ep.trace_item_decision(g,'I01').summary[['label','stage']])
