# Scanpaths and process episodes

`representative_scanpath()` summarizes a collection of AOI sequences or coordinate paths using medoid, consensus, or barycentric representations under the declared distance rule. `scanpath_dispersion()`, permutation-based group comparison, and bootstrap representative stability retain uncertainty around the chosen representative rather than presenting one path as uniquely canonical.

For multichannel time-ordered data, `detect_process_changepoints()` compares local channel means on either side of candidate boundaries. `segment_process_episodes()` converts accepted boundaries into episodes, and `label_process_episodes()` applies transparent rules or an explicitly supplied classifier. Default labels are descriptive workflow labels, not inferred cognitive states.
