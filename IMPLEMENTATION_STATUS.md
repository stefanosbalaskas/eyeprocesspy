# Implementation status

## Frozen scope

| Metric | Count |
|---|---:|
| R exports | 1182 |
| S3 registrations | 435 |
| R source files | 114 |
| Rd files | 969 |
| R testthat files | 113 |
| Articles/vignettes | 88 |
| Stan programs | 13 |
| External/data resources | 66 |
| Plot candidates (initial heuristic) | 341 |

## Python parity

| Stage | Count |
|---|---:|
| P0 discovered | 1182 |
| P1 API | 56 |
| P2 structural | 56 |
| P3 semantic | 0 |
| P4 numerical | 0 |
| P5 algorithmic | 0 |
| P6 plot | 0 |
| P7 docs/examples | 0 |

Phase 0 manifests are generated. No generated placeholders are counted as implementations.

Direct R definition lookup resolved all 1,182 exported names to source definitions or registered generics/methods in the frozen source inventory.

## Milestone 1 initial foundational tranche

Implemented without placeholders: **56** frozen exports.

Families: canonical schema, generic mapping/inference/validation, canonical dataset construction/validation, table mutation/provenance, timebase primitives, clock transforms, coordinate-space registration/conversion/audit, generic import, adapter registry/detection, folder import, dataset combining and recording-ID remapping, and first-class Gazepoint 7.x gaze/fixation/biometric ingestion, pairing, profiling and event parsing.

These are **initial source-level ports**; cross-language R-oracle verification remains required before promoting them to final P3/P4/P5 status.
