"""Smoke-check a manual eyeprocesspy installation."""
import eyeprocesspy as ep

print("eyeprocesspy version:", ep.__version__)
print("frozen R reference:", ep.__r_reference_version__)

study = ep.eyeprocess_benchmark_study()
audit = ep.validate_benchmark_study(study)
print("benchmark valid:", audit["valid"])
print("manual installation: OK")
