"""Pipeline harness: registry, benchmarks, run store, viewer, and live replay.

Run from the repository root with the acoustic environment: `.venv/bin/python -m harness`.
See harness/README.md. Outputs live under ignored results/harness/.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
