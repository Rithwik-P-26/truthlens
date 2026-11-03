"""Shared fixtures for the TruthLens test suite (run: .venv/bin/python -m pytest -q)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MODELS = ROOT / "models"
EXAMPLES = ROOT / "examples"


@pytest.fixture(scope="session")
def detector():
    if not (MODELS / "config.json").is_file():
        pytest.skip("models/config.json not found (no trained weights)")
    import torch

    from truthlens.inference import Detector

    torch.set_num_threads(2)
    return Detector(str(MODELS), device="cpu")


@pytest.fixture(scope="session")
def example_rows() -> list[dict]:
    with open(EXAMPLES / "labels.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
