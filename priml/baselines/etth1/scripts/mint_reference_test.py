"""Mint source goldens through pytest when both source paths are supplied."""

from pathlib import Path

import json

import pytest

from priml.baselines.etth1.scripts.reference import capture_reference
from priml.baselines.etth1.testing import golden_record
from priml.testing.golden import write_tensors


def test_mint_source_goldens(pytestconfig: pytest.Config) -> None:
    reference = pytestconfig.getoption("--etth1-reference")
    directory = pytestconfig.getoption("--etth1-directory")
    if reference is None and directory is None:
        pytest.skip("Source minting requires explicit reference and dataset paths.")
    assert isinstance(reference, str)
    assert isinstance(directory, str)
    report, model, training = capture_reference(
        Path(reference),
        directory=Path(directory),
    )
    destination = Path(__file__).resolve().parents[1] / "testdata"
    for name, record in (
        ("dlinear_model.pt", golden_record(model, training=False)),
        ("dlinear_training.pt", golden_record(training, training=True)),
    ):
        write_tensors(destination / name, record=record)
        assert (destination / name).stat().st_size <= 32768
    report["mint_environment"] = "pytest with priml/conftest.py"
    (destination / "source.json").write_text(json.dumps(report, indent=2) + "\n")
