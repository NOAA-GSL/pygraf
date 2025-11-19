"""
Add command line options to the pytest suite.

Each CLA needs to be defined in pytest_addoption and to have a pytest.fixture
function defined.
"""

import glob
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def cleanup_data_idx():
    yield  # Nothing to be done before tests
    print("Removing idx files from test data")
    for path in glob.glob("tests/data/*.idx"):
        Path(path).unlink()


def pytest_addoption(parser):
    """Define command line arguments to be parsed."""

    parser.addoption(
        "--nat-file",
        action="store",
        help="Path to nat-file.",
    )

    parser.addoption(
        "--prs-file",
        action="store",
        help="Path to prs-file.",
    )


@pytest.fixture(scope="session")
def natfile():
    """Interface to  pass a grib file to pytest."""

    return Path("tests", "data", "wrfnat_hrconus_16.grib2")


@pytest.fixture(scope="session")
def prsfile():
    """Interface to  pass a grib file to pytest."""
    return Path("tests", "data", "wrfprs_hrconus_16.grib2")


@pytest.fixture(scope="session")
def spec_file():
    """Interface to  pass a grib file to pytest."""
    return Path("adb_graphics", "default_specs.yml")
