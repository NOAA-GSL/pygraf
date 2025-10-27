"""
Add command line options to the pytest suite.

Each CLA needs to be defined in pytest_addoption and to have a pytest.fixture
function defined.
"""

from pathlib import Path

import pytest


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


@pytest.fixture
def natfile():
    """Interface to  pass a grib file to pytest."""

    return Path("tests", "data", "wrfnat_hrconus_16.grib2")


@pytest.fixture
def prsfile():
    """Interface to  pass a grib file to pytest."""
    return Path("tests", "data", "wrfprs_hrconus_16.grib2")


@pytest.fixture
def spec_file():
    """Interface to  pass a grib file to pytest."""
    return Path("adb_graphics", "default_specs.yml")
