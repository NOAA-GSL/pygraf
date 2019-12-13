'''
Add command line options to the pytest suite.
'''

import pytest


def pytest_addoption(parser):
    parser.addoption('--grib-file',
            action='store',
            help='Path to grib-file.',
            )

@pytest.fixture
def gribfile(request):
    return request.config.getoption('--grib-file')
