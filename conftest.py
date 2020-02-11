'''
Add command line options to the pytest suite.

Each CLA needs to be defined in pytest_addoption and to have a pytest.fixture
function defined.
'''

import pytest


def pytest_addoption(parser):

    ''' Define command line arguments to be parsed. '''

    parser.addoption('--grib-file',
                     action='store',
                     help='Path to grib-file.',
                     )

@pytest.fixture
def gribfile(request):

    ''' Interface to  pass a grib file to pytest'''

    return request.config.getoption('--grib-file')
