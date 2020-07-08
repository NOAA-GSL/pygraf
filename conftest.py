'''
Add command line options to the pytest suite.

Each CLA needs to be defined in pytest_addoption and to have a pytest.fixture
function defined.
'''

import pytest


def pytest_addoption(parser):

    ''' Define command line arguments to be parsed. '''

    parser.addoption('--nat-file',
                     action='store',
                     help='Path to nat-file.',
                     )

    parser.addoption('--prs-file',
                     action='store',
                     help='Path to prs-file.',
                     )

@pytest.fixture
def natfile(request):

    ''' Interface to  pass a grib file to pytest'''

    return request.config.getoption('--nat-file')

@pytest.fixture
def prsfile(request):

    ''' Interface to  pass a grib file to pytest'''

    return request.config.getoption('--prs-file')
