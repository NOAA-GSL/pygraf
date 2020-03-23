# pylint: disable=invalid-name
''' Test suite for grib datahandler. '''

import datetime
import tempfile

import numpy as np
from matplotlib import colors as mcolors
import pygrib
import Nio
import yaml

import adb_graphics.datahandler.grib as grib

def test_UPPData(gribfile):

    ''' Test the UPPData class methods. '''

    upp = grib.UPPData(gribfile, level='500mb', short_name='temp')

    # Ensure appropriate typing and size (where applicable)
    assert isinstance(upp.anl_dt, datetime.datetime)
    assert isinstance(upp.clevs, np.ndarray)
    assert isinstance(upp.cmap, mcolors.Colormap)
    assert isinstance(upp.colors, np.ndarray)
    assert isinstance(upp.corners, list)
    assert isinstance(upp.date_to_str(datetime.datetime.now()), str)
    assert isinstance(upp.fhr, str)
    assert isinstance(upp.field, Nio.NioVariable)
    assert isinstance(upp.lev_descriptor, str)
    assert isinstance(upp.spec, dict)
    assert isinstance(upp.ticks, int)
    assert isinstance(upp.units, str)
    assert isinstance(upp.valid_dt, datetime.datetime)
    assert isinstance(upp.values(), np.ndarray)
    assert isinstance(upp.wind, list)
    assert len(upp.corners) == 4
    assert len(upp.wind) == 2
    for component in upp.wind:
        assert isinstance(component, np.ndarray)

    # Test for appropriate date formatting
    test_date = datetime.datetime(2020, 12, 5, 12)
    assert upp.date_to_str(test_date) == '20201205 12 UTC'

