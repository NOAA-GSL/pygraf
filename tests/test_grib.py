# pylint: disable=invalid-name
''' Test suite for grib datahandler. '''

import datetime

import numpy as np
from matplotlib import colors as mcolors
import Nio

import adb_graphics.datahandler.grib as grib

def test_UPPData(natfile, prsfile):

    ''' Test the UPPData class methods on both types of input files. '''

    class UPP(grib.UPPData):

        ''' Test class needed to define the values as an abstract class '''

        def values(self, level=None, name=None, **kwargs):
            return 1

    upp_nat = UPP(natfile, short_name='temp')
    upp_prs = UPP(prsfile, short_name='temp')

    # Ensure appropriate typing and size (where applicable)
    for upp in [upp_nat, upp_prs]:
        assert isinstance(upp.anl_dt, datetime.datetime)
        assert isinstance(upp.clevs, np.ndarray)
        assert isinstance(upp.date_to_str(datetime.datetime.now()), str)
        assert isinstance(upp.fhr, str)
        assert isinstance(upp.field, Nio.NioVariable)
        assert isinstance(upp.latlons(), list)
        assert isinstance(upp.lev_descriptor, str)
        assert isinstance(upp.ncl_name(upp.vspec), str)
        assert isinstance(upp.numeric_level(), tuple)
        assert isinstance(upp.spec, dict)
        assert isinstance(upp.valid_dt, datetime.datetime)
        assert isinstance(upp.vspec, dict)
        # Test for appropriate date formatting
        test_date = datetime.datetime(2020, 12, 5, 12)
        assert upp.date_to_str(test_date) == '20201205 12 UTC'

def test_fieldData(prsfile):

    ''' Test the fieldData class methods on a prs file'''

    field = grib.fieldData(prsfile, level='500mb', short_name='temp')

    assert isinstance(field.cmap, mcolors.Colormap)
    assert isinstance(field.colors, np.ndarray)
    assert isinstance(field.corners, list)
    assert isinstance(field.ticks, int)
    assert isinstance(field.units, str)
    assert isinstance(field.values(), np.ndarray)
    assert isinstance(field.wind(True), list)
    assert len(field.corners) == 4
    assert len(field.wind(True)) == 2
    assert len(field.wind('850mb')) == 2
    for component in field.wind(True):
        assert isinstance(component, np.ndarray)

    # Test retrieving other values
    assert np.array_equal(field.values(), field.values(name='temp', level='500mb'))

    # Return zeros by subtracting same field
    diff = field.field_diff(field.values(), variable2='temp', level2='500mb')
    assert isinstance(diff, np.ndarray)
    assert not np.any(diff)

    # Windspeed
    windspeed = field.windspeed()
    u_wind, v_wind = field.wind(True)
    assert np.array_equal(windspeed, np.sqrt(u_wind**2 + v_wind**2))

    # Test transform
    assert np.array_equal(field.get_transform('conversions.k_to_f', field.values(), {}), \
                          (field.values() - 273.15) * 9/5 +32)

    field2 = grib.fieldData(prsfile, level='ua', short_name='ceil')
    transforms = field2.vspec.get('transform')
    transform_kwargs = field2.vspec.get('transform_kwargs', {})
    assert np.array_equal(field2.get_transform(transforms, field2.values(), transform_kwargs), \
                          field2.field_diff(field2.values(), variable2='gh', level2='sfc') / 304.8)

    # Expected size of values
    assert len(np.shape((field.values()))) == 2
    assert len(np.shape((field.values(name='u')))) == 2
    assert len(np.shape((field.values(name='u', level='850mb')))) == 2

def test_profileData(natfile):

    ''' Test the profileData class methods on a nat file'''

    loc = ' BNA   9999 99999  36.12  86.69  597 Nashville, TN\n'
    profile = grib.profileData(natfile, loc=loc, short_name='temp')

    assert isinstance(profile.get_xypoint(), tuple)
    assert isinstance(profile.values(), np.ndarray)

    # The values should return a single number (0) or a 1D array (1)
    assert len(np.shape((profile.values(level='best', name='li')))) == 0
    assert len(np.shape((profile.values(name='temp')))) == 1
