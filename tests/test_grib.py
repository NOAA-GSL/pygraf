# pylint: disable=invalid-name
''' Test suite for grib datahandler. '''

import datetime
import tempfile

import numpy as np
from matplotlib import colors as mcolors
import pygrib
import yaml

import adb_graphics.datahandler.grib as grib

def test_get_fields(gribfile):

    ''' Ensure that get_fields returns a pygrib message '''

    grib_file = grib.GribFile(gribfile)
    field = grib_file.get_fields(level=500, lev_type='isobaricInhPa', short_name='t')
    assert isinstance(field, pygrib.gribmessage) # pylint: disable=c-extension-no-member


def test_UPPData(gribfile):

    ''' Test the UPPData class methods. '''

    upp = grib.UPPData(gribfile, level=500, lev_type='isobaricInhPa', short_name='t')

    # Ensure appropriate typing and size (where applicable)
    assert isinstance(upp.anl_dt, datetime.datetime)
    assert isinstance(upp.clevs, list)
    assert isinstance(upp.cmap, mcolors.Colormap)
    assert isinstance(upp.colors, np.ndarray)
    assert isinstance(upp.corners, list)
    assert isinstance(upp.data, pygrib.gribmessage) # pylint: disable=c-extension-no-member
    assert isinstance(upp.date_to_str(datetime.datetime.now()), str)
    assert isinstance(upp.fhr, str)
    assert isinstance(upp.lev_unit, str)
    assert isinstance(upp.spec, dict)
    assert isinstance(upp.ticks, int)
    assert isinstance(upp.units, str)
    assert isinstance(upp.valid_dt, datetime.datetime)
    assert isinstance(upp.values, np.ndarray)
    assert isinstance(upp.wind, list)
    assert len(upp.corners) == 4
    assert len(upp.wind) == 2
    for component in upp.wind:
        assert isinstance(component, np.ndarray)

    # Test for appropriate date formatting
    test_date = datetime.datetime(2020, 12, 5, 12)
    assert upp.date_to_str(test_date) == '20201205 12 UTC'

def test_UPPData_load_spec(gribfile):

    ''' Test the _load_spec method of UPPData thoroughly. Also check methods
    that interact with UPPData.spec '''

    test_dict = {
        'a': {
            'clevs': 'range [0, 13, 4]',
            'colors': 'ps_colors',
            'ticks': 8,
            'transform': 'conversions.k_to_c',
            'unit': 'C'
            },
        'b': {
            'subst': 'a',
            'cmap': 'jet',
            'transform': 'None',
            'unit': 'None',
            },
        't': {
            250: {
                'colors': 't_colors',
                'subst': 'b',
                'unit': 'K',
                },
            },
        }

    expected_dict = {
        'clevs': 'range [0, 13, 4]',
        'colors': 't_colors',
        'cmap': 'jet',
        'ticks': 8,
        'transform': 'None',
        'unit': 'K',
        }

    # Create a temporary yaml config file with the config dict above.
    with tempfile.NamedTemporaryFile(mode='w') as tmpf:
        yaml.dump(test_dict, tmpf, default_flow_style=False)
        upp = grib.UPPData(gribfile,
                           config=tmpf.name,
                           level=250,
                           lev_type='isobaricInhPa',
                           short_name='t',
                           )

        # Check that the subst key appropriately substitutes
        assert upp.spec == expected_dict

        # Check clev, cmap, colors, ticks, units, transform
        expected_clevs = list(range(0, 13, 4))
        expected_n = len(expected_clevs)
        assert upp.clevs == expected_clevs
        assert len(upp.colors) >= expected_n
        assert upp.ticks == 8
        assert upp.units == 'K'
        assert np.amax(upp.values) > 200

        # Check results when applying a transform to the data
        upp.spec['transform'] = 'conversions.k_to_c'
        assert np.amin(upp.values) < 0
