# pylint: disable=invalid-name

'''
Pytests for the common utilities included in this package. Includes:

    - conversions.py
    - specs.py
    - utils.py

To run the tests, type the following in the top level repo directory:

    python -m pytest --grib-file [path/to/gribfile]

'''

from string import ascii_letters

from matplotlib import cm
from matplotlib import colors as mcolors
from metpy.plots import ctables
import numpy as np

import adb_graphics.conversions as conversions
import adb_graphics.specs as specs
import adb_graphics.utils as utils
import adb_graphics.datahandler.grib as grib

def test_conversion():

    ''' Test that conversions return at numpy array for input of np.ndarray,
    list, or int '''

    a = np.ones([3, 2]) * 300
    b = list(a)
    c = a[0, 0]

    # Check for the right answer
    assert np.array_equal(conversions.k_to_c(a), a - 273.15)
    assert np.array_equal(conversions.k_to_f(a), (a - 273.15) * 9/5 + 32)
    assert np.array_equal(conversions.kgm2_to_in(a), a * 0.04)
    assert np.array_equal(conversions.m_to_dm(a), a / 10)
    assert np.array_equal(conversions.m_to_kft(a), a / 304.8)
    assert np.array_equal(conversions.ms_to_kt(a), a * 1.9438)
    assert np.array_equal(conversions.pa_to_hpa(a), a / 100)
    assert np.array_equal(conversions.vvel_scale(a), a * -10)
    assert np.array_equal(conversions.vort_scale(a), a / 1E-05)

    functions = [
        conversions.k_to_c,
        conversions.k_to_f,
        conversions.kgm2_to_in,
        conversions.m_to_dm,
        conversions.m_to_kft,
        conversions.ms_to_kt,
        conversions.pa_to_hpa,
        conversions.vvel_scale,
        conversions.vort_scale,
        ]

    # Check that all functions return a np.ndarray given a collection, or single float
    for f in functions:
        for collection in [b, c]:
            assert isinstance(f(collection), (float, np.ndarray))


class MockSpecs(specs.VarSpec):

    ''' Mock class for the VarSpec abstract class '''

    @property
    def clevs(self):
        return np.asarray(range(15))

    @property
    def vspec(self):
        return {}


def test_specs():

    ''' Test VarSpec properties. '''

    config = 'adb_graphics/default_specs.yml'
    varspec = MockSpecs(config)

    # Ensure correct return type
    assert isinstance(varspec.t_colors, np.ndarray)
    assert isinstance(varspec.ps_colors, np.ndarray)
    assert isinstance(varspec.yml, dict)

    # Ensure the appropriate number of colors is returned
    assert np.shape(varspec.t_colors) == (len(varspec.clevs), 4)
    assert np.shape(varspec.ps_colors) == (105, 4)


def test_utils():

    ''' Test that utils works appropriately. '''

    assert callable(utils.get_func('conversions.k_to_c'))


class TestDefaultSpecs():

    ''' Test contents of default_specs.yml. '''

    config = 'adb_graphics/default_specs.yml'
    varspec = MockSpecs(config)

    cfg = varspec.yml

    @property
    def allowable(self):

        ''' Each entry in the dict names a function that tests a key in
        default_specs.yml. '''

        return {
            'clevs': self.is_a_clev,
            'cmap': self.is_a_cmap,
            'colors': self.is_a_color,
            'contour': self.is_a_key,
            'contour_colors': self.is_a_color,
            'layer':self.is_int,
            'ncl_name': True,
            'ticks': self.is_number,
            'title': self.is_string,
            'transform': self.is_callable,
            'transform_kwargs': self.is_dict,
            'unit': self.is_string,
            'wind': self.is_wind,
            }

    @staticmethod
    def is_a_clev(clev):

        ''' Returns true for a clev that is a list, a range, or a callable function. '''

        if isinstance(clev, (list, np.ndarray)):
            return True

        if 'range' in clev.split('[')[0]:
            clean = lambda x: x.strip().split('-')[-1].replace('.', '1')
            items = clev.split(' ', 1)[1].strip('[').strip(']').split(',')
            nums = [clean(i).isnumeric() for i in items]
            return all(nums)

        return callable(utils.get_func(clev))

    @staticmethod
    def is_a_cmap(cmap):

        ''' Returns true for a cmap that is a Colormap object. '''
        return cmap in dir(cm) + list(ctables.colortables.keys())

    def is_a_color(self, color):

        ''' Returns true if color is contained in the list of recognized colors. '''

        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        return color in colors.keys() or self.is_callable(color)

    @staticmethod
    def is_a_level(key):

        '''
        Returns true if the key fits one of the level descriptor formats.

        Allowable formats include:

            [str_descriptor]     e.g. sfc, max, mup
            [numeric][lev_type]  e.g. 500mb, or 2m
            [stat][numeric]      e.g. mn02, mx25

        '''

        allowed_levels = [
            'esbl',    # ???
            'esblmn',  # ???
            'max',     # maximum in column
            'maxsfc',  # max surface value
            'mdn',     # maximum downward
            'mnsfc',   # min surface value
            'msl',     # mean sea level
            'mup',     # maximum upward
            'sfc',     # surface
            'ua',      # upper air
            ]

        allowed_lev_type = [
            'cm',      # centimeters
            'ds',      # difference
            'm',       # meters
            'mb',      # milibars
            ]

        allowed_stat = [
            'in',      # ???
            'm',       # ???
            'maxm',    # ???
            'mn',      # minimum
            'mx',      # maximum
            ]

        # Easy check first -- it is in the allowed_levels list
        if key in allowed_levels:
            return True

        # Check for [numeric][lev_type] pattern
        for lev in allowed_lev_type:
            ks = key.split(lev)

            # If the lev didn't appear in the key, length of list is 1.
            # If the lev didn't match exactly, the second element will the remainder of the string
            if len(ks) == 2 and len(ks[1]) == 0:
                numeric = ks[0].isnumeric()
                allowed = ''.join([c for c in key if c in ascii_letters]) in allowed_lev_type

                if numeric and allowed:
                    return True

        # Check for [stat][numeric]
        for stat in allowed_stat:
            ks = key.split(stat)
            if len(ks) == 2 and len(ks[0]) == 0:

                numeric = ks[1].isnumeric()
                allowed = ''.join([c for c in key if c in ascii_letters]) in allowed_stat

                if numeric and allowed:
                    return True

        return False

    def is_a_key(self, key):

        ''' Returns true if key exists as a key in the config file. '''

        return self.cfg.get(key) is not None

    @staticmethod
    def is_bool(k):

        ''' Returns true if k is a boolean variable. '''

        return isinstance(k, bool)

    def is_callable(self, funcs):

        ''' Returns true if func in funcs list is the name of a callable function. '''

        in_varspec = False
        in_grib = False
        in_package = False

        funcs = funcs if isinstance(funcs, list) else [funcs]
        callables = []

        for func in funcs:
            # Check datahandler.grib objects if a single word is provided
            if len(func.split('.')) == 1:
                for attr in dir(grib):
                    if func in dir(grib.__getattribute__(attr)):
                        callables.append(True)

            else:

                in_package = callable(utils.get_func(func))
                in_varspec = hasattr(self.varspec, func)
                callables.append(in_package or in_varspec)

        return all(callables)

    @staticmethod
    def is_dict(d):

        ''' Returns true if d is a dictionary '''

        return is_instance(d, dict)

    @staticmethod
    def is_int(i):

        ''' Returns true if i is an integer. '''

        if isinstance(i, int):
            return True
        return i.isnumeric() and len(i.split('.')) == 1

    @staticmethod
    def is_number(i):

        ''' Returns true if i is a number. '''

        if isinstance(i, (int, float)):
            return True
        return i.isnumeric() and len(i.split('.')) == 1

    @staticmethod
    def is_string(s):

        ''' Returns true if s is a string. '''

        return isinstance(s, str)

    def is_wind(self, wind):

        ''' Returns true if wind is a bool or is_a_level. '''

        return isinstance(wind, bool) or self.is_a_level(wind)

    def check_keys(self, d, depth=0):

        ''' Helper function that recursively checks the keys in the dictionary by calling the
        function defined in allowable. '''

        max_depth = 2

        # Only proceed if d is a dictionary
        if not isinstance(d, dict):
            return

        # Proceed only up to max depth.
        if depth >= max_depth:
            return
        else:
            level = depth+1

        for k, v in d.items():
            assert (k in self.allowable.keys()) or self.is_a_level(k)
            if isinstance(v, dict):
                self.check_keys(v, depth=level)
            else:
                checker = self.allowable.get(k)
                if isinstance(checker, bool):
                    assert checker
                else:
                    assert checker(v)

    def test_keys(self):

        ''' Tests each of top-level variables in the config file by calling the helper function. '''

        for spec in self.cfg.values():
            self.check_keys(spec)
