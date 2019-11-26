import pytest
from matplotlib import cm, colors
from matplotlib import colors as mcolors
import numpy as np
from unittest.mock import Mock, patch

import adb_graphics.conversions as conversions
import adb_graphics.specs as specs
import adb_graphics.utils as utils



def test_conversion():

    ''' Test that conversions return at numpy array for input of np.ndarray,
    list, or int '''

    a = np.ones([3, 2]) * 300
    b = list(a)
    c = a[0,0]
    assert np.array_equal(conversions.k_to_c(a), a - 273.15)
    assert np.array_equal(conversions.m_to_dm(a), a / 10)
    assert np.array_equal(conversions.pa_to_hpa(a), a / 100)

    assert np.array_equal(conversions.k_to_c(b), a - 273.15)
    assert np.array_equal(conversions.m_to_dm(b), a / 10)
    assert np.array_equal(conversions.pa_to_hpa(b), a / 100)

    assert np.array_equal(conversions.k_to_c(c), a[0,0] - 273.15)
    assert np.array_equal(conversions.m_to_dm(c), a[0,0] / 10)
    assert np.array_equal(conversions.pa_to_hpa(c), a[0,0] / 100)

def test_utils():

    ''' Test that utils works appropriately. '''

    assert callable(utils.get_func('conversions.k_to_c'))


class MockSpecs(specs.VarSpec):

    ''' Mock class for the VarSpec abstract class '''

    @property
    def clevs(self):
        return list(range(15))


def test_specs():

    ''' Test VarSpec properties. '''
    config = 'adb_graphics/default_specs.yml'
    varspec = MockSpecs(config)

    assert isinstance(varspec.t_colors, np.ndarray)
    assert isinstance(varspec.ps_colors, np.ndarray)
    assert isinstance(varspec.yml, dict)

    assert np.shape(varspec.t_colors) == (len(varspec.clevs), 4)
    assert np.shape(varspec.ps_colors) == (105, 4)


class TestDefaultSpecs():

    ''' Test contents of default_specs.yml. '''
    config = 'adb_graphics/default_specs.yml'
    varspec = MockSpecs(config)

    cfg = varspec.yml

    @property
    def allowable(self):
        return {
            'clevs': self.is_a_clev,
            'cmap': self.is_a_cmap,
            'colors': self.is_a_color,
            'subst': self.is_a_key,
            'ticks': self.is_int,
            'transform': self.is_callable,
            'unit': self.is_string,
            'warm': self.is_a_clev,
            'cold': self.is_a_clev,
            }

    def is_a_clev(self, clev):

        ''' A clev can be a list, a range, or a callable function. '''

        if isinstance(clev, list):
            return True

        if 'range' in clev.split('[')[0]:
            clean = lambda x: x.strip().split('-')[-1].replace('.', '1')
            nums = [clean(i).isnumeric() for i in clev.split(' ', 1)[1].strip('[').strip(']').split(',')]
            return all(nums)

        return callable(utils.get_func(clev))

    def is_a_cmap(self, cmap):
        return isinstance(cm.get_cmap(cmap), colors.Colormap)

    def is_a_color(self, color):
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        return color in colors.keys() or self.is_callable(color)

    def is_a_key(self, key):
        return self.cfg.get(key) is not None

    def is_callable(self, func):
        return hasattr(self.varspec, func) or callable(utils.get_func(func))

    def is_int(self, i):
        if isinstance(i, int):
            return True
        return (i.isnumeric() and len(i.split('.')) == 1)

    def is_string(self, s):
        return isinstance(s, str)

    def check_keys(self, d):
        if not isinstance(d, dict):
            return
        for k, v in d.items():
            assert (k in self.allowable.keys()) or self.is_int(k)
            if isinstance(v, dict):
                self.check_keys(v)
            else:
                checker = self.allowable.get(k)
                assert checker(v)


    def test_keys(self):
        # Test that each variable (top level) has only allowed keys
        for var, specs in self.cfg.items():
            self.check_keys(specs)

