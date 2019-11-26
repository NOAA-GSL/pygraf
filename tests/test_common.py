import pytest
import numpy as np

import adb_graphics.conversions as conversions
import adb_graphics.specs as specs



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


def test_specs():

    '''Test that the methods in specs.py work as expected. '''

    pass
