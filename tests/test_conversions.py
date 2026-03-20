import numpy as np
from pytest import fixture
from xarray import DataArray

from adb_graphics import conversions


@fixture
def array():
    return np.ones([3, 2]) * 300


def test_k_to_c(array):
    assert np.array_equal(conversions.k_to_c(array), array - 273.15)


def test_k_to_f(array):
    assert np.array_equal(conversions.k_to_f(array), (array - 273.15) * 9 / 5 + 32)


def test_kgm2_to_in(array):
    assert np.array_equal(conversions.kgm2_to_in(array), array * 0.03937)


def test_magnitude():
    ones = DataArray(np.ones([3, 2]))
    field1 = ones * 3
    field2 = ones * 4
    out = conversions.magnitude(field1, field2)
    assert np.array_equal(out, ones * 5)


def test_m_to_dm(array):
    assert np.array_equal(conversions.m_to_dm(array), array / 10.0)
    assert conversions.m_to_dm(array).dtype == np.float64


def test_m_to_in(array):
    assert np.array_equal(conversions.m_to_in(array), array * 39.3701)


def test_m_to_kft(array):
    assert np.array_equal(conversions.m_to_kft(array), array / 304.8)


def test_m_to_mi(array):
    assert np.array_equal(conversions.m_to_mi(array), array / 1609.344)


def test_ms_to_kt(array):
    assert np.array_equal(conversions.ms_to_kt(array), array * 1.9438)


def test_pa_to_hpa(array):
    assert np.array_equal(conversions.pa_to_hpa(array), array / 100)


def test_percent(array):
    assert np.array_equal(conversions.percent(array), array * 100)


def test_sden_to_slr(array):
    assert np.array_equal(conversions.sden_to_slr(array), 1000.0 / array)


def test_to_micro(array):
    assert np.array_equal(conversions.to_micro(array), array * 1e6)


def test_to_micrograms_per_m3(array):
    assert np.array_equal(conversions.to_micrograms_per_m3(array), array * 1e9)


def test_vvel_scale(array):
    assert np.array_equal(conversions.vvel_scale(array), array * -10)


def test_vort_scale(array):
    assert np.array_equal(conversions.vort_scale(array), array / 1e-05)


def test_weasd_to_1hsnw(array):
    assert np.array_equal(conversions.weasd_to_1hsnw(array), array * 10)
