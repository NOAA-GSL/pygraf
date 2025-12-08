"""
This module contains functions for converting the units of a field. The
interface requires a single atmospheric field in a Numpy array, and returns the
converted values as output.
"""

from numpy import ndarray
from xarray import DataArray
from xarray.ufuncs import sqrt, square


def k_to_c(field: ndarray, **_kwargs):
    """Conversion from Kelvin to Celsius."""

    return field - 273.15


def k_to_f(field: ndarray, **_kwargs):
    """Conversion from Kelvin to Fahrenheit."""
    return (field - 273.15) * 9 / 5 + 32


def kgm2_to_in(field: ndarray, **_kwargs):
    """Conversion from kg per m^2 to inches."""

    return field * 0.03937


def magnitude(a: DataArray, b: DataArray, **_kwargs) -> DataArray:
    """Return the magnitude of vector components."""

    return DataArray(sqrt(square(a) + square(b)))


def m_to_dm(field: ndarray, **_kwargs):
    """Conversion from meters to decameters."""

    return field / 10.0


def m_to_in(field: ndarray, **_kwargs):
    """Conversion from meters to inches."""

    return field * 39.3701


def m_to_kft(field: ndarray, **_kwargs):
    """Conversion from meters to kilofeet."""

    return field / 304.8


def m_to_mi(field: ndarray, **_kwargs):
    """Conversion from meters to miles."""
    return field / 1609.344


def ms_to_kt(field: ndarray, **_kwargs):
    """Conversion from m s-1 to knots."""

    return field * 1.9438


def pa_to_hpa(field: ndarray, **_kwargs):
    """Conversion from Pascals to hectopascals."""

    return field / 100.0


def percent(field: ndarray, **_kwargs):
    """Conversion from values between 0 - 1 to percent."""

    return field * 100.0


def sden_to_slr(field: ndarray, **_kwargs):
    """Convert snow density (kg m-3) to snow-liquid ratio."""

    return 1000.0 / field


def to_micro(field: ndarray, **_kwargs):
    """Convert field to micro."""

    return field * 1e6


def to_micrograms_per_m3(field: ndarray, **_kwargs):
    """Convert field to micrograms per cubic meter."""

    return field * 1e9


def vvel_scale(field: ndarray, **_kwargs):
    """Scale vertical velocity for plotting."""

    return field * -10


def vort_scale(field: ndarray, **_kwargs):
    """Scale vorticity for plotting."""

    return field / 1e-05


def weasd_to_1hsnw(field: ndarray, **_kwargs):
    """Conversion from snow water equiv to snow (10:1 ratio)."""

    return field * 10.0
