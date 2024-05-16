# pylint: disable=unused-argument,invalid-name
'''
This module contains functions for converting the units of a field. The
interface requires a single atmospheric field in a Numpy array, and returns the
converted values as output.
'''

import numpy as np

def k_to_c(field, **kwargs):

    ''' Conversion from Kelvin to Celsius '''

    return field - 273.15

def k_to_f(field, **kwargs):

    ''' Conversion from Kelvin to Farenheit '''

    return (field - 273.15) * 9/5 + 32

def kgm2_to_in(field, **kwargs):

    ''' Conversion from kg per m^2 to inches '''

    return field * 0.03937

def kg_to_g(field, **kwargs):

    ''' Conversion from kg to g '''

    return field * 1000.
    
def magnitude(a, b, **kwargs):

    ''' Return the magnitude of vector components '''

    return np.sqrt(np.square(a) + np.square(b))

def m_to_dm(field, **kwargs):

    ''' Conversion from meters to decameters '''

    return field / 10.

def m_to_in(field, **kwargs):

    ''' Conversion from meters to inches '''

    return field * 39.3701

def m_to_kft(field, **kwargs):

    ''' Conversion from meters to kilofeet '''

    return field / 304.8

def m_to_mi(field, **kwargs):

    ''' Conversion from meters to miles '''
    return field / 1609.344

def ms_to_kt(field, **kwargs):

    ''' Conversion from m s-1 to knots '''

    return field * 1.9438

def pa_to_hpa(field, **kwargs):

    ''' Conversion from Pascals to hectopascals '''

    return field / 100.

def percent(field, **kwargs):

    ''' Conversion from values between 0 - 1 to percent '''

    return field * 100.

def to_micro(field, **kwargs):

    ''' Convert field to micro '''

    return field * 1E6

def to_micrograms_per_m3(field, **kwargs):

    ''' Convert field to micrograms per cubic meter '''

    return field * 1E9

def vvel_scale(field, **kwargs):

    ''' Scale vertical velocity for plotting  '''

    return field * -10

def vort_scale(field, **kwargs):

    ''' Scale vorticity for plotting  '''

    return field / 1E-05

def weasd_to_1hsnw(field, **kwargs):

    ''' Conversion from snow wter equiv to snow (10:1 ratio) '''

    return field * 10.
