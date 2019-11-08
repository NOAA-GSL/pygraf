'''
This module contains functions for converting the units of a field. The
interface requires a single atmospheric field in a Numpy array, and returns the
converted values as output.
'''

import numpy as np

def k_to_c(field: np.ndarray) -> np.ndarray:

    ''' Conversion from Kelvin to Celsius '''

    return field - 273.15

def m_to_dm(field: np.ndarray) -> np.ndarray:

    ''' Conversion from meters to decameters '''

    return field / 10.

def pa_to_hpa(field: np.ndarray) -> np.ndarray:

    ''' Conversion from Pascals to hectopascals '''

    return field / 100.
