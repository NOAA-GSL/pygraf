import pygrib
import numpy as np
import Nio
import pandas as pd

import adb_graphics.datahandler.grib as grib
import adb_graphics.figures.maps as maps
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.font_manager as fm

import metpy.calc as mpcalc

from metpy.plots import Hodograph, SkewT
from metpy.units import units
from metpy.future import precipitable_water

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class skewT():

    def __init__(filename, loc_name, point):

        self.filename = filename
        self.point = point
        self.loc_name = loc_name

    def atmo_profiles(self):

        atmo_vars = [
            'pres',
            'gh',
            'temp',
            'sphum',
            'u',
            'v',
            'dewpt',
           ]

        ret = {}

        if var in atmo_vars:

            ret[var] = grib.profileData(
                    filename=self.filename,
                    loc_name=self.loc_name,
                    profile_loc=self.point,
                    short_name=var,
                    )
        return ret

    def thermo_variables(self):

        thermo = 
'cape': {
  'format': '10.0f',
  'units': 'J/kg',
'cin':   {:10.0f} J/kg
'mucape':{:10.0f} J/kg
'mucin': {:10.0f} J/kg
'li':    {:10.1f} K
'bli':   {:10.1f} K
'lcl':   {:10.0f} m
'lpl':   {:10.0f} hPa
'srh03': {:10.0f} m2/s2
'srh01': {:10.0f} m2/s2
'shr06': {:10.0f} kt
'shr01': {:10.0f} kt
'cell':  {:10.0f} kt
'pw':    {:10.1f} mm

       for var in thermo.keys():

            ret[var]['data'] = grib.profileData(
                    filename=self.filename,
                    loc_name=self.loc_name,
                    profile_loc=self.point,
                    short_name=var,
                    )


    def 
           

