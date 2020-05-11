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


class skewT(grib.profileData):

    def __init__(figure, filename, loc, **kwargs):

        super().__init__(self,
                         filename,
                         loc=loc,
                         short_name='temp',
                         **kwargs
                         )

        self.figure = figure

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
            ret[var] = self.values(short_name=var)

        return ret

    def thermo_variables(self):

        thermo = 
            'cape': {
              'format': '10.0f',
              'units': 'J/kg',
              },
            'cin': {
              'format': '10.0f',
              'units': 'J/kg',
              },
            'mucape': {
              'format': '10.0f',
              'units': 'J/kg',
              'vertical_lev': 2,
              },
            'mucin': {
              'format': '10.0f',
              'units': 'J/kg',
              'vertical_lev': 2,
              },
            'li': {
              'format': '10.1f',
              'units': 'K',
              },
            'bli': {
              'format': '10.1f',
              'units': 'K',
              },
            'lcl': {
              'format': '10.0f',
              'units': 'm',
              },
            'lpl': {
              'format': '10.0f',
              'units': 'hPa',
              'transform': conversions.pa_to_hpa,
              },
            'storm_rel_hlcy': {
              'format': '10.0f',
              'units': 'm2/s2',
              'vertical_lev': [0, 1],
              },
            'vshear': {
              'format': '10.0f',
              'units': 'kt',
              'vertical_lev': [0, 1],
              },
            'ushear': {
              'format': '10.0f',
              'units': 'kt',
              'vertical_lev': [0, 1],
              },
            'pw': {
              'format': '10.1f',
              'units': 'mm',
              },
            'u_storm_motion': {
              'format': '10.0f',
              'units': 'kt',
              'transform': conversions.ms_to_kt,
              },
            'v_storm_motion': {
              'format': '10.0f',
              'units': 'kt',
              'transform': conversions.ms_to_kt,
              },
            }

       for var, items in thermo.items():
            tmp = self.values(lev=items.get('vertical_lev'), name=var)

            tranform = items.get('transform'):
            if transform:
                tmp = utils.get_func(transform)(tmp)

            ret[var]['data'] = tmp

       return ret

   def create_skewT(self, **kwargs):

       skew = SkewT(self.fig, rotation=45, **kwargs)

   def create_title(self):
       init_date = self.date_to_str(self.anl_dt)
       fhr = self.fhr
       valid = self.date_to_str(self.valid_dt)
       site_code = 
       title = f"{init_data} {fhr} hr fcst      Valid {valid}\n"\
        "{self.site_code} {self.site_num} {self.site_name} at nearest HRRR grid
        pt over land {self.site_lat} {self.site_lon}"
