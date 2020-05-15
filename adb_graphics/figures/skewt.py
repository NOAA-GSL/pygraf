from collections import OrderedDict
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

    def __init__(filename, loc, **kwargs):

        super().__init__(self,
                         filename,
                         loc=loc,
                         short_name='temp',
                         **kwargs
                         )

        self.max_plev = kwargs.get('max_plev', 0)

        self.atmo = self.atmo_profiles()
        self.thermo = self.thermo_variables()


    def atmo_profiles(self):

        atmo_vars = OrderedDict({
                'pres': {
                    'transform': 'hecto-Pa',
                    'units': units.Pa,
                    },
                'gh': {
                    'units': units.gpm,
                    },
                'temp': {
                    'transform': 'degF',
                    'units': units.degC,
                    },
                'sphum': {
                    'units': units.dimensionless,
                    },
                'u': {
                    'transform': 'knots',
                    'units': units.meter_per_second,
                    },
                'v': {
                    'transform': 'knots',
                    'units': units.meter_per_second,
                    },
                'dewpt': {
                    'transform': 'degF',
                    'units': units.degC,
                    },
           })

        top = None
        ret = {}
        if var, items in atmo_vars.items():

            tmp = self.values(short_name=var) * items['units']

            tranform = items.get('transform'):
            if transform:
                tmp = tmp.to(transform)

            if var == 'pres' and not top:
                top = np.sum(np.where(tmp >= self.max_plev)) - 1

            ret[var]['data'] = tmp[:top]

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

        skew = self._setup_skewT()
        self._title()
        self._plot_profile(skew)
        self._plot_wind_barbs(skew)
        self._plot_labels(skew)

        self._plot_hodograph(skew)

        self._add_thermo_inset(skew)

    def _add_thermo_inset(self, skew):


        lines = []
        for name, items in self.thermo.items():
            line = "{name.upper():<7c}: {data:10.0f} {units}".format(**items)
            lines.append(line)

        contents = '\n'.join(lines)

        skew.ax.text(0.65, 0.98, contents,
                     transform=skew.ax.transAxes,
                     size=8,
                     fontproperties=fm.FontProperties(family='monospace'),
                     verticalalignment='top',
                     bbox=dict(facecolor='white',edgecolor='white'))

    def _plot_hodograph(self, skew):

        # Alias the wind data proviles
        u = self.atmo.get('u')
        v = self.atmo.get('v')
        speed = self.vector_magnitude(u, v)

        # Create an inset axes object that is 30% width and height of the
        # figure and put it in the upper left hand corner.

        ax = inset_axes(skew.ax, '30%', '30%', loc=2)
        h = Hodograph(ax, component_range=80.)
        h.add_grid(increment=20)

        # Plot a line colored by wind speed
        h.plot_colormapped(u, v, speed)


    def _plot_labels(self, skew):

        skew.ax.set_xlabel('Temperature (F)')
        skew.ax.set_ylabel('Pressure (hPa)')

    def _plot_profile(self, skew):


        # Pressure vs temperature
        skew.plot(self.atmo.get('pres'), self.atmo.get('temp'),
                'r',
                linewidth=1,
                )

        # Pressure vs dew point temperature
        skew.plot(self.atmo.get('pres'), self.atmo.get('dewpt'),
                'blue',
                linewidth=1,
                )

    def _plot_wind_barbs(self, skew):

       # Pressure vs wind
       skew.plot_barbs(self.atmo.get('pres'), self.atmo.get('u'), self.atmo.get('v'),
               color='blue',
               linewidth=0.2,
               )

    def _setup_skewT(self):

        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig, rotation=45)

        skew.ax.set_ylim(1050, 100)
        skew.ax.set_xlim(-35, 50)

        # Add the relevant special lines

        dry_adiabats = np.arange(-40, 210, 10) * units.degC
        skew.plot_dry_adiabats(dry_adiabats,
                colors='tan',
                linestyles='solid',
                linewidth=0.7,
                )

        moist_adiabats = np.arange(8, 36, 4) * units.degC
        moist_pr = np.arange(1050, 220, -10) * units.hPa
        skew.plot_moist_adiabats(moist_adiabats, moist_pr,
                colors='green',
                linestyles='solid',
                linewidth=0.7,
                )

        mixing_lines = [1, 2, 3, 5, 8, 12, 16, 20] * units.dimensionless
        mix_pr = np.arange(1050, 400, -50) * units.hPa
        skew.plot_mixing_lines(
                colors='green',
                linewidth=0.7,
                )


   def _title(self):

        ''' Creates standard annotation for a skew-T. '''

        atime = self.date_to_str(self.anl_dt)
        vtime = self.date_to_str(self.valid_dt)

        # Top Left
        plt.title(f"Analysis: {atime}\nFcst Hr: : {self.fhr}", pos=(0, 1.04), fontsize=16)

        # Top Right
        plt.title(f"Valid: {vtime}", pos=(1, 1.04), horizontal_alignment='right', fontsize=16)

        # Center
        site = f"{self.site_code} {self.site_num} {self.site_name}"
        site_loc = f"{self.site_lat} {self.site_lon}"
        site_title = f"{site} at nearest grid pt over land {site_loc}"
        plt.title(site_title, loc='center', fontsize=12)

