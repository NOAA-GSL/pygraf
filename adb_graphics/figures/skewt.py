from collections import OrderedDict
from functools import lru_cache
import pygrib
import numpy as np
import Nio
import pandas as pd

import adb_graphics.datahandler.grib as grib
import adb_graphics.figures.maps as maps
import adb_graphics.conversions as conversions
import adb_graphics.utils as utils
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator, FixedLocator, NullLocator

import metpy.calc as mpcalc

from metpy.plots import Hodograph, SkewT
from metpy.units import units
from metpy.future import precipitable_water

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class skewT(grib.profileData):

    def __init__(self, filename, loc, **kwargs):

        # Initialize on the temperature field since we need to gather
        # field-specific data from this object, e.g. dates, lat, lon, etc.

        super().__init__(filename=filename,
                         loc=loc,
                         short_name='temp',
                         **kwargs,
                         )

        self.max_plev = kwargs.get('max_plev', 0)

    @property
    @lru_cache()
    def atmo_profiles(self):

        ''' Return a dictionary of atmospheric data profiles for each variable
        needed by the skewT. '''

        # OrderedDict because we need to get pressure profile first.
        atmo_vars = OrderedDict({
                'pres': {
                    'transform': 'hectoPa',
                    'units': units.Pa,
                    },
                'gh': {
                    'units': units.gpm,
                    },
                'sphum': {
                    'units': units.dimensionless,
                    },
                'temp': {
                    'transform': 'degF',
                    'units': units.degK,
                    },
                'u': {
                    'transform': 'knots',
                    'units': units.meter_per_second,
                    },
                'v': {
                    'transform': 'knots',
                    'units': units.meter_per_second,
                    },
           })

        top = None
        for var, items in atmo_vars.items():

            # Get the profile values and attach MetPy units
            tmp = np.asarray(self.values(short_name=var)) * items['units']

            # Apply any needed transdecimals
            transform = items.get('transform')
            if transform:
                tmp = tmp.to(transform)

            # Only return values up to the maximum pressure level requested
            if var == 'pres' and top is None:
                top = np.sum(np.where(tmp.magnitude >= self.max_plev)) - 1

            atmo_vars[var]['data'] = tmp[:top]

        return atmo_vars

    @property
    @lru_cache()
    def thermo_variables(self):

        '''
        Return an ordered dictionary of thermodynamic variables needed for the skewT.
        Ordered because we want to print these values in this order on the SkewT
        diagram.
        '''

        thermo = OrderedDict({
            'cape': { # Convective available potential energy
              'units': 'J/kg',
              'level': 'sfc',
              },
            'cin': { # Convective inhibition
              'units': 'J/kg',
              'level': 'sfc',
              },
            'mucape': { # Most Unstable CAPE
              'units': 'J/kg',
              'level': 'mu',
              'variable': 'cape',
              },
            'mucin': { # CIN from MUCAPE level
              'units': 'J/kg',
              'level': 'mu',
              'variable': 'cin',
              },
            'li': { # Lifted Index
              'decimals': 1,
              'units': 'K',
              'level': 'sfc',
              },
            'bli': { # Best Lifted Index
              'decimals': 1,
              'units': 'K',
              'level': 'best',
              'variable': 'li',
              },
            'lcl': { # Lifted Condensation Level
              'units': 'm',
              },
            'lpl': { # Lifted Parcel Level
              'units': 'hPa',
              'transform': conversions.pa_to_hpa,
              },
            'srh03': { # 0-3 km Storm relative helicity
              'units': 'm2/s2',
              'level': 'sr03',
              'variable': 'hlcy',
              },
            'srh01': { # 0-1 km Storm relative helicity
              'units': 'm2/s2',
              'level': 'sr01',
              'variable': 'hlcy',
              },
            'shr06': { # 0-6 km Shear
              'units': 'kt',
              'level': '06km',
              'variable': 'shear',
              },
            'shr01': { # 0-1 km Shear
              'units': 'kt',
              'level': '01km',
              'variable': 'shear',
              },
            'cell': { # Cell motion
              'units': 'kt',
              'transform': conversions.ms_to_kt,
              },
            'pwtr': { # Precipitable water
              'decimals': 1,
              'units': 'mm',
              'level': 'sfc',
              },
            })

        for var, items in thermo.items():

            varname = items.get('variable', var)
            tmp = self.values(lev=items.get('level', 'ua'), short_name=varname)

            transform = items.get('transform')
            if transform:
                tmp = transform(tmp)

            thermo[var]['data'] = tmp

        return thermo

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
        for name, items in self.thermo_variables.items():
            decimals = items.get('decimals', 0)
            value = items['data']
            value = round(int(value)) if decimals == 0 else round(value, decimals)
            line = f"{name.upper():<7s}: {str(value):>10} {items['units']}"
            lines.append(line)

        contents = '\n'.join(lines)

        skew.ax.text(0.65, 0.98, contents,
                     transform=skew.ax.transAxes,
                     size=8,
                     fontproperties=fm.FontProperties(family='monospace'),
                     verticalalignment='top',
                     bbox=dict(facecolor='white',edgecolor='black', alpha=0.7))

    def _plot_hodograph(self, skew):

        # Alias the wind data proviles
        u = self.atmo_profiles.get('u', {}).get('data')
        v = self.atmo_profiles.get('v', {}).get('data')
        speed = self.vector_magnitude(u, v)

        # Create an inset axes object that is 30% width and height of the
        # figure and put it in the upper left hand corner.

        ax = inset_axes(skew.ax, '28%', '28%', loc=2)
        h = Hodograph(ax, component_range=80.,)
        h.add_grid(increment=20,)
        #        bbox=dict(facecolor='white',edgecolor='black', alpha=0.7))

        # Plot a line colored by wind speed
        h.plot_colormapped(u, v, speed)

    def _plot_labels(self, skew):

        skew.ax.set_xlabel('Temperature (F)')
        skew.ax.set_ylabel('Pressure (hPa)')

    def _plot_profile(self, skew):

        profiles = self.atmo_profiles
        pres = profiles.get('pres').get('data')
        temp = profiles.get('temp').get('data')
        sphum = profiles.get('sphum').get('data')

        dewpt = mpcalc.dewpoint_from_specific_humidity(sphum, temp, pres).to('degF')

        # Pressure vs temperature
        skew.plot(pres, temp, 'r', linewidth=1.5)

        # Pressure vs dew point temperature
        skew.plot(pres, dewpt, 'blue', linewidth=1.5)

        # Compute parcel profile and plot it
        parcel_profile = mpcalc.parcel_profile(pres, temp[0],
                dewpt[0]).to('degC')

        skew.plot(pres, parcel_profile, 'orange', linewidth=1.2,
        linestyle='dashed')

    def _plot_wind_barbs(self, skew):

        # Pressure vs wind
        skew.plot_barbs(self.atmo_profiles.get('pres', {}).get('data'),
                        self.atmo_profiles.get('u', {}).get('data'),
                        self.atmo_profiles.get('v', {}).get('data'),
                color='blue',
                linewidth=0.2,
                y_clip_radius=0,
                )

    def _setup_skewT(self):

        fig = plt.figure(figsize=(12, 12))
        skew = SkewT(fig, rotation=45)

        ticks = [str(int(t)) for t in skew.ax.get_xticks()]

        skew.ax.set_ylim(1050, self.max_plev)
        skew.ax.set_xlim(-35, 50)
        labels_F = list(range(-20, 125, 20)) * units.degF
        labels = labels_F.to('degC').magnitude
        skew.ax.xaxis.set_minor_locator(FixedLocator(labels))
        skew.ax.set_xticklabels(ticks[1:-2])
        skew.ax.set_xticklabels(labels_F.magnitude, minor=True)
        skew.ax.tick_params(which='minor', length=8)
        skew.ax.tick_params(axis='x', which='major', length=0, labelbottom=True,
                labeltop=True, labelright=True, labelrotation=45, pad=-25,
                labelcolor='gray')

        # Add the relevant special lines

        dry_adiabats = np.arange(-40, 210, 10) * units.degC
        skew.plot_dry_adiabats(dry_adiabats,
                colors='tan',
                linestyles='solid',
                linewidth=0.7,
                )

        moist_adiabats = np.arange(8, 36, 4) * units.degC
        moist_pr = np.arange(1001, 220, -10) * units.hPa
        adiabats = skew.plot_moist_adiabats(moist_adiabats, moist_pr,
                colors='green',
                linestyles='solid',
                linewidth=0.7,
                )

        mixing_lines = np.array([1, 2, 3, 5, 8, 12, 16, 20]).reshape(-1, 1)  / 1000
        mix_pr = np.arange(1001, 400, -50) * units.hPa
        skew.plot_mixing_lines(w=mixing_lines, p=mix_pr,
                colors='green',
                linestyles=(0, (5, 10)),
                linewidth=0.7,
                )
        return skew


    def _title(self):

         ''' Creates standard annotation for a skew-T. '''

         atime = self.date_to_str(self.anl_dt)
         vtime = self.date_to_str(self.valid_dt)

         # Top Left
         plt.title(f"Analysis: {atime}\nFcst Hr: {self.fhr}", loc='left',
                 fontsize=16, position=(0, 1.03))

         # Top Right
         plt.title(f"Valid: {vtime}", loc='right', position=(1, 1.03), fontsize=16)

         # Center
         site = f"{self.site_code} {self.site_num} {self.site_name}"
         site_loc = f"{self.site_lat},  {self.site_lon}"
         site_title = f"{site} at nearest grid pt over land {site_loc}"
         plt.title(site_title, loc='center', fontsize=12)

