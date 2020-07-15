# pylint: disable=invalid-name
'''
The module the contains the SkewTDiagram class responsible for creating a Skew-T
Log-P diagram using MetPy.
'''

from collections import OrderedDict
from functools import lru_cache
import numpy as np

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import metpy.calc as mpcalc
from metpy.plots import Hodograph, SkewT
from metpy.units import units
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import adb_graphics.datahandler.grib as grib
import adb_graphics.errors as errors
import adb_graphics.utils as utils

class SkewTDiagram(grib.profileData):

    ''' The class responsible for gathering all data needed from a grib file to
    produce a Skew-T Log-P diagram.

    Input:

      filename         the full path to the grib file
      loc              the entire line entry of the sites file.

    Key word arguments:

      max_plev         maximum pressure level to plot in mb

    Additional keyword arguments for the grib.profileData base class should also
    be included.
    '''

    def __init__(self, grib_data, loc, **kwargs):

        # Initialize on the temperature field since we need to gather
        # field-specific data from this object, e.g. dates, lat, lon, etc.

        super().__init__(grib_data=grib_data,
                         loc=loc,
                         short_name='temp',
                         **kwargs,
                         )

        self.max_plev = kwargs.get('max_plev', 0)

    def _add_thermo_inset(self, skew):

        # Build up the text that goes in the thermo-dyniamics box
        lines = []
        for name, items in self.thermo_variables.items():

            # Magic to get the desired number of decimals to appear.
            decimals = items.get('decimals', 0)
            value = items['data']
            value = round(int(value)) if decimals == 0 else round(value, decimals)

            # Sure would have been nice to use a variable in the f string to
            # denote the format per variable.
            line = f"{name.upper():<7s}: {str(value):>10} {items['units']}"
            lines.append(line)

        contents = '\n'.join(lines)

        # Draw the text box
        skew.ax.text(0.70, 0.98, contents,
                     bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
                     fontproperties=fm.FontProperties(family='monospace'),
                     size=8,
                     transform=skew.ax.transAxes,
                     verticalalignment='top',
                     )

    @property
    @lru_cache()
    def atmo_profiles(self):

        '''
        Return a dictionary of atmospheric data profiles for each variable
        needed by the skewT.

        Each of these variables must be have units set appropriately for use
        with MetPy SkewT. Handle those units and conversions here since it
        differs from the requirements of other graphics units/transforms.
        '''

        # OrderedDict because we need to get pressure profile first. Entries in
        # the dict are as follows:
        #
        #   Variable short name:   consistent with default_specs.yml
        #      transform:          units string to pass to MetPy's to() function
        #      units:              the end unit of the field (after transform,
        #                          if applicable).
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
            tmp = np.asarray(self.values(name=var)) * items['units']

            # Apply any needed transdecimals
            transform = items.get('transform')
            if transform:
                tmp = tmp.to(transform)

            # Only return values up to the maximum pressure level requested
            if var == 'pres' and top is None:
                top = np.sum(np.where(tmp.magnitude >= self.max_plev)) - 1

            atmo_vars[var]['data'] = tmp[:top]

        return atmo_vars

    def create_diagram(self):

        ''' Calls the private methods for creating each component of the SkewT
        Diagram. '''

        skew = self._setup_diagram()
        self._title()
        self._plot_profile(skew)
        self._plot_wind_barbs(skew)
        self._plot_labels(skew)

        self._plot_hodograph(skew)
        self._add_thermo_inset(skew)

    def _plot_hodograph(self, skew):


        # Create an array that indicates which layer (10-3, 3-1, 0-1 km) the
        # wind belongs to. The array, agl, will be set to the height
        # corresponding to the top of the layer. The resulting array will look
        # something like this:
        #
        #   agl = [1.0 1.0 1.0 3.0 3.0 3.0 10.0 10.0 10.0 10.87 ]
        #
        # Where the values above 10 km are unchanged, and there are three levels
        # in each of the 3 layers of interest.
        #
        agl = np.copy(self.atmo_profiles.get('gh', {}).get('data')).to('km')

        heights = [10, 3, 1]
        agl_arr = agl.magnitude
        for i, height in enumerate(heights):

            mag_top = height
            mag_bottom = 0 if i >= len(heights) - 1 else heights[i+1]

            # Use exclude later to remove values above 10km
            if i == 0:
                exclude = -np.sum(agl_arr > mag_top)

            # Check for the values between two levels
            condition = np.logical_and(agl_arr <= mag_top, agl_arr > mag_bottom)
            agl.magnitude[condition] = mag_top

        # Note: agl is now an array with values corresponding to the heights
        # array

        # Retrieve the wind data profiles
        u_wind = self.atmo_profiles.get('u', {}).get('data')
        v_wind = self.atmo_profiles.get('v', {}).get('data')

        # Drop the points above 10 km
        u_wind = u_wind.magnitude[:exclude] * u_wind.units
        v_wind = v_wind.magnitude[:exclude] * v_wind.units

        # Create an inset axes object that is 28% width and height of the
        # figure and put it in the upper left hand corner.
        ax = inset_axes(skew.ax, '25%', '25%', loc=2)
        h = Hodograph(ax, component_range=80.)
        h.add_grid(increment=20, linewidth=0.5)

        # Plot the line colored by height AGL only up to the 10km level
        h.plot_colormapped(u_wind, v_wind, agl[:exclude], linewidth=2)

    @staticmethod
    def _plot_labels(skew):

        skew.ax.set_xlabel('Temperature (F)')
        skew.ax.set_ylabel('Pressure (hPa)')

    def _plot_profile(self, skew):

        profiles = self.atmo_profiles # dictionary
        pres = profiles.get('pres').get('data')
        temp = profiles.get('temp').get('data')
        sphum = profiles.get('sphum').get('data')

        dewpt = mpcalc.dewpoint_from_specific_humidity(sphum, temp, pres).to('degF')

        # Pressure vs temperature
        skew.plot(pres, temp, 'r', linewidth=1.5)

        # Pressure vs dew point temperature
        skew.plot(pres, dewpt, 'blue', linewidth=1.5)

        # Compute parcel profile and plot it
        parcel_profile = mpcalc.parcel_profile(pres,
                                               temp[0],
                                               dewpt[0]).to('degC')
        skew.plot(pres,
                  parcel_profile,
                  'orange',
                  linestyle='dashed',
                  linewidth=1.2,
                  )

    def _plot_wind_barbs(self, skew):

        # Pressure vs wind
        skew.plot_barbs(self.atmo_profiles.get('pres', {}).get('data'),
                        self.atmo_profiles.get('u', {}).get('data'),
                        self.atmo_profiles.get('v', {}).get('data'),
                        color='blue',
                        linewidth=0.2,
                        y_clip_radius=0,
                        )

    def _setup_diagram(self):

        # Create a new figure. The dimensions here give a good aspect ratio.
        fig = plt.figure(figsize=(12, 12))
        skew = SkewT(fig, rotation=45, aspect=85)

        # Set the range covered by the x and y axes.
        skew.ax.set_ylim(1050, self.max_plev)
        skew.ax.set_xlim(-35, 50)

        # The upper air grid is in Celcius, but we want ticks at the surface to
        # display in Fahrenheit.

        # Fahrenheit tick labels that will display
        labels_F = list(range(-20, 125, 20)) * units.degF

        # Celcius VALUES for those tick marks. These put the ticks in the right
        # spot.
        labels = labels_F.to('degC').magnitude

        # Set the MINOR tick values to the CELCIUS values.
        skew.ax.xaxis.set_minor_locator(FixedLocator(labels))

        # Set the MINOR tick labels to the FAHRENHEIT values.
        skew.ax.set_xticklabels(labels_F.magnitude, minor=True)
        skew.ax.tick_params(which='minor',
                            length=8)

        # Turn off the MAJOR (celcius) tick marks, label the grid lines inside
        # the axes.
        skew.ax.tick_params(axis='x',
                            labelbottom=True,
                            labelcolor='gray',
                            labelright=True,
                            labelrotation=45,
                            labeltop=True,
                            length=0,
                            pad=-25,
                            which='major',
                            )

        # Add the relevant special lines
        dry_adiabats = np.arange(-40, 210, 10) * units.degC
        skew.plot_dry_adiabats(dry_adiabats,
                               colors='tan',
                               linestyles='solid',
                               linewidth=0.7,
                               )

        moist_adiabats = np.arange(8, 36, 4) * units.degC
        moist_pr = np.arange(1001, 220, -10) * units.hPa
        skew.plot_moist_adiabats(moist_adiabats,
                                 moist_pr,
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

    @property
    @lru_cache()
    def thermo_variables(self):

        '''
        Return an ordered dictionary of thermodynamic variables needed for the skewT.
        Ordered because we want to print these values in this order on the SkewT
        diagram.
        The return dictionary contains a 'data' entry for each variable that
        includes the value of the metric.

        Variables' transforms and units are handled by default specs in much the
        same way as in fieldData class since these are not used by MetPy
        explictly.
        '''

        # OrderedDict so that we get the thermodynamic variables printed in the
        # same order every time in the resulting SkewT inset. The fields
        # include:
        #
        #    Variable short name:     can be consistent with default_specs.yml.
        #                             If not, must provide level and variable
        #                             entries
        #       level:                (optional) level to choose in
        #                             default_specs.yml. Default is 'ua'
        #       variable:             (optional) top-level variable to choose
        #                             from default_specs.yml.
        #       decimals:             (optional) number of decimal places to
        #                             include when formatting output. Defaults
        #                             to 0 (integer).
        thermo = OrderedDict({
            'cape': { # Convective available potential energy
                'level': 'sfc',
                },
            'cin': { # Convective inhibition
                'level': 'sfc',
                },
            'mucape': { # Most Unstable CAPE
                'level': 'mu',
                'variable': 'cape',
                },
            'mucin': { # CIN from MUCAPE level
                'level': 'mu',
                'variable': 'cin',
                },
            'li': { # Lifted Index
                'decimals': 1,
                'level': 'sfc',
                },
            'bli': { # Best Lifted Index
                'decimals': 1,
                'level': 'best',
                'variable': 'li',
                },
            'lcl': { # Lifted Condensation Level
                },
            'lpl': { # Lifted Parcel Level
                },
            'srh03': { # 0-3 km Storm relative helicity
                'level': 'sr03',
                'variable': 'hlcy',
                },
            'srh01': { # 0-1 km Storm relative helicity
                'level': 'sr01',
                'variable': 'hlcy',
                },
            'shr06': { # 0-6 km Shear
                'level': '06km',
                'variable': 'shear',
                },
            'shr01': { # 0-1 km Shear
                'level': '01km',
                'variable': 'shear',
                },
            'cell': { # Cell motion
                },
            'pwtr': { # Precipitable water
                'decimals': 1,
                'level': 'sfc',
                },
            })

        for var, items in thermo.items():

            varname = items.get('variable', var)
            lev = items.get('level', 'ua')
            spec = self.grib_data.spec.get(varname, {}).get(lev)

            if not spec:
                raise errors.NoGraphicsDefinitionForVariable(varname, lev)

            tmp = self.values(level=lev, name=varname)

            transforms = spec.get('transform')
            if transforms:
                transform_kwargs = spec.get('transform_kwargs', {})

                # Treat any transforms as a list
                transforms = transforms if isinstance(transforms, list) else [transforms]

                for transform in transforms:

                    if len(transform.split('.')) == 1:
                        tmp = self.__getattribute__(transform)(tmp, **transform_kwargs)
                    else:
                        tmp = utils.get_func(transform)(tmp, **transform_kwargs)

            thermo[var]['data'] = tmp
            thermo[var]['units'] = spec.get('unit')

        return thermo

    def _title(self):

        ''' Creates standard annotation for a skew-T. '''

        atime = self.grib_data.date_to_str(self.grib_data.anl_dt)
        vtime = self.grib_data.date_to_str(self.grib_data.valid_dt)

        # Top Left
        plt.title(f"Analysis: {atime}\nFcst Hr: {self.grib_data.fhr}",
                  fontsize=16,
                  loc='left',
                  position=(0, 1.03),
                  )

        # Top Right
        plt.title(f"Valid: {vtime}",
                  fontsize=16,
                  loc='right',
                  position=(1, 1.03),
                  )

        # Center
        site = f"{self.site_code} {self.site_num} {self.site_name}"
        site_loc = f"{self.site_lat},  {self.site_lon}"
        site_title = f"{site} at nearest grid pt over land {site_loc}"
        plt.title(site_title, loc='center', fontsize=12)
