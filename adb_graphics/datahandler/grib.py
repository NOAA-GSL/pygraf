'''
Classes that handle generic grib file handling, and those that handle the
specifics of grib files from UPP.
'''

import datetime
from functools import lru_cache

from matplotlib import cm
import numpy as np
import Nio

from .. import errors
from .. import specs
from .. import utils


class GribFile():

    ''' Wrappers and helper functions for interfacing with pyNIO.'''

    def __init__(self, filename):
        self.filename = filename
        self.contents = self._load()

    def _load(self):

        ''' Internal method that opens the grib file. Returns a grib message
        iterator. '''

        return Nio.open_file(self.filename) # pylint: disable=c-extension-no-member

    def get_field(self, ncl_name):

        ''' Given a numeric level and an ncl_name, return the entire grib field. '''

        try:
            field = self.contents.variables[ncl_name]
        except KeyError:
            raise errors.GribReadError(f'{ncl_name}')


class UPPData(GribFile, specs.VarSpec):

    '''
    Class handles grib file manipulation for a given variable in a UPP output
    file.

    Input:
        filename:    Path to grib file.
        level:       level corresponding to entry in specs configuration
        short_name:  name of variable corresponding to entry in specs configuration

    Key Word Arguments:
        config:      path to a user-specified configuration file
    '''

    def __init__(self, filename, level, short_name, **kwargs):

        # Parse kwargs first
        config = kwargs.get('config', 'adb_graphics/default_specs.yml')

        GribFile.__init__(self, filename)
        specs.VarSpec.__init__(self, config)

        self.level = level
        self.short_name = short_name
        self.spec = self.yml
        self.vspec = self.spec.get(short_name, {}).get(level)

        if not self.vspec:
            raise errors.NoGraphicsDefinitionForVariable(short_name, level)

    @property
    def anl_dt(self) -> datetime.datetime:

        ''' Returns the initial time of the grib file as a datetime object from
        the grib file.'''

        return datetime.datetime.strptime(self.data.initial_time, '%m/%d/%Y (%H:%M)') self.data.analDate

    @property
    def clevs(self) -> list:

        '''
        Uses the information contained in the yaml config file to determine
        the set of levels to be contoured. Returns the list of levels.

        The yaml file "clevs" key may contain a list, a range, or a call to a
        function. The logic to parse those options is included here.
        '''

        clev = self.vspec['clevs']

        # Is clevs a list?
        if isinstance(clev, (list, np.ndarray)):
            return clev

        # Is clev a call to another function?
        try:
            return utils.get_func(clev)()
        except ImportError:
            print(f'Check yaml file definition of CLEVS for {self.short_name}.',
                  f'Must be a list, range, or function call!')

    @property
    def cmap(self):

        ''' Returns the LinearSegmentedColormap specified by the config key
        "cmap" '''

        return cm.get_cmap(self.vspec['cmap'])

    @property
    def colors(self) -> np.ndarray:

        '''
        Returns a list of colors, specified by the config key "colors".

        The yaml file "colors" key may contain a list or a function to be
        called.
        '''

        color_spec = self.vspec.get('colors')

        if isinstance(color_spec, (list, np.ndarray)):
            return np.asarray(color_spec)
        try:
            ret = self.__getattribute__(color_spec)
            if callable(ret):
                return ret()
            return ret
        except AttributeError:
            return color_spec

    @property
    def corners(self):

        '''
        Returns lat and lon of lower left (ll) and upper right(ur) corners:
               ll_lat, ur_lat, ll_lon, ur_lon
        '''

        lat, lon = [self.field[var][::] for var in ['gridlat_0', 'gridlat_1']]
        return [lat[0, 0], lat[-1, -1], lon[0, 0], lon[-1, -1]]

    @property
    @lru_cache()
    def field(self):

        ''' Wrapper that calls get_field method for the current variable. '''

        return self.get_field(self.vspec.get(ncl_name))

    @staticmethod
    def date_to_str(date: datetime) -> str:

        ''' Returns a formatted string (for graphic title) from a datetime
        object'''

        return date.strftime('%Y%m%d %H UTC')

    @property
    def fhr(self) -> str:

        ''' Returns the forecast hour from the grib file. '''

        return self.field.forecast_time

    @property
    def lev_descriptor(self):

        ''' Returns the descriptor for the variable's level type. '''

        return self.data.level_type

    @property
    def ticks(self) -> int:

        ''' Returns the number of color bar tick marks from the yaml config
        settings. '''

        return self.vspec.get('ticks', 10)

    @property
    def units(self) -> str:

        ''' Returns the variable unit from the yaml config, if available. If not
        specified in the yaml file, returns the value set in the Grib file. '''

        return self.vspec.get('unit', self.field['units'])

    @property
    def valid_dt(self) -> datetime.datetime:

        ''' Returns a datetime object corresponding to the forecast hour's valid
        time as set in the Grib file. '''

        fh = datetime.timedelta(hours=self.forecastTime)
        return self.anl_dt + fh

    @property
    @lru_cache()
    def values(self) -> np.ndarray:

        ''' Returns the numpy array of values for the variable after applying any
        unit conversion to the original data. '''

        transform = self.vspec.get('transform')

        # Retrieve appropriate level
        # --------------------------

        if len(self.field.shape) == 2:
            fld = self.field[::]
        elif len(self.field.shape) == 3:
            levs = f.variables[tmp.dimensions[0]][::]
            lev = int(np.argwhere(levs == self.level))







        if transform and transform != 'None':
            return utils.get_func(transform)(self.field[::])
        return self.field[::]

    @property
    @lru_cache()
    def wind(self) -> [np.ndarray, np.ndarray]:

        ''' Returns the u, v wind components as a list (length 2) of arrays. '''

        comps = ['u', 'v']

        # Get u, v ncl_names
        u, v = [self.spec.get(var, {}).get(self.level, {}).get('ncl_name') in ['u', 'v']]

        if u is None or v is None:
            raise errors.NoGraphicsDefinitionForVariable(short_name, level)

        return [self.get_fields(self.level, self.lev_type, comp).values for comp in comps]
