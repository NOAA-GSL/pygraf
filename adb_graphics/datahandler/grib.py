# pylint: disable=invalid-name,too-few-public-methods

'''
Classes that handle generic grib file handling, and those that handle the
specifics of grib files from UPP.
'''

import datetime
from functools import lru_cache
from string import digits, ascii_letters

from matplotlib import cm
import numpy as np
import Nio

from .. import conversions
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

        ''' Given an ncl_name, return the NioVariable object. '''

        try:
            field = self.contents.variables[ncl_name]
        except KeyError:
            raise errors.GribReadError(f'{ncl_name}')

        return field


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

    @property
    def anl_dt(self) -> datetime.datetime:

        ''' Returns the initial time of the grib file as a datetime object from
        the grib file.'''

        return datetime.datetime.strptime(self.field.initial_time, '%m/%d/%Y (%H:%M)')

    @property
    def clevs(self) -> np.ndarray:

        '''
        Uses the information contained in the yaml config file to determine
        the set of levels to be contoured. Returns the list of levels.

        The yaml file "clevs" key may contain a list, a range, or a call to a
        function. The logic to parse those options is included here.
        '''

        clev = np.asarray(self.vspec['clevs'])

        # Is clevs a list?
        if isinstance(clev, (list, np.ndarray)):
            return np.asarray(clev)

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

        lat, lon = self.latlons()
        return [lat[0, 0], lat[-1, -1], lon[0, 0], lon[-1, -1]]

    @staticmethod
    def date_to_str(date: datetime) -> str:

        ''' Returns a formatted string (for graphic title) from a datetime
        object'''

        return date.strftime('%Y%m%d %H UTC')

    @property
    def fhr(self) -> str:

        ''' Returns the forecast hour from the grib file. '''

        return str(self.field.forecast_time[0])

    def field_diff(self, values, variable2, level2):

        ''' Subtracts the values from variable2 from self.field. '''

        return values - self.values(variable2, level2)

    @property
    @lru_cache()
    def field(self):

        ''' Wrapper that calls get_field method for the current variable.
        Returns the NioVariable object '''

        return self.get_field(self.vspec.get('ncl_name'))

    def latlons(self):

        ''' Returns the set of latitudes and longitudes '''

        return [self.contents.variables[var][::] for var in \
                self.field.coordinates.split()]

    @property
    def lev_descriptor(self):

        ''' Returns the descriptor for the variable's level type. '''

        return self.field.level_type

    @property
    def numeric_level(self):

        '''
        Split the numeric level and unit associated with the level key.

        A blank string is returned for lev_val for levels that do not contain a
        numeric, e.g., 'sfc' or 'ua'.
        '''

        lev_val = ''.join([c for c in self.level if c in digits])
        lev_val = int(lev_val) if lev_val else lev_val
        lev_unit = ''.join([c for c in self.level if c in ascii_letters])

        return lev_val, lev_unit

    @property
    def ticks(self) -> int:

        ''' Returns the number of color bar tick marks from the yaml config
        settings. '''

        return self.vspec.get('ticks', 10)

    @property
    def units(self) -> str:

        ''' Returns the variable unit from the yaml config, if available. If not
        specified in the yaml file, returns the value set in the Grib file. '''

        return self.vspec.get('unit', self.field.units)

    @property
    def valid_dt(self) -> datetime.datetime:

        ''' Returns a datetime object corresponding to the forecast hour's valid
        time as set in the Grib file. '''

        fh = datetime.timedelta(hours=int(self.fhr))
        return self.anl_dt + fh

    @lru_cache()
    def values(self, name=None, level=None) -> np.ndarray:

        '''
        Returns the numpy array of values at the requested level for the
        variable after applying any unit conversion to the original data.

        Optional Input:
            name       the name of a field other than defined in self
            level      the level of the alternate field to use
        '''

        if name is None:
            field = self.field
            spec = self.vspec
        else:
            spec = self.spec.get(name, {}).get(level)
            if not spec:
                raise errors.NoGraphicsDefinitionForVariable(name, level)
            field = self.get_field(spec.get('ncl_name'))

        transforms = spec.get('transform')

        if len(field.shape) == 2:
            vals = field[::]
        elif len(field.shape) == 3:

            # Available variable levels
            levs = self.contents.variables[field.dimensions[0]][::]

            # Requested level
            lev_val, lev_unit = self.numeric_level
            lev_val = lev_val * 100. if lev_unit == 'mb' else lev_val

            # The index of the reqested level
            lev = int(np.argwhere(levs == lev_val))
            vals = field[lev, :, :]

        if transforms:
            transform_kwargs = spec.get('transform_kwargs', {})

            # Treat any transforms as a list
            transforms = transforms if isinstance(transforms, list) else [transforms]

            for transform in transforms:

                print(f'Transform: {transform}')

                if len(transform.split('.')) == 1:
                    print(transform_kwargs)
                    vals = self.__getattribute__(transform)(vals, **transform_kwargs)
                else:
                    vals = utils.get_func(transform)(vals, **transform_kwargs)

        return vals

    @property
    def vspec(self):

        ''' Return the graphics specification for a given level. '''

        vspec = self.spec.get(self.short_name, {}).get(self.level)
        if not vspec:
            raise errors.NoGraphicsDefinitionForVariable(self.short_name, self.level)
        return vspec

    @lru_cache()
    def wind(self, level) -> [np.ndarray, np.ndarray]:

        '''
        Returns the u, v wind components as a list (length 2) of arrays.

            Input:
                level      bool or level key. If True, use same level as self,
                           if a string level key is provided, use wind at that
                           level.
        '''

        level = self.level if level and isinstance(level, bool) else level

        # Just in case wind gets called with level=False
        if not level:
            return False

        # Create UPPData objects for u, v components
        u, v = [UPPData(filename=self.filename, level=level, short_name=var) for var in ['u', 'v']]

        return [component.values() for component in [u, v]]

    def windspeed(self):

        ''' Compute the wind speed from the components. '''

        u, v = self.wind()
        return conversions.magnitude(u, v)
