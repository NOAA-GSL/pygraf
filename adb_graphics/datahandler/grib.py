# pylint: disable=invalid-name,too-few-public-methods

'''
Classes that handle generic grib file handling, and those that handle the
specifics of grib files from UPP.
'''

import abc
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

    def __init__(self, filename, filetype):
        self.filename = filename
        self.contents = self._load()

        self.filetype = filetype

    def _load(self):

        ''' Internal method that opens the grib file. Returns a grib message
        iterator. '''

        return Nio.open_file(self.filename, format="grib2") # pylint: disable=c-extension-no-member

    def get_field(self, ncl_name):

        ''' Given an ncl_name, return the NioVariable object. '''

        try:
            field = self.contents.variables[ncl_name]
        except KeyError:
            raise errors.GribReadError(f'{ncl_name}')

        return field


class UPPData(GribFile, specs.VarSpec):

    '''
    Class provides interface for accessing field  data from UPP in
    Grib2 format.

    Input:
        filename:    Path to grib file.
        short_name:  name of variable corresponding to entry in specs configuration

    Keyword Arguments:
        config:      path to a user-specified configuration file
    '''

    def __init__(self, filename, short_name, **kwargs):

        # Parse kwargs first
        config = kwargs.get('config', 'adb_graphics/default_specs.yml')
        filetype = kwargs.get('filetype', 'prs')

        GribFile.__init__(self, filename, filetype)
        specs.VarSpec.__init__(self, config)

        self.spec = self.yml
        self.short_name = short_name
        self.level = 'ua'

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
            print(f'Check yaml file definition of CLEVS for {self.short_name}. ',
                  'Must be a list, range, or function call!')

    @staticmethod
    def date_to_str(date: datetime) -> str:

        ''' Returns a formatted string (for graphic title) from a datetime
        object'''

        return date.strftime('%Y%m%d %H UTC')

    @property
    def fhr(self) -> str:

        ''' Returns the forecast hour from the grib file. '''

        return str(self.field.forecast_time[0])

    @property
    @lru_cache()
    def field(self):

        ''' Wrapper that calls get_field method for the current variable.
        Returns the NioVariable object '''

        return self.get_field(self.ncl_name(self.vspec))

    def field_diff(self, values, variable2, level2) -> np.ndarray:

        ''' Subtracts the values from variable2 from self.field. '''

        return values - self.values(name=variable2, level=level2)

    def get_transform(self, transforms: str, val: np.ndarray, transform_kwargs):

        ''' Applies a set of one or more transforms to an np.array of data values '''

        # Treat any transforms as a list
        transforms = transforms if isinstance(transforms, list) else [transforms]

        for transform in transforms:
            if len(transform.split('.')) == 1:
                val = self.__getattribute__(transform)(val, **transform_kwargs)
            else:
                val = utils.get_func(transform)(val, **transform_kwargs)
        return val

    def latlons(self):

        ''' Returns the set of latitudes and longitudes '''

        return [self.contents.variables[var][::] for var in \
                self.field.coordinates.split()]

    @property
    def lev_descriptor(self):

        ''' Returns the descriptor for the variable's level type. '''

        return self.field.level_type

    def ncl_name(self, spec: dict):

        ''' Get the ncl_name from the specified dict. '''

        name = spec.get('ncl_name')
        if isinstance(name, dict):
            name = name.get(self.filetype)
        return name

    def numeric_level(self, level=None):

        '''
        Split the numeric level and unit associated with the level key.

        A blank string is returned for lev_val for levels that do not contain a
        numeric, e.g., 'sfc' or 'ua'.
        '''

        level = level if level else self.level
        # Gather all the numbers and convert to integer
        lev_val = ''.join([c for c in level if (c in digits or c == '.')])
        if lev_val:
            lev_val = float(lev_val) if '.' in lev_val else int(lev_val)

        # Gather all the letters
        lev_unit = ''.join([c for c in level if c in ascii_letters])

        return lev_val, lev_unit

    @property
    def valid_dt(self) -> datetime.datetime:

        ''' Returns a datetime object corresponding to the forecast hour's valid
        time as set in the Grib file. '''

        fh = datetime.timedelta(hours=int(self.fhr))
        return self.anl_dt + fh

    @abc.abstractmethod
    def values(self, level=None, name=None, **kwargs):

        ''' Returns the values of a given variable. '''
        ...

    @property
    def vspec(self):

        ''' Return the graphics specification for a given level. '''

        vspec = self.spec.get(self.short_name, {}).get(self.level)
        if not vspec:
            raise errors.NoGraphicsDefinitionForVariable(self.short_name, self.level)
        return vspec


class fieldData(UPPData):

    '''
    Class provides interface for accessing field (2D plan view) data from UPP in
    Grib2 format.

    Input:
        filename:    Path to grib file.
        level:       level corresponding to entry in specs configuration
        name      :  name of variable corresponding to entry in specs configuration

    Keyword Arguments:
        config:      path to a user-specified configuration file
    '''

    def __init__(self, filename, level, short_name, **kwargs):

        super().__init__(filename, short_name, **kwargs)

        self.level = level

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
    def corners(self) -> list:

        '''
        Returns lat and lon of lower left (ll) and upper right(ur) corners:
               ll_lat, ur_lat, ll_lon, ur_lon
        '''

        lat, lon = self.latlons()
        return [lat[0, 0], lat[-1, -1], lon[0, 0], lon[-1, -1]]

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


    @lru_cache()
    def values(self, level=None, name=None, **kwargs) -> np.ndarray:

        '''
        Returns the numpy array of values at the requested level for the
        variable after applying any unit conversion to the original data.

        Optional Input:
            name       the name of a field other than defined in self
            level      the level of the alternate field to use
        '''


        level = level if level else self.level

        if name is None:
            field = self.field
            spec = self.vspec
        else:
            spec = self.spec.get(name, {}).get(level)
            if not spec:
                raise errors.NoGraphicsDefinitionForVariable(name, level)
            field = self.get_field(self.ncl_name(spec))

        if len(field.shape) == 2:
            vals = field[::]
        elif len(field.shape) == 3:

            # Available variable levels
            dim_name = spec.get('vertical_level_name',
                                field.dimensions[0])
            print('grib.py: dim_name = ', dim_name)
            levs = self.contents.variables[dim_name][::]
            print('grib.py: levs = ', levs)

            # Requested level
            lev_val, lev_unit = self.numeric_level()
            lev_val = lev_val / 100. if lev_unit == 'cm' else lev_val
            lev_val = lev_val * 100. if lev_unit in ['mb', 'mxmb'] else lev_val
            lev_val = lev_val * 1000. if lev_unit in ['km', 'mx', 'sr'] else lev_val
            print('grib.py: lev_val, lev_unit = ', lev_val, lev_unit)

            # The index of the requested level
            lev = spec.get('vertical_index')
            print('grib.py: lev = ', lev)
            if lev is None:
                lev = int(np.argwhere(levs == lev_val))
            print('grib.py: lev = ', lev)
            vals = field[lev, :, :]

            transforms = spec.get('transform')
            if transforms:
                transform_kwargs = spec.get('transform_kwargs', {})
                vals = self.get_transform(transforms, vals, transform_kwargs)

        return vals

    def vector_magnitude(self, field1, field2, vertical_index=0):

        '''
        Returns the vector magnitude of two component vector fields. The
        input fields can be either NCL names (string) or full data fields. The
        first layer of a variable is returned if none is provided.
        '''

        if isinstance(field1, str):
            field1 = self.get_field(field1)[vertical_index]

        if isinstance(field2, str):
            field2 = self.get_field(field2)[vertical_index]

        return conversions.magnitude(field1, field2)

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

        # Create fieldData objects for u, v components
        field_lambda = lambda fn, level, var: fieldData(filename=fn,
                                                        level=level,
                                                        short_name=var,
                                                        )
        u, v = [field_lambda(self.filename, level, var) for var in ['u', 'v']]

        return [component.values() for component in [u, v]]

    def windspeed(self) -> np.ndarray:

        ''' Compute the wind speed from the components. '''

        u, v = self.wind(level=True)
        return conversions.magnitude(u, v)


class profileData(UPPData):

    '''
    Class provides methods for getting profiles from a specific lat/lon location
    from a grib file.

    Input:

      filename     full path to grib file
      loc          single entry from sites file. Use the first 31 spaces to get
                   site_code, site_num, lat, lon. Past 31 spaces is the site's
                   long name.
      short_name

    Key word arguments:

      Only used for base classes.

    '''

    def __init__(self, filename, loc, short_name, **kwargs):

        super().__init__(filename, short_name, **kwargs)

        # The first 31 columns are space delimted
        self.site_code, _, self.site_num, lat, lon = \
                loc[:31].split()

        # The variable lenght site name is included past column 37
        self.site_name = loc[37:].rstrip()

        # Convert the string to a number. Longitude should be negative for all
        # these sites.
        self.site_lat = float(lat)
        self.site_lon = -float(lon)

    @lru_cache()
    def get_xypoint(self) -> tuple:

        '''
        Return the X, Y grid point corresponding to the site location. No
        interpolation is used.
        '''

        lats, lons = self.latlons()
        max_x, max_y = np.shape(lats)

        # Numpy magic to grab the X, Y grid point nearest the profile site
        # pylint: disable=unbalanced-tuple-unpacking
        x, y = np.unravel_index((np.abs(lats - self.site_lat) \
               + np.abs(lons - self.site_lon)).argmin(), lats.shape)
        # pylint: enable=unbalanced-tuple-unpacking

        if x == 0 or y == 0 or x == max_x or y == max_y:
            msg = f"{self.site_name} is outside your domain!"
            raise errors.OutsideDomain(msg)

        return (x, y)

    def values(self, level=None, name=None, **kwargs):

        '''
        Returns the numpy array of values at the object's x, y location for the
        requested variable. Transforms are performed in the child class.

        Optional Input:
            name       the short name of a field other than defined in self
            level      the level of the alternate field to use, default='ua' for
                       upper air

        Keyword Args:
            layer      a numeric layer corresponding to an index of the field to
                       be returned
            ncl_name   the NCL name of the variable to be retrieved
        '''

        layer = kwargs.get('layer')
        ncl_name = kwargs.get('ncl_name')

        # Set the default here since this is an instance of an abstract method
        level = level if level else 'ua'

        if not name:
            name = self.short_name

        x, y = self.get_xypoint()

        # Retrieve the default_specs section for the specified level
        var_spec = self.spec.get(name, {}).get(level, {})
        ncl_name = ncl_name if ncl_name else self.ncl_name(var_spec)

        if not ncl_name:
            raise errors.NoGraphicsDefinitionForVariable(
                name,
                'ua',
                )

        # Specifies the vertical index if one is needed.
        layer = layer if layer else var_spec.get('layer')

        profile = self.contents.variables[ncl_name][::]
        if len(profile.shape) == 2:
            profile = profile[x, y]
        elif len(profile.shape) == 3:
            if layer is not None:
                profile = profile[layer, x, y]
            else:
                profile = profile[:, x, y]
        return profile

    def vector_magnitude(self, field1, field2, layer=None, level='ua'):

        '''
        Returns the vector magnitude of two component vector profiles. The
        input fields can be either NCL names (string) or full data fields.

        If no layer or level is provided, the default 'ua' will be used in
        self.values.
        '''

        if isinstance(field1, str):
            field1 = self.values(ncl_name=field1, layer=layer, level=level)

        if isinstance(field2, str):
            field2 = self.values(ncl_name=field2, layer=layer, level=level)

        return conversions.magnitude(field1, field2)
