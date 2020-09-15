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

    def __init__(self, filename, filetype, **kwargs):
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

    @property
    def grid_suffix(self):
        ''' Return the suffix of the first variable in the file. This should
        correspond to the grid tag. '''

        var = list(self.contents.variables.keys())[0]
        return var.split('_')[-1]

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

        GribFile.__init__(self, filename, filetype, **kwargs)
        specs.VarSpec.__init__(self, config)

        self.spec = self.yml
        self.short_name = short_name
        self.level = 'ua'

        self.fhr = str(kwargs['fhr'])

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

        clev = np.asarray(self.vspec.get('clevs', []))

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
    @lru_cache()
    def field(self):

        ''' Wrapper that calls get_field method for the current variable.
        Returns the NioVariable object '''

        return self.get_field(self.ncl_name(self.vspec))

    def field_diff(self, values, variable2, level2, **kwargs) -> np.ndarray:

        # pylint: disable=unused-argument

        ''' Subtracts the values from variable2 from self.field. '''

        return values - self.values(name=variable2, level=level2)

    def get_level(self, field, level, spec):

        ''' Returns the value of the level to for a 3D array '''

        # The index of the requested level
        lev = spec.get('vertical_index')
        if lev is not None:
            return lev

        # Follow convention of fieldData objects for getting vertical level
        dim_name = spec.get('vertical_level_name',
                            field.dimensions[0])
        levs = self.contents.variables[dim_name][::]

        # Requested level
        lev_val, _ = self.numeric_level(level=level)

        return int(np.argwhere(levs == lev_val))

    def get_transform(self, transforms, val):

        ''' Applies a set of one or more transforms to an np.array of
        data values.

        Input:

          transforms:    the transform section of a variable spec
          val:           a value, list, or array of values to be
                         transformed

        Return:

          val:           updated values after transforms have been
                         applied
        '''

        transform_kwargs = {}
        if isinstance(transforms, dict):
            transform_list = transforms.get('funcs')
            if not isinstance(transform_list, list):
                transform_list = [transform_list]
            transform_kwargs = transforms.get('kwargs')
        elif isinstance(transforms, str):
            transform_list = [transforms]
        else:
            transform_list = transforms

        for transform in transform_list:
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
        return name.format(fhr=self.fhr, grid=self.grid_suffix)

    def numeric_level(self, index_match=True, level=None):

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

        if index_match:
            lev_val = lev_val / 100. if lev_unit == 'cm' else lev_val
            lev_val = lev_val * 100. if lev_unit in ['mb', 'mxmb'] else lev_val
            lev_val = lev_val * 1000. if lev_unit in ['km', 'mx', 'sr'] else lev_val

        return lev_val, lev_unit

    @staticmethod
    def opposite(values, **kwargs) -> np.ndarray:
    # pylint: disable=unused-argument

        ''' Returns the opposite of input values  '''

        return - values

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
        self.contour_kwargs = kwargs.get('contour_kwargs', {})

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
    def grid_info(self):

        ''' Returns a dict that includes the grid info for the full grid. '''

        # Keys are grib names, values are Basemap argument names
        ncl_to_basemap = dict(
                CenterLon='lon_0',
                CenterLat='lat_0',
                Latin2='lat_1',
                Latin1='lat_2',
                Lov='lon_0',
                La1='lat_0',
                La2='lat_2',
                Lo1='lon_1',
                Lo2='lon_2',
                )

        # Last coordinate listed should be latitude or longitude
        lat_var, lon_var = self.field.coordinates.split()

        # Get the latitude variable
        lat = self.contents.variables[lat_var]
        lon = self.contents.variables[lon_var]

        grid_info = {}

        grid_info['corners'] = self.corners

        if self.grid_suffix in ['GLC0']:
            attrs =  ['Latin1', 'Latin2', 'Lov']
            grid_info['projection'] = 'lcc'
            grid_info['lat_0'] = 39.0
        elif self.grid_suffix == 'GST0':
            attrs = ['Lov']
            grid_info['projection'] = 'stere'
            grid_info['lat_0'] = 90
            grid_info['lat_ts'] = 90

        else:
            attrs = []
            grid_info['projection'] = 'rotpole'
            grid_info['lon_0'] = lat.attributes['CenterLon'][0] - 360
            grid_info['o_lat_p'] = 90 - lat.attributes['CenterLat'][0]
            grid_info['o_lon_p'] = 180

        for attr in attrs:
            bm_arg = ncl_to_basemap[attr]
            val = lat.attributes[attr]
            val = val[0] if isinstance(val, np.ndarray) else val
            grid_info[bm_arg] = val

        print('GRID_INFO')
        for k, v in grid_info.items():
            print(f'{k}: {v}')

        return grid_info

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

            lev = self.get_level(field, level, spec)
            vals = field[lev, :, :]

        transforms = spec.get('transform')
        if transforms:
            vals = self.get_transform(transforms, vals)

        return vals

    def vector_magnitude(self, field1, field2, vertical_index=0, **kwargs):

        # pylint: disable=unused-argument

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
        field_lambda = lambda fn, level, var: fieldData(
            fhr=self.fhr,
            filename=fn,
            level=level,
            short_name=var,
            )
        u, v = [field_lambda(self.filename, level, var) for var in ['u', 'v']]

        return [component.values() for component in [u, v]]

    @property
    def wind_stride(self):

        ''' Returns the wind stride associated with the field '''



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
            ncl_name         the NCL name of the variable to be retrieved
            one_lev          bool flag. if True, get the single level of the variable
            vertical_index   the index of the required level
        '''

        # Set the defaults here since this is an instance of an abstract method
        # level refers to the level key in the specs file.
        level = level if level else 'ua'

        if not name:
            name = self.short_name

        one_lev = kwargs.get('one_lev', False)

        # Retrive the location for the profile
        x, y = self.get_xypoint()

        # Retrieve the default_specs section for the specified level
        var_spec = self.spec.get(name, {}).get(level, {})

        # Set the NCL name from the specs section, unless otherwise specified
        ncl_name = kwargs.get('ncl_name') or self.ncl_name(var_spec)
        ncl_name = ncl_name.format(fhr=self.fhr, grid=self.grid_suffix)

        if not ncl_name:
            raise errors.NoGraphicsDefinitionForVariable(
                name,
                'ua',
                )

        # Get the full 2- or 3-D field
        field = self.contents.variables[ncl_name]

        profile = field[::]
        lev = 0
        if len(profile.shape) == 2:
            profile = profile[x, y]
        elif len(profile.shape) == 3:
            if one_lev:
                lev = self.get_level(field, level, var_spec)
                profile = profile[lev, x, y]
            else:
                profile = profile[:, x, y]
        return profile

    def vector_magnitude(self, field1, field2, level='ua', vertical_index=None, **kwargs):

        '''
        Returns the vector magnitude of two component vector profiles. The
        input fields can be either NCL names (string) or full data fields.

        If no layer or level is provided, the default 'ua' will be used in
        self.values.
        '''


        if isinstance(field1, str):
            field1 = self.values(
                level=level,
                ncl_name=field1,
                vertical_index=vertical_index,
                **kwargs,
                )

        if isinstance(field2, str):
            field2 = self.values(
                level=level,
                ncl_name=field2,
                vertical_index=vertical_index,
                **kwargs,
                )

        return conversions.magnitude(field1, field2)
