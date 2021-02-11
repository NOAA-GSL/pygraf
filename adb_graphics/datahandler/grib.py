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
import xarray as xr

from .. import conversions
from .. import errors
from .. import specs
from .. import utils

class GribFile():

    ''' Wrappers and helper functions for interfacing with pyNIO.'''

    def __init__(self, filename, **kwargs):

        # pylint: disable=unused-argument

        self.filename = filename
        self.contents = self._load()

    def _load(self):

        ''' Internal method that opens the grib file. Returns a grib message
        iterator. '''

        return xr.open_dataset(self.filename,
                               engine='pynio',
                               lock=False,
                               backend_kwargs=dict(format="grib2"),
                               )

class GribFiles():

    ''' Class for loading in a set of grib files and combining them over
    forecast hours. '''

    def __init__(self, filenames, filetype):

        '''
        Arguments:

          filenames   dict containing list of files names for the 0h and 1h
                      forecast lead times ('01fcst'), and all the free forecast
                      hours after that ('free_fcst').
          filetype    key to use for dict when setting variable_names

        '''
        self.filenames = filenames
        self.filetype = filetype
        self.contents = self._load()


    def _load(self):

        ''' Load the set of files into a single XArray structure. '''

        all_leads = []

        # 0h and 1h Forecast
        all_leads.append(xr.open_mfdataset(
            self.filenames['01fcst'],
            backend_kwargs=dict(format="grib2"),
            combine='nested',
            compat='override',
            concat_dim='fcst_hour',
            coords='minimal',
            data_vars=self.variable_names,
            engine='pynio',
            lock=False,
            ).rename_vars(self.variable_names))

        #
        all_leads.append(xr.open_mfdataset(
            self.filenames['free_fcst'],
            backend_kwargs=dict(format="grib2"),
            combine='nested',
            compat='override',
            coords='minimal',
            concat_dim='fcst_hour',
            data_vars=self.variable_names.values(),
            engine='pynio',
            lock=False,
            ))

        return xr.combine_nested(all_leads,
                                 compat='override',
                                 concat_dim='fcst_hour',
                                 coords='minimal',
                                 data_vars='minimal',
                                )

    @property
    def variable_names(self):

        '''
        Defines the variable name transitions that need to happen for each
        model to combine along forecast hours. 

        Keys are original variable names, values are the updated variable names.

        Choosing to update the 0 and 1 hour forecast variables to the longer
        lead times for efficiency's sake.
        '''

        names = {
            'hrrrx': {
                'REFC_P0_L10_GLC0': 'REFC_P0_L10_GLC0',
                'MXUPHL_P8_2L103_GLC0_max': 'MXUPHL_P8_2L103_GLC0_max1h',
                'UGRD_P0_L103_GLC0': 'UGRD_P0_L103_GLC0', 
                'VGRD_P0_L103_GLC0': 'VGRD_P0_L103_GLC0',
                'WEASD_P8_L1_GLC0_acc': 'WEASD_P8_L1_GLC0_acc1h',
                'APCP_P8_L1_GLC0_acc': 'APCP_P8_L1_GLC0_acc1h',
                'PRES_P0_L1_GLC0': 'PRES_P0_L1_GLC0',
                'VAR_0_7_200_P8_2L103_GLC0_min': 'VAR_0_7_200_P8_2L103_GLC0_min1h',
                }
            }

        return names[self.filetype]


class UPPData(specs.VarSpec):

    '''
    Class provides interface for accessing field  data from UPP in
    Grib2 format.

    Input:
        ds:          xarray dataset from grib file
        short_name:  name of variable corresponding to entry in specs configuration

    Keyword Arguments:
        config:      path to a user-specified configuration file
    '''

    def __init__(self, ds, short_name, **kwargs):


        # Parse kwargs first
        config = kwargs.get('config', 'adb_graphics/default_specs.yml')
        self.model = kwargs.get('model')
        self.filetype = kwargs.get('filetype', 'prs')


        specs.VarSpec.__init__(self, config)

        self.spec = self.yml
        self.short_name = short_name
        self.level = 'ua'

        self.fhr = str(kwargs['fhr'])

        self.ds = ds

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

    def field_mean(self, values, variable, levels, **kwargs) -> np.ndarray:

        # pylint: disable=unused-argument

        ''' Returns the mean of the values. '''

        fsum = np.zeros_like(values)
        for level in levels:
            fsum = fsum + self.values(name=variable, level=level)

        return fsum / len(levels)

    def get_field(self, ncl_name):

        ''' Given an ncl_name, return the NioVariable object. '''

        try:
            field = self.ds[ncl_name.format(level_type=self.level_type)]
        except KeyError:
            raise errors.GribReadError(f'{ncl_name}')
        return field

    def get_level(self, field, level, spec):

        ''' Returns the value of the level to for a 3D array '''

        # The index of the requested level
        lev = spec.get('vertical_index')
        if lev is not None:
            return lev

        # Follow convention of fieldData objects for getting vertical level
        dim_name = spec.get('vertical_level_name',
                            list(field.dims)[0])
        levs = self.ds[dim_name].values

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

    @property
    def grid_suffix(self):
        ''' Return the suffix of the first variable with 4 sections (split on _)
        in the file. This should correspond to the grid tag. '''

        for var in self.ds.keys():
            vsplit = var.split('_')
            if len(vsplit) == 4:
                return vsplit[-1]


    def latlons(self):

        ''' Returns the set of latitudes and longitudes '''

        coords = sorted([c for c in list(self.ds.coords) if any(ele in c for ele in ['lat',
            'lon'])])
        return [self.ds.coords[c].values for c in coords]

    @property
    def lev_descriptor(self):

        ''' Returns the descriptor for the variable's level type. '''

        return self.field.level_type

    @property
    def level_type(self):

        ''' Returns a Grib2 code for type of level. 10 is used for
        entire atmosphere in HRRR, while 200 is used in RRFS. '''

        if self.filetype == 'prs':
            if self.model == 'rrfs':
                return 200
            return 10
        return 105

    def ncl_name(self, spec: dict):

        ''' Get the ncl_name from the specified spec dict. '''

        name = spec.get('ncl_name')

        if isinstance(name, dict):
            if self.model in name.keys():
                name = name.get(self.model)
            else:
                name = name.get(self.filetype)
        # The level_type for the entire atmosphere could be L10 or L200. Thanks
        # Grib2! Handle that in "try" statement when reading file.
        return name.format(fhr=self.fhr,
                           grid=self.grid_suffix,
                           level_type=self.level_type)

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
        ds:          xarray dataset from grib file
        level:       level corresponding to entry in specs configuration
        name:        name of variable corresponding to entry in specs configuration

    Keyword Arguments:
        config:      path to a user-specified configuration file
    '''

    def __init__(self, ds, level, short_name, **kwargs):

        super().__init__(ds, short_name, **kwargs)

        self.level = level
        self.contour_kwargs = kwargs.get('contour_kwargs', {})

    def aviation_flight_rules(self, values, **kwargs) -> np.ndarray:
        # pylint: disable=unused-argument

        '''
        Generates a field of Aviation Flight Rules from Ceil and Vis
        '''

        ceil = values
        vis = self.values(name='vis', level='sfc')

        flru = np.where((ceil > 1.) & (ceil < 3.), 1.01, 0.0)
        flru = np.where((vis > 3.) & (vis < 5.), 1.01, flru)
        flru = np.where((ceil > 0.5) & (ceil < 1.), 2.01, flru)
        flru = np.where((vis > 1.) & (vis < 3.), 2.01, flru)
        flru = np.where((ceil > 0.0) & (ceil < 0.5), 3.01, flru)
        flru = np.where((vis < 1.), 3.01, flru)

        return flru

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
        lat_var, _ = list(self.field.coords)[-2:]

        # Get the latitude variable
        lat = self.ds[lat_var]

        grid_info = {}
        grid_info['corners'] = self.corners
        if self.grid_suffix in ['GLC0']:
            attrs = ['Latin1', 'Latin2', 'Lov']
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
            grid_info['lon_0'] = lat.attrs['CenterLon'][0] - 360
            grid_info['o_lat_p'] = 90 - lat.attrs['CenterLat'][0]
            grid_info['o_lon_p'] = 180

        for attr in attrs:
            bm_arg = ncl_to_basemap[attr]
            val = lat.attrs[attr]
            val = val[0] if isinstance(val, np.ndarray) else val
            grid_info[bm_arg] = val

        return grid_info

    def supercooled_liquid_water(self, values, **kwargs) -> np.ndarray:

        # pylint: disable=unused-argument

        '''
        Generates a field of Supercooled Liquid Water

        This method uses wrfnat data to find regions where
        cloud and rain moisture are in below-freezing temps.

        Because pressures represent mid-layer values, the calculation
        works from the surface and (1) computes the depth of a pressure layer,
        and (2) computes supercooled liquid water for the layer and sums the
        columns, and (3) uses the layer depth to find the pressure at the
        next sigma level.

        The process is iterative to the topof the atmosphere.
        '''

        pres_sfc = self.values(name='pres', level='sfc') * 100. # convert back to Pa
        pres_nat_lev = self.values(name='pres', level='ua', one_lev=False)
        temp = self.values(name='temp', level='ua', one_lev=False)
        cloud_mixing_ratio = self.values(name='clwmr', level='ua', one_lev=False)
        rain_mixing_ratio = self.values(name='rwmr', level='ua', one_lev=False)

        gravity = 9.81
        slw = pres_sfc * 0. # start with array of zero values

        nlevs = np.shape(pres_nat_lev)[0] # determine number of vertical levels
        for n in range(nlevs):
            if n == 0:
                pres_layer = 2 * (pres_sfc[:, :] - pres_nat_lev[n, :, :])  # layer depth
                pres_sigma = pres_sfc - pres_layer        # pressure at next sigma level
            else:
                pres_layer = 2 * (pres_sigma[:, :] - pres_nat_lev[n, :, :]) # layer depth
                pres_sigma = pres_sigma - pres_layer       # pressure at next sigma level
            # compute supercooled water in layer and add to previous values
            supercool_locs = np.where((temp[n, :, :] < 0.0), \
                             cloud_mixing_ratio[n, :, :]+rain_mixing_ratio[n, :, :], 0.0)
            slw = slw + pres_layer / gravity * supercool_locs

        return slw

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

    def values(self, level=None, name=None, **kwargs) -> np.ndarray:

        '''
        Returns the numpy array of values at the requested level for the
        variable after applying any unit conversion to the original data.

        Optional Input:
            name       the name of a field other than defined in self
            level      the desired level of the named field

        Keyword Args:
            ncl_name        the NCL-assigned Grib2 name
            one_lev    bool flag. if True, get the single level of the variable
            vertical_index  the index (int) of the desired vertical level
        '''

        level = level if level else self.level

        one_lev = kwargs.get('one_lev', True)
        vertical_index = kwargs.get('vertical_index')

        ncl_name = kwargs.get('ncl_name', '')
        ncl_name = ncl_name.format(fhr=self.fhr, grid=self.grid_suffix)


        if name is None and not ncl_name:

            # Use field and spec from the current object
            field = self.field
            spec = self.vspec

        else:

            # Get the spec dict and ncl_name for the given variable name
            spec = self.spec.get(name, {}).get(level, {})
            if not spec and name is not None:
                raise errors.NoGraphicsDefinitionForVariable(name, level)
            field = self.get_field(ncl_name or self.ncl_name(spec))

        if len(field.shape) == 2:
            vals = field[::]

        elif len(field.shape) == 3:
            if one_lev:
                lev = vertical_index
                if vertical_index is None:
                    lev = self.get_level(field, level, spec)
                vals = field[lev, :, :]
            else:
                vals = field[:, :, :]

        transforms = spec.get('transform')
        if transforms:
            vals = self.get_transform(transforms, vals)

        return vals

    def vector_magnitude(self, field1, field2, level=None, vertical_index=None, **kwargs):

        # pylint: disable=unused-argument

        '''
        Returns the vector magnitude of two component vector fields. The
        input fields can be either NCL names (string) or full data fields. The
        first layer of a variable is returned if none is provided.
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
        field_lambda = lambda ds, level, var: fieldData(
            ds=ds,
            fhr=self.fhr,
            level=level,
            short_name=var,
            )
        u, v = [field_lambda(self.ds, level, var) for var in ['u', 'v']]

        return [component.values() for component in [u, v]]


class profileData(UPPData):

    '''
    Class provides methods for getting profiles from a specific lat/lon location
    from a grib file.

    Input:

      ds           xarray dataset from grib file
      loc          single entry from sites file. Use the first 31 spaces to get
                   site_code, site_num, lat, lon. Past 31 spaces is the site's
                   long name.
      short_name

    Key word arguments:

      Only used for base classes.

    '''

    def __init__(self, ds, loc, short_name, **kwargs):

        super().__init__(ds, short_name, **kwargs)

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
        vertical_index = kwargs.get('vertical_index')

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
        field = self.ds[ncl_name]

        profile = field[::]
        lev = 0
        if len(profile.shape) == 2:
            profile = profile[x, y]
        elif len(profile.shape) == 3:
            if one_lev:
                lev = vertical_index
                if vertical_index is None:
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

class timeLaggedData(fieldData):

    '''
    Class provides interface for accessing time lagged fields (2D plan view)
    data from UPP in Grib2 format.

    Input:
        ds:          xarray dataset from full set of grib files
        fcst_hours   list of forecast hours to include
        level:       level corresponding to entry in specs configuration, likely
                     esbl or total
        name:        name of variable corresponding to entry in specs configuration

    Keyword Arguments:
        config:      path to a user-specified configuration file
    '''

    def __init__(self, ds, fcst_hours, level, short_name, **kwargs):

        super().__init__(ds, level, short_name, **kwargs)

        self.level = level
        self.contour_kwargs = kwargs.get('contour_kwargs', {})

        self.fcst_hours = fcst_hours

    @property
    def valid_dt(self) -> datetime.datetime:

        ''' Returns a datetime object corresponding to the forecast hour's valid
        time as set in the Grib file. '''

        deltas = [datetime.timedelta(hours=int(hr)) for hr in self.fcst_hours]
        return [self.anl_dt + delta for delta in deltas]

    def values(self, level=None, name=None, **kwargs) -> np.ndarray:

        '''
        Returns the numpy array of values at the requested level for the
        variable after applying any unit conversion to the original data.

        Optional Input:
            name       the name of a field other than defined in self
            level      the level of the alternate field to use
        '''

        level = level if level else self.level

        vertical_index = kwargs.get('vertical_index')

        ncl_name = kwargs.get('ncl_name', '')
        ncl_name = ncl_name.format(grid=self.grid_suffix)

        if name is None and not ncl_name:
            field = self.field
            spec = self.vspec
        else:
            spec = self.spec.get(name, {}).get(level, {})
            if not spec and name is not None:
                raise errors.NoGraphicsDefinitionForVariable(name, level)
            field = self.get_field(ncl_name or self.ncl_name(spec))


        # 3D Variables have shape (fcst_hour, ygrid_0, xgrid_0)
        vals = field.sel(fcst_hour=self.fcst_hours)

        # Need to choose vertical level
        if len(field.shape) == 4:

            lev = vertical_index
            if vertical_index is None:
                lev = self.get_level(field, level, spec)
            vals = vals[:,lev, :, :]

        transforms = spec.get('transform')
        if transforms:
            vals = self.get_transform(transforms, vals)

        return vals
