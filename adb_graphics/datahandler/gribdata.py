# pylint: disable=invalid-name,too-few-public-methods

'''
Classes that handle the specifics of grib files from UPP.
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

class UPPData(specs.VarSpec):

    '''
    Class provides interface for accessing field  data from UPP in
    Grib2 format.

    Input:
        ds:          xarray dataset from grib file
        short_name:  name of variable corresponding to entry in specs configuration

    Keyword Arguments:
        config:      path to a user-specified configuration file
        model:       string describing the model type
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
    def field(self):

        ''' Wrapper that calls get_field method for the current variable.
        Returns the NioVariable object '''

        return self._get_field(self.ncl_name(self.vspec))

    def field_diff(self, values, variable2, level2, **kwargs):

        # pylint: disable=unused-argument

        ''' Subtracts the values from variable2 from self.field. '''

        value2 = self.values(name=variable2, level=level2)
        diff = values - value2
        value2.close()

        return diff

    def field_mean(self, values, variable, levels, **kwargs):

        # pylint: disable=unused-argument

        ''' Returns the mean of the values. '''

        fsum = np.zeros_like(values)
        for level in levels:
            val_lev = self.values(name=variable, level=level)
            fsum = fsum + val_lev
            val_lev.close()

        return fsum / len(levels)

    def _get_data_levels(self, vertical_dim):

        ''' Return a list of vertical dimension values corresponding to the
        requested vertical dimension to get the values of those dimensions '''

        fcst_hr = 0 if self.ds.sizes.get('fcst_hr', 0) <= 1 else int(self.fhr)

        ret = []
        for dim in [var for var in self.ds.variables \
                if vertical_dim in var]:

            # Get the current forecast hour slice, if it's in the dataset
            selector = {'fcst_hr': fcst_hr} if 'fcst_hr' in self.ds[dim].dims else {}
            ret.append(self.ds[dim].sel(**selector).values)
        return ret

    def _get_field(self, ncl_name):

        ''' Given an ncl_name, return the NioVariable object. '''

        try:
            field = self.ds[ncl_name.format(level_type=self.level_type)]
        except KeyError:
            raise errors.GribReadError(f'{ncl_name}')
        return field

    def _get_level(self, field, level, spec, **kwargs):

        ''' Returns the value of the level to for a 3D array

        Arguments:

          field      dataset object for a given variable
          level      string describing the level atmospheric level; corresponds
                     to a key in default specs
          spec       the specifications dictionary to use for the variable in
                     question


        Keyword Arguments:
          split      bool sometimes passed in through transforms that indicates
                     a level string should be split, e.g. 06km.


        Return:

          Integer value corresponding to the array index for the atmospheric
          level.
        '''

        # The index of the requested level
        lev = spec.get('vertical_index')
        if lev is not None:
            return lev

        vertical_dim = self.vertical_dim(field)

        # numeric_level returns a list of length 1 (e.g. [500] for 500 mb) or of
        # length 2 when split=True and it's like 0-6 km, so returns [0, 6000]
        requested_level, _ = self.numeric_level(level=level,
                                                split=kwargs.get('split', spec.get('split')),
                                                )

        # data_levels contains a list of vertical dimension values
        data_levels = self._get_data_levels(vertical_dim)

        # For split-level variables, like 0-6km, find the matching index by
        # looping through both the possible vertical level arrays.
        if len(data_levels) == 2 and len(requested_level) == 2:
            for lev, levset in enumerate(zip(*[list(lev) for lev in data_levels])):
                if sorted(levset) == requested_level:
                    return lev

        # For single-level variables, like 500mb, use the argwhere function to
        # return the matching index
        if len(requested_level) == 1:
            for dim_levels in data_levels:
                lev = np.argwhere(dim_levels == requested_level[0])
                try:
                    if lev or lev == [0]:
                        lev = int(lev[0])
                        return lev
                except ValueError:
                    print(f'BAD LEVEL is {lev} for {field.name}')

            print(f"Could not find a level for {field.name} at requested \
                  level = {requested_level} for variable levels = {data_levels}. Index \
                  was {lev}.")

        # If neither of those cases worked out appropriately, raise an error.
        msg = f'Length of requested_level ({len(requested_level)}) or '\
              f'data_levels ({len(data_levels)}) bad!' \
              f' {level} {field.name}'
        raise ValueError(msg)

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

    @lru_cache()
    def get_xypoint(self, site_lat, site_lon) -> tuple:

        '''
        Return the X, Y grid point corresponding to the site location. No
        interpolation is used.
        '''

        lats, lons = self.latlons()
        adjust = 360 if np.any(lons < 0) else 0
        lons = lons + adjust
        max_x, max_y = np.shape(lats)

        # Numpy magic to grab the X, Y grid point nearest the profile site
        # pylint: disable=unbalanced-tuple-unpacking
        x, y = np.unravel_index((np.abs(lats - site_lat) \
               + np.abs(lons - site_lon)).argmin(), lats.shape)
        # pylint: enable=unbalanced-tuple-unpacking

        if x <= 0 or y <= 0 or x >= max_x or y >= max_y:
            print(f'site location is outside your domain! {site_lat} {site_lon}')
            return(-1, -1)

        return (x, y)

    @property
    def grid_suffix(self):

        ''' Return the suffix of the first variable with 4 sections (split on _)
        in the file. This should correspond to the grid tag. '''

        for var in self.ds.keys():
            vsplit = var.split('_')
            if len(vsplit) == 4:
                return vsplit[-1]
        return 'GRID NOT FOUND'


    def latlons(self):

        ''' Returns the set of latitudes and longitudes '''

        coords = sorted([c for c in list(self.ds.coords) if
                         any(ele in c for ele in ['lat', 'lon'])])
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

        if name is None:
            print(f"Cannot find ncl_name for: ")
            for key, value in spec.items():
                print(f'{key}: {value}')
            raise KeyError

        # The level_type for the entire atmosphere could be L10 or L200. Thanks
        # Grib2! Handle that in "try" statement when reading file.

        name = name if isinstance(name, list) else [name]

        for try_name in name:
            try_name = try_name.format(fhr=self.fhr,
                                       grid=self.grid_suffix,
                                       level_type=self.level_type)

            try:
                self._get_field(try_name)
            except errors.GribReadError:
                continue
            else:
                return try_name

        msg = f'Could not find any of {name} in input file'
        raise errors.GribReadError(msg)

    def numeric_level(self, index_match=True, level=None, split=None):

        '''
        Split the numeric level and unit associated with the level key.

        A blank string is returned for lev_val for levels that do not contain a
        numeric, e.g., 'sfc' or 'ua'.
        '''

        level = level if level else self.level

        # Gather all the numbers in the string
        lev_val = ''.join([c for c in level if (c in digits or c == '.')])

        # Convert the numbers to a list, and make integers or floats
        if lev_val:
            if split is not None:
                lev_val = [int(lev) for lev in lev_val]
            else:
                lev_val = [float(lev_val) if '.' in lev_val else int(lev_val)]

        # Gather all the letters
        lev_unit = ''.join([c for c in level if c in ascii_letters])

        if index_match:
            if lev_unit == 'cm':
                lev_val = [val / 100. for val in lev_val]
            if lev_unit in ['mb', 'mxmb']:
                lev_val = [val * 100. for val in lev_val]
            if lev_unit in ['in', 'km', 'mn', 'mx', 'sr']:
                lev_val = [val * 1000. for val in lev_val]

        return lev_val, lev_unit

    @staticmethod
    def opposite(values, **kwargs):
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

    @staticmethod
    def vertical_dim(field):

        ''' Determine the vertical dimension of the variable by looking through
        the field's dimensions for one that includes "lv". Return the first
        matching instance. '''

        vert_dim = [dim for dim in field.dims if ('lv' in dim or 'probability' in dim)]
        if vert_dim:
            return vert_dim[0]
        return ''


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
        member:      integer describing the ensemble member number to
                     grab data for
    '''

    def __init__(self, ds, level, short_name, **kwargs):

        super().__init__(ds, short_name, **kwargs)

        self.level = level
        self.contour_kwargs = kwargs.get('contour_kwargs', {})
        # The following print statement always shows {} for self.contour_kwargs
        # print(f'in fieldData: short_name = {short_name} self.contour_kwargs = {self.contour_kwargs}')
        self.mem = kwargs.get('member', None)

    def aviation_flight_rules(self, values, **kwargs):
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

        vis.close()

        return xr.DataArray(flru)

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
        if self.model in ['global', 'obs']:
            ret = [lat[-1], lat[0], lon[0], lon[-1]]
        else:
            ret = [lat[0, 0], lat[-1, -1], lon[0, 0], lon[-1, -1]]

        return ret

    def fire_weather_index(self, values, **kwargs):

        # pylint: disable=unused-argument

        '''
        Generates a field of Fire Weather Index

        This method uses wrfprs data to find regions where
        weather conditions are most likely to lead to wildfires.

        '''

        # Gather fields from the input
        veg = values # Chose this value as the main one in the default_specs
        temp = self.values(name='temp', level='2m', do_transform=False)
        dewpt = self.values(name='dewp', level='2m', do_transform=False)
        weasd = self.values(name='weasd', level='sfc', do_transform=False)
        gust = self.values(name='gust', level='10m', do_transform=False)
        soilm = self.values(name='soilm', level='sfc', do_transform=False)

        # A few derived fields
        dewpt_depression = temp - dewpt
        dewpt_depression = np.where(dewpt_depression < 0, 0, dewpt_depression)
        dewpt_depression = np.maximum(15.0, dewpt_depression)
        gust_max = np.maximum(3.0, gust)

        snowc = (25.0 - weasd) / 25.0
        snowc = np.where(snowc > 0.0, snowc, 0.0)

        mois = 0.01*(100.0 - soilm)

        # Set urban (13), snow/ice (15), barren (16), and water (17) to 0.
        for vegtype in [13, 15, 16, 17]:
            veg = np.where(veg == vegtype, 0, veg)

        # Set all others vegetation types to 1
        veg = np.where(veg > 0, 1, veg)

        fwi = veg * (2.15 *
                     gust_max *
                     dewpt_depression *
                     (mois ** 6.42) *
                     snowc)

        fwi = fwi / 10.0

        temp.close()
        dewpt.close()
        weasd.close()
        gust.close()
        soilm.close()

        return fwi

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
        lat_var = [var for var in self.field.coords if 'lat' in var][0]

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
        elif self.grid_suffix == 'GLL0':
            attrs = []
            grid_info['projection'] = 'cyl'
        else:
            attrs = []
            grid_info['projection'] = 'rotpole'

            # CenterLon in RAP and Longitude_of_southern_pole in RRFS
            lon_0 = lat.attrs.get('CenterLon', lat.attrs.get('Longitude_of_southern_pole'))
            grid_info['lon_0'] = lon_0[0] - 360

            # CenterLat in RAP and Latitude_of_southern_pole in RRFS
            center_lat = lat.attrs.get('CenterLat', lat.attrs.get('Latitude_of_southern_pole'))
            grid_info['o_lat_p'] = - center_lat[0] if center_lat[0] < 0 else 90 - center_lat[0]

            grid_info['o_lon_p'] = 180

        for attr in attrs:
            bm_arg = ncl_to_basemap[attr]
            val = lat.attrs[attr]
            val = val[0] if isinstance(val, np.ndarray) else val
            grid_info[bm_arg] = val
            del val

        del lat

        return grid_info

    def run_total(self, values, **kwargs):

        ''' Sums over all the forecast lead times available. '''

        # pylint: disable=unused-argument,no-self-use

        return values.sum(dim='fcst_hr')

    def supercooled_liquid_water(self, values, **kwargs):

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

        pres_sfc.close()
        pres_nat_lev.close()
        temp.close()
        cloud_mixing_ratio.close()
        rain_mixing_ratio.close()
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

    def values(self, level=None, name=None, **kwargs):

        '''
        Returns the numpy array of values at the requested level for the
        variable after applying any unit conversion to the original data.

        Optional Input:
            name       the name of a field other than defined in self
            level      the desired level of the named field

        Keyword Args:
            do_transform    bool flag. to call, or not, the transform specified
                            in specs (default: True)
            ncl_name        the NCL-assigned Grib2 name (default: '')
            one_lev         bool flag. if True, get the single level of the variable
                            (default: True)
            vertical_index  the index (int) of the desired vertical level
        '''

        level = level if level else self.level

        one_lev = kwargs.get('one_lev', True)
        vertical_index = kwargs.get('vertical_index')

        ncl_name = kwargs.get('ncl_name', '')
        ncl_name = ncl_name.format(fhr=self.fhr, grid=self.grid_suffix)

        do_transform = kwargs.get('do_transform', True)

        if name is None and not ncl_name:

            # Use field and spec from the current object
            field = self.field
            spec = self.vspec

        else:

            # Get the spec dict and ncl_name for the given variable name
            spec = self.spec.get(name, {}).get(level, {})
            if not spec and name is not None:
                raise errors.NoGraphicsDefinitionForVariable(name, level)
            field = self._get_field(ncl_name or self.ncl_name(spec))

        lev = vertical_index
        vals = field
        if one_lev:

            # Check if it's a 3D variable (lv in any dimension field)
            dim_name = self.vertical_dim(field)

            if dim_name: # Field has a vertical dimension

                # Use vertical_index if provided in kwargs
                lev = vertical_index if vertical_index is not None else \
                        self._get_level(field, level, spec)

                if lev is None or dim_name is None:
                    print(f'ERROR: Could not find dim_name ({dim_name}) or' \
                          f'lev {lev} for {vals}')
                    raise ValueError

                try:
                    vals = vals.isel(**{dim_name: lev})
                except:
                    print(f'Error for {vals.name} : {dim_name} {lev} \
                            {level} {spec}')
                    raise

        if self.mem is not None:
            vals = vals.isel(**{'ens_mem': self.mem})

        # Select a single forecast hour (only if there are many)
        if not spec.get('accumulate', False):
            if 'fcst_hr' in vals.dims:
                fcst_hr = 0 if self.ds.sizes['fcst_hr'] <= 1 else int(self.fhr)
                vals = vals.sel(**{'fcst_hr': fcst_hr})

        transforms = spec.get('transform')
        if transforms and do_transform:
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

        mag = conversions.magnitude(field1, field2)
        field1.close()
        field2.close()

        return mag

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

        # Convert the string to a number. Longitude should be positive for all
        # these sites.
        # The conus_raobs file uses -180 to 180, but leaves off the minus sign,
        # i.e., the values are in degrees West. So, first we need to add the
        # minus sign to convert the longitude to deg East, and then need to
        # adjust to the 0 to 360 system.
        self.site_lat = float(lat)
        self.site_lon = -float(lon) # lons are -180 but without minus sign in input file
        if self.site_lon < 0:
            self.site_lon = self.site_lon + 360.0

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
            split            bool flag. if True, level string numbers are split
                             into a list, e.g. used to get [0, 6000] from 06km
            vertical_index   the index of the required level
        '''

        # Set the defaults here since this is an instance of an abstract method
        # level refers to the level key in the specs file.
        level = level if level is not None else 'ua'

        if not name:
            name = self.short_name

        one_lev = kwargs.get('one_lev', False)
        vertical_index = kwargs.get('vertical_index')
        split = kwargs.get('split')

        # Retrive the location for the profile
        x, y = self.get_xypoint(self.site_lat, self.site_lon)

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
                    lev = self._get_level(field, level, var_spec, split=split)
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
