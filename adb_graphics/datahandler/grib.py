'''
Classes that handle generic grib file handling, and those that handle the
specifics of grib files from UPP.
'''

import datetime
from functools import lru_cache

from matplotlib import cm
import numpy as np
import pygrib

from .. import errors
from .. import specs
from .. import utils

# Define the unit of the various vertical level types. Used for figure titles.
LEV_DESCRIPT = {
    'surface': 'm',
    'heightAboveGround': 'm',
    'isobaricInhPa': 'hPa',
}

class GribFile():

    ''' Wrappers and helper functions for interfacing with pygrib '''

    def __init__(self, filename):
        self.filename = filename
        self.contents = self._load()

    def _load(self):

        ''' Internal method that opens the grib file. Returns a grib message
        iterator. '''

        return pygrib.open(self.filename) # pylint: disable=c-extension-no-member

    def get_fields(self, level, lev_type, short_name):

        ''' Given three parameters (level, level type, and the short name of the
        variable), this method queries the grib file for the requested field and
        returns a single field. Pygrib generates an error if no matching fields
        are found. This method raises an exception if multiple fields are found
        matching the search criteria. '''

        fields = self.contents.select(
            shortName=short_name,
            typeOfLevel=lev_type,
            level=level,
        )
        if len(fields) > 1:
            msg = f'{len(fields)} fields were found for {short_name} at {level} {lev_type}'
            raise errors.FieldNotUnique(msg)
        return fields[0]

    @property
    def list_vars(self):

        ''' Helper functions that lists all variables, giving their short name,
        level type, and level values. '''

        for grb in self.contents:
            print(grb.shortName, grb.typeOfLevel, grb.level)


class UPPData(GribFile, specs.VarSpec):

    ''' Class handles grib file manipulation for a given variable in a UPP output
    file. '''

    def __init__(self, filename, level, lev_type, short_name, **kwargs):

        # Parse kwargs first
        self.season = kwargs.get('season', 'warm')
        config = kwargs.get('config', 'adb_graphics/default_specs.yml')

        GribFile.__init__(self, filename)
        specs.VarSpec.__init__(self, config)

        self.level = level
        self.lev_type = lev_type
        self.short_name = short_name
        self.spec = self._load_spec(level, short_name)

    @property
    def anl_dt(self) -> datetime.datetime:

        ''' Returns the initial time of the grib file as a datetime object from
        the grib file.'''

        return self.data.analDate

    @property
    def clevs(self) -> list:

        '''
        Uses the information contained in the yaml config file to determine
        the set of levels to be contoured. Returns the list of levels.

        The yaml file "clevs" key may contain a list, a range, or a call to a
        function. The logic to parse those options is included here.
        '''

        clev = self.spec['clevs']
        clev = clev.get(self.season, clev) if isinstance(clev, dict) else clev

        # Is clevs a list?
        if isinstance(clev, list):
            return clev

        # Does clev have a range call?
        if 'range' in clev.split('[')[0]:
            nums = [float(i) for i in clev.split(' ', 1)[1].strip('[').strip(']').split(',')]
            return list(np.arange(*nums))

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

        return cm.get_cmap(self.spec['cmap'])

    @property
    def colors(self) -> list:

        '''
        Returns a list of colors, specified by the config key "colors".

        The yaml file "colors" key may contain a list or a function to be
        called.
        '''

        try:
            return self.__getattribute__(self.spec['colors'])()
        except AttributeError:
            return self.spec.get('colors')

    @property
    def corners(self):

        '''
        Returns lat and lon of lower left (ll) and upper right(ur) corners:
               ll_lat, ur_lat, ll_lon, ur_lon
        '''

        lat, lon = self.data.latlons()
        return [lat[0, 0], lat[-1, -1], lon[0, 0], lon[-1, -1]]

    @property
    @lru_cache()
    def data(self):

        ''' Wrapper that calls get_fields method for the current variable. '''

        try:
            field = self.get_fields(
                level=self.level,
                lev_type=self.lev_type,
                short_name=self.short_name,
            )
        except errors.FieldNotUnique:
            print('NetCDF field not unique!')
            raise
        return field

    @staticmethod
    def date_to_str(date: datetime) -> str:

        ''' Returns a formatted string (for graphic title) from a datetime
        object'''

        return date.strftime('%Y%m%d %H UTC')

    @property
    def lev_unit(self):

        ''' Returns the unit for the variable's lev_type. '''

        return LEV_DESCRIPT.get(self.lev_type, '')

    def _load_spec(self, lev, varname) -> dict:

        '''
        Loads the configuration for the current variable from the yaml
        config. Returns only a single section.

        Handles parsing the special key "subst" to fill in unset key, value
        pairs from other yaml sections.
        '''

        spec = self.yml.get(varname)
        spec = spec.get(lev, spec)
        sub_s = {}
        if 'subst' in spec.keys():
            sub_s = self._load_spec(lev, spec['subst'])
        for k, val in spec.items():
            if k != 'subst':
                sub_s[k] = val
        return sub_s

    @property
    def fhr(self) -> str:

        ''' Returns the forecast hour from the grib file. '''

        return self.data['forecastTime']

    def short_summary(self):

        ''' Helper that prints out the keys describing the variable requested. '''

        for k in sorted(self.data.keys()):
            val = self.data[k] if self.data.valid_key(k) else None
            print(f'{k}: {val}')

    @property
    def ticks(self) -> int:

        ''' Returns the number of color bar tick marks from the yaml config
        settings. '''

        return self.spec.get('ticks', 10)

    @property
    def units(self) -> str:

        ''' Returns the variable unit from the yaml config, if available. If not
        specified in the yaml file, returns the value set in the Grib file. '''

        return self.spec.get('unit', self.data.parameterUnits)

    @property
    def valid_dt(self) -> datetime.datetime:

        ''' Returns a datetime object corresponding to the forecast hour's valid
        time as set in the Grib file. '''

        return self.data.validDate

    @property
    def values(self) -> np.ndarray:

        ''' Returns the numpy array of values for the variable after applying any
        unit conversion to the original data. '''

        transform = self.spec.get('transform')
        if transform:
            return utils.get_func(transform)(self.data.values)
        return self.data.values

    @property
    @lru_cache()
    def wind(self) -> [np.ndarray, np.ndarray]:

        ''' Returns the u, v wind components as a list (length 2) of arrays. '''

        return [self.get_fields(self.level, self.lev_type, component) for component in ['u', 'v']]
