import datetime
from functools import lru_cache

from matplotlib import cm
import numpy as np
import pygrib

from .. import errors
from .. import specs
from .. import utils

LEV_DESCRIPT = {
    'surface': 'm',
    'heightAboveGround': 'm',
    'isobaricInhPa': 'hPa',
}

class GribFile():

    def __init__(self, filename):
        self.filename = filename
        self.contents = self._load()

    def _load(self):
        return pygrib.open(self.filename)

    def get_fields(self, level, lev_type, short_name):
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
        for grb in self.contents:
            print(grb.shortName, grb.typeOfLevel, grb.level)


class UPPData(GribFile, specs.VarSpec):

    def __init__(self, filename, level, lev_type, short_name, config='adb_graphics/default_specs.yml'):
        GribFile.__init__(self, filename)
        specs.VarSpec.__init__(self, config)

        self.level = level
        self.lev_type = lev_type
        self.short_name = short_name
        self.spec = self._load_spec(short_name)

    @property
    def anl_dt(self):
        return self.data.analDate

    @property
    def clevs(self):
        clev = self.spec['clevs']

        # Is clevs a list?
        if isinstance(clev, list):
            return clev

        # Does clev have a range call?
        if 'range' in clev.split('[')[0]:
            nums = [int(i) for i in clev.split(' ', 1)[1].strip('[').strip(']').split(',')]
            return np.arange(*nums)

        # Is clev a call to another function?
        try:
            return utils.get_func(clev)
        except:
            print(f'Check yaml file definition of CLEV for {self.short_name}.',
                  f'Must be a list, range, or function call!')

    @property
    def cmap(self):
        return cm.get_cmap(self.spec['cmap'])

    @property
    def colors(self):
        return self.__getattribute__(self.spec['colors'])

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
        return date.strftime('%Y%m%d %H UTC')

    @property
    def lev_unit(self):
        return LEV_DESCRIPT.get(self.lev_type, '')

    def _load_spec(self, varname):
        spec = self.yml.get(varname)
        sub_s = {}
        if 'subst' in spec.keys():
            sub_s = self._load_spec(spec['subst'])
        for k, val in spec.items():
            if k != 'subst':
                sub_s[k] = val
        return sub_s

    @property
    def fhr(self):
        return self.data['forecastTime']

    def short_summary(self):
        for k in sorted(self.data.keys()):
            val = self.data[k] if self.data.valid_key(k) else None
            print(f'{k}: {val}')

    @property
    def units(self):
        return self.spec.get('unit', self.data.parameterUnits)

    @property
    def valid_dt(self):
        return self.data.validDate

    @property
    def values(self):
        transform = self.spec.get('transform')
        if transform:
            return utils.get_func(transform)(self.data)
        return self.data.values

    @property
    @lru_cache()
    def wind(self):
        return [self.get_fields(self.level, self.lev_type, component) for component in ['u', 'v']]
