# pylint: disable=too-many-public-methods
'''
This module sets the specifications for certain atmospheric variables. Typically
this is related to a spec that needs some level of computation, i.e. a set of
colors from a color map.
'''

import abc
from itertools import chain
from functools import lru_cache
from matplotlib import cm
import numpy as np
import yaml
from metpy.plots import ctables

class VarSpec(abc.ABC):

    '''
    Loads a yaml config file with spec settings. Also defines methods for
    declaring more complex specifications for variables based on settings within
    the config file.
    '''

    def __init__(self, config):

        with open(config, 'r') as cfg:
            self.yml = yaml.load(cfg, Loader=yaml.Loader)

    @property
    @lru_cache()
    def accumulated_precip_colors(self) -> np.ndarray:

        ''' Default color map for Accumulated Precipitation '''

        grays = cm.get_cmap('Greys', 6)([0, 3])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([18, 20, 25, 50, 60, 70, 80, 85, 90, 100, 120])
        return np.concatenate((grays, ncar))

    def centered_diff(self):

        ''' Returns the colors specified by levels and cmap in default spec, but
        with white center. '''

        clevs = self.vspec.get('clevs')
        nlev = len(clevs) + 1

        colors = cm.get_cmap(self.vspec.get('cmap'), nlev)(range(nlev))
        mid = nlev // 2

        colors[mid] = [1, 1, 1, 1]
        colors[mid-1] = [1, 1, 1, 1]

        return colors

    @property
    @lru_cache()
    def cin_colors(self) -> np.ndarray:

        ''' Default color map for Convective Inhibition '''

        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([120, 100, 90, 85, 80, 70, 60, 50, 25, 20, 18])
        grays = cm.get_cmap('Greys', 2)([0])
        return np.concatenate((ncar, grays))

    @property
    @abc.abstractmethod
    def clevs(self) -> np.ndarray:

        ''' An abstract method responsible for returning the np.ndarray of contour
        levels for a given field. Numpy arange supports non-integer values. '''

    @property
    @abc.abstractproperty
    def vspec(self):

        ''' The variable plotting specification. The level-specific subgroup
        from a config file like default_specs.yml. '''

    @property
    @lru_cache()
    def ceil_colors(self) -> np.ndarray:

        ''' Default color map for Ceiling '''

        grays = cm.get_cmap('Greys', 2)([0])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([15, 18, 20, 25, 50, 60, 70, 80, 85, 90, 100, 120])
        return np.concatenate((grays, ncar, grays))

    @property
    @lru_cache()
    def cldcov_colors(self) -> np.ndarray:

        ''' Default color map for Cloud Cover '''

        grays = cm.get_cmap('Greys', 7)([0, 1, 3])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([120, 100, 90, 85, 80, 70, 60, 50, 25, 20])
        return np.concatenate((grays, ncar))

    @property
    @lru_cache()
    def cref_colors(self) -> np.ndarray:

        ''' Default color map for Reflectivity '''

        ncolors = len(self.clevs)-1
        grays = cm.get_cmap('Greys', 5)([0])
        nws = ctables.colortables.get_colortable(self.vspec.get('cmap'))(range(ncolors))
        white = cm.get_cmap('Greys', 5)([0])
        return np.concatenate((grays, nws, white))

    @property
    @lru_cache()
    def dewp_colors(self) -> np.ndarray:

        ''' Default color map for Dew point temperature '''

        ctable = ctables.colortables.get_colortable(self.vspec.get('cmap')) \
                    (range(0, 42, 1)) # Carbone42_r
        return ctable

    @property
    @lru_cache()
    def goes_colors(self) -> np.ndarray:

        ''' Default color map for simulated GOES IR satellite '''

        grays = cm.get_cmap('Greys_r', 33)(range(33))
        ctable2 = ctables.colortables.get_colortable(self.vspec.get('cmap')) \
                    (range(65, 150))
        return np.concatenate((grays[-1:], grays, ctable2, grays[1:]))

    @property
    @lru_cache()
    def hlcy_colors(self) -> np.ndarray:

        ''' Default color map for Helicity '''

        grays = cm.get_cmap('Greys', 5)([0])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([15, 18, 20, 25, 50, 60, 70, 80, 85, 90, 100, 120])
        return np.concatenate((grays, ncar))

    @property
    @lru_cache()
    def lcl_colors(self) -> np.ndarray:

        ''' Default color map for Lifted Condensation Level '''

        ctable = ctables.colortables.get_colortable(self.vspec.get('cmap')) \
                    (range(60, 180, 6)) # rainbow
        return ctable

    @property
    @lru_cache()
    def pcp_colors(self) -> np.ndarray:

        ''' Default color map for Hourly Precipitation '''

        grays = cm.get_cmap('Greys', 6)([0, 3])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([25, 50, 60, 70, 80, 85, 90, 115])
        return np.concatenate((grays, ncar))

    @property
    @lru_cache()
    def ps_colors(self) -> np.ndarray:

        ''' Default color map for Surface Pressure '''

        grays = cm.get_cmap('Greys', 13)(range(13))
        segments = [[16, 53], [86, 105], [110, 151, 2], [172, 202, 2]]
        ncar = cm.get_cmap('gist_ncar', 200)(list(chain(*[range(*i) for i in segments])))
        return np.concatenate((grays, ncar))

    @property
    @lru_cache()
    def pw_colors(self) -> np.ndarray:

        ''' Default color map for Precipitable Water '''

        grays = cm.get_cmap('Greys', 5)([1, 3])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([120, 100, 95, 85, 80, 70, 65, 50, 25, 22, 20, 17])
        bupu = cm.get_cmap('BuPu', 15)([13, 14])
        cool = cm.get_cmap('cool', 15)([10, 9, 12, 7, 5])
        return np.concatenate((grays, ncar, bupu, cool))

    @property
    @lru_cache()
    def rh_colors(self) -> np.ndarray:

        ''' Default color map for Relative Humidity '''

        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([122, 114, 96, 85, 79, 69, 60, 49, 31, 21, 17, 109, 113])
        return ncar

    @property
    @lru_cache()
    def soilt_colors(self) -> np.ndarray:

        ''' Default color map for Soil Temperature '''

        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([15, 20, 25, 50, 70, 80, 83, 88, 110])
        return ncar

    @property
    @lru_cache()
    def soilw_colors(self) -> np.ndarray:

        ''' Default color map for Soil Moisture '''

        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([88, 83, 80, 70, 50, 25, 20, 15])
        return ncar

    @property
    @lru_cache()
    def t_colors(self) -> np.ndarray:

        ''' Default color map for Temperature '''

        ncolors = len(self.clevs)
        return cm.get_cmap(self.vspec.get('cmap', 'jet'), ncolors)(range(ncolors))

    @property
    @lru_cache()
    def terrain_colors(self) -> np.ndarray:

        ''' Default color map for Terrain '''

        ctable = ctables.colortables.get_colortable(self.vspec.get('cmap')) \
                    (range(0, 21, 1))
        return ctable

    @property
    @lru_cache()
    def vil_colors(self) -> np.ndarray:

        ''' Default color map for Vertically Integrated Liquid '''

        ctable = ctables.colortables.get_colortable(self.vspec.get('cmap')) \
                    ([0, 11, 15, 16, 4, 5, 12, 2, 19, 18, 7, 6, 8, 9, 10, 17])
        return ctable

    @property
    @lru_cache()
    def vis_colors(self) -> np.ndarray:

        ''' Default color map for Visibility '''

        grays = cm.get_cmap('Greys', 3)([1, 0])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([15, 18, 20, 25, 50, 60, 70, 80, 85, 90, 100, 120])
        return np.concatenate((grays, ncar))

    @property
    @lru_cache()
    def vvel_colors(self) -> np.ndarray:

        ''' Default color map for Vetical Velocity '''

        ncar1 = cm.get_cmap(self.vspec.get('cmap'), 128)([15, 18, 20, 25])
        grays = cm.get_cmap('Greys', 2)([0])
        ncar2 = cm.get_cmap(self.vspec.get('cmap'), 128)([60, 70, 80, 85, 90, 100, 120])
        return np.concatenate((ncar1, grays, ncar2))

    @property
    @lru_cache()
    def vort_colors(self) -> np.ndarray:

        ''' Default color map for Absolute Vorticity '''

        grays = cm.get_cmap('Greys', 2)([0])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([15, 18, 20, 25, 50, 60, 70, 80, 85, 90, 100, 120])
        return np.concatenate((grays, ncar))

    @property
    @lru_cache()
    def wind_colors(self) -> np.ndarray:

        ''' Default color map for Wind Speed '''

        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([21, 27, 46, 62, 69, 77, 83, 95, 102, \
                          119, 129, 17, 19, 21, 27, 46, 62, 69, 77])
        return ncar
