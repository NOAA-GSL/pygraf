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
    def flru_colors(self) -> np.ndarray:

        ''' Default color map for Ceiling '''

        ctable = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([50, 15, 90, 120])
        return ctable

    @property
    @lru_cache()
    def frzn_colors(self) -> np.ndarray:

        ''' Default color map for Frozen Precip % '''

        grays = cm.get_cmap('Greys', 7)([0, 2])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([120, 90, 85, 80, 70, 60, 50, 25, 20, 15])
        return np.concatenate((grays, ncar))

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
    def graupel_colors(self) -> np.ndarray:

        ''' Default color map for Max Vertically Integrated Graupel '''

        grays = cm.get_cmap('Greys', 3)([0])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          (range(20, 128, 6))
        return np.concatenate((grays, ncar))

    @property
    @lru_cache()
    def hail_colors(self) -> np.ndarray:

        ''' Default color map for Hail diameter '''

        grays = cm.get_cmap('Greys', 2)([0])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([100, 15, 18, 20, 25, 60, 80, 85, 90])
        return np.concatenate((grays, ncar))

    @property
    @lru_cache()
    def heat_flux_colors(self) -> np.ndarray:

        ''' Default color map for Latent/Sensible Heat Flux '''

        grays = cm.get_cmap('Greys', 8)([6, 5, 4, 3, 2])
        ctable = ctables.colortables.get_colortable(self.vspec.get('cmap')) \
                    (range(0, 33, 2))
        return np.concatenate((grays, ctable))

    @property
    @lru_cache()
    def lcl_colors(self) -> np.ndarray:

        ''' Default color map for Lifted Condensation Level '''

        ctable = ctables.colortables.get_colortable(self.vspec.get('cmap')) \
                    (range(50, 180, 7)) # rainbow
        return ctable

    @property
    @lru_cache()
    def lifted_index_colors(self) -> np.ndarray:

        ''' Default color map for Lifted Index '''

        ctable = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          (range(4, 125, 4))
        ctable[14] = [1, 1, 1, 1]
        ctable[15] = [1, 1, 1, 1]
        return ctable

    @property
    @lru_cache()
    def mdn_colors(self) -> np.ndarray:

        ''' Default color map for Max Downdraft '''

        grays = cm.get_cmap('Greys', 2)([0])
        others = cm.get_cmap(self.vspec.get('cmap'), 18)(range(18, 1, -1), alpha=0.6)
        return np.concatenate((others, grays))

    @property
    @lru_cache()
    def mean_vvel_colors(self) -> np.ndarray:

        ''' Default color map for Mean Vertical Velocity '''

        ctable = cm.get_cmap(self.vspec.get('cmap'), 128)(range(0, 114, 6))
        ctable[9] = [1, 1, 1, 1]
        return ctable

    @property
    @lru_cache()
    def mup_colors(self) -> np.ndarray:

        ''' Default color map for Max Updraft '''

        grays = cm.get_cmap('Greys', 2)([0])
        others = cm.get_cmap(self.vspec.get('cmap'), 18)(range(1, 18, 1), alpha=0.6)
        return np.concatenate((grays, others))

    @property
    @lru_cache()
    def pbl_colors(self) -> np.ndarray:

        ''' Default color map for PBL Height '''

        return ctables.colortables.get_colortable(self.vspec.get('cmap')) \
                          (range(15, 60, 3))

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
    def radiation_colors(self) -> np.ndarray:

        ''' Default color map for Longwave Radiation '''

        grays = cm.get_cmap('Greys', 2)([0])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          (range(0, 126, 5))
        return np.concatenate((grays, ncar))

    @property
    @lru_cache()
    def radiation_bw_colors(self) -> np.ndarray:

        ''' Default grayscale map for Outgoing Shortwave Radiation '''

        return cm.get_cmap(self.vspec.get('cmap'), 128) \
                          (range(30, 110))

    @property
    @lru_cache()
    def radiation_mix_colors(self) -> np.ndarray:

        ''' Default color map for Longwave Radiation '''

        ncar = ctables.colortables.get_colortable(self.vspec.get('cmap')) \
                            (range(0, 40))
        grays = cm.get_cmap('Greys', 100)(range(10, 100))
        return np.concatenate((ncar, grays))

    @property
    @lru_cache()
    def rainbow12_colors(self) -> np.ndarray:

        ''' Default color map for ACPCP, ACSNOD, HLCY, RH, and SNOD '''

        grays = cm.get_cmap('Greys', 2)([0])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([15, 18, 20, 25, 50, 60, 70, 80, 85, 90, 100, 120])
        return np.concatenate((grays, ncar))

    @property
    @lru_cache()
    def shear_colors(self) -> np.ndarray:

        ''' Default color map for Vertical Shear '''

        ctable = cm.get_cmap(self.vspec.get('cmap'), 16) \
                          (range(5, 15))
        ctable[9] = [1, 1, 1, 1]
        return ctable

    @property
    @lru_cache()
    def snow_colors(self) -> np.ndarray:

        ''' Default color map for Snow fields '''

        grays = cm.get_cmap('Greys', 5)([0, 2])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([15, 18, 20, 25, 50, 60, 70, 80, 85, 90, 100])
        return np.concatenate((grays, ncar))

    @property
    @lru_cache()
    def soilm_colors(self) -> np.ndarray:

        ''' Default color map for Soil Moisture Availability '''

        ncar = cm.get_cmap(self.vspec.get('cmap'), 128)(range(0, 122, 11))
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

        ''' Default color map for Upper-Air Temperature '''

        ncolors = len(self.clevs)
        return cm.get_cmap(self.vspec.get('cmap', 'jet'), ncolors)(range(ncolors))

    @property
    @lru_cache()
    def tsfc_colors(self) -> np.ndarray:

        ''' Default color map for Surface Temperature '''  # WeatherBell-inspired scheme

        temp1 = cm.get_cmap('cool_r', 8)(range(0, 8))
        temp2 = cm.get_cmap('BuGn', 6)(range(2, 6))
        temp3 = cm.get_cmap('Greens_r', 4)(range(0, 4))
        temp4 = cm.get_cmap('RdPu_r', 8)(range(0, 8))
        temp5 = cm.get_cmap('BuPu', 5)(range(0, 4))
        temp6 = cm.get_cmap('RdYlBu_r', 10)(range(1, 10))
        temp7 = cm.get_cmap('RdYlGn', 10)(range(0, 10))

        return np.concatenate((temp1, temp2, temp3, temp4, temp5, temp6, temp7))

    @property
    @lru_cache()
    def terrain_colors(self) -> np.ndarray:

        ''' Default color map for Terrain '''

        ctable = ctables.colortables.get_colortable(self.vspec.get('cmap')) \
                    (range(0, 21, 1))
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
                          ([15, 18, 20, 25, 50, 60, 70, 80, 83, 90, 100, 120])
        return np.concatenate((grays, ncar))

    @property
    @lru_cache()
    def wind_colors(self) -> np.ndarray:

        ''' Default color map for Wind Speed '''

        low = cm.get_cmap(self.vspec.get('cmap'), 129)(range(129, 109, -5))
        high = cm.get_cmap(self.vspec.get('cmap'), 129)(range(18, 103, 6))
        return np.concatenate((low, high))

    @property
    @lru_cache()
    def wind_colors_high(self) -> np.ndarray:

        ''' Default color map for High Wind Speed '''

        low = cm.get_cmap(self.vspec.get('cmap'), 129)(range(129, 108, -7))
        high = cm.get_cmap(self.vspec.get('cmap'), 129)(range(18, 95, 7))
        return np.concatenate((low, high))
