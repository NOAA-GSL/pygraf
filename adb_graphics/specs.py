# pylint: disable=too-many-public-methods
'''
This module sets the specifications for certain atmospheric variables. Typically
this is related to a spec that needs some level of computation, i.e. a set of
colors from a color map.
'''

import abc
from itertools import chain
from matplotlib import cm
from matplotlib import colors as mpcolors
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
    def ceil_colors(self) -> np.ndarray:

        ''' Default color map for Ceiling '''

        grays = cm.get_cmap('Greys', 2)([0])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([15, 18, 20, 25, 50, 60, 70, 80, 85, 90, 100, 120])
        return np.concatenate((grays, ncar, grays))

    @property
    def cldcov_colors(self) -> np.ndarray:

        ''' Default color map for Cloud Cover '''

        grays = cm.get_cmap('Greys', 7)([0, 1, 3])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([120, 100, 90, 85, 80, 70, 60, 50, 25, 20])
        return np.concatenate((grays, ncar))

    @property
    def cref_colors(self) -> np.ndarray:

        ''' Default color map for Reflectivity '''

        ncolors = len(self.clevs)-1
        grays = cm.get_cmap('Greys', 5)([0])
        nws = ctables.colortables.get_colortable(self.vspec.get('cmap'))(range(ncolors))
        white = cm.get_cmap('Greys', 5)([0])
        return np.concatenate((grays, nws, white))

    @property
    def dewp_colors(self) -> np.ndarray:

        ''' Default color map for Dew point temperature '''

        ctable = ctables.colortables.get_colortable(self.vspec.get('cmap')) \
                    (range(0, 42, 1)) # Carbone42_r
        return ctable

    @property
    def fire_power_colors(self) -> np.ndarray:

        ''' Default color map for fire power plot. '''

        blues = cm.get_cmap('Blues', 3)(range(3))
        green_orange = cm.get_cmap('RdYlGn_r', 10)([1, 7, 8, 9])
        return np.concatenate((blues, green_orange))

    def flru_colors(self) -> np.ndarray:

        ''' Default color map for Ceiling '''

        ctable = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([50, 15, 90, 120])
        return ctable

    @property
    def frzn_colors(self) -> np.ndarray:

        ''' Default color map for Frozen Precip % '''

        grays = cm.get_cmap('Greys', 7)([0, 2])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([120, 90, 85, 80, 70, 60, 50, 25, 20, 15])
        return np.concatenate((grays, ncar))

    @property
    def goes_colors(self) -> np.ndarray:

        ''' Default color map for simulated GOES IR satellite '''

        grays = cm.get_cmap('Greys_r', 33)(range(33))
        ctable2 = ctables.colortables.get_colortable(self.vspec.get('cmap')) \
                    (range(65, 150))
        return np.concatenate((grays[-1:], grays, ctable2, grays[1:]))

    @property
    def graupel_colors(self) -> np.ndarray:

        ''' Default color map for Max Vertically Integrated Graupel '''

        grays = cm.get_cmap('Greys', 3)([0])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          (range(20, 128, 6))
        return np.concatenate((grays, ncar))

    @property
    def hail_colors(self) -> np.ndarray:

        ''' Default color map for Hail diameter '''

        grays = cm.get_cmap('Greys', 2)([0])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([100, 15, 18, 20, 25, 60, 80, 85, 90])
        return np.concatenate((grays, ncar))

    @property
    def heat_flux_colors(self) -> np.ndarray:

        ''' Default color map for Latent/Sensible Heat Flux '''

        grays = cm.get_cmap('Greys', 8)([6, 5, 4, 3, 2])
        ctable = ctables.colortables.get_colortable(self.vspec.get('cmap')) \
                    (range(0, 33, 2))
        return np.concatenate((grays, ctable))

    @property
    def lcl_colors(self) -> np.ndarray:

        ''' Default color map for Lifted Condensation Level '''

        ctable = ctables.colortables.get_colortable(self.vspec.get('cmap')) \
                    (range(50, 180, 7)) # rainbow
        return ctable

    @property
    def lifted_index_colors(self) -> np.ndarray:

        ''' Default color map for Lifted Index '''

        ctable = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          (range(4, 125, 4))
        ctable[14] = [1, 1, 1, 1]
        ctable[15] = [1, 1, 1, 1]
        return ctable

    @property
    def mdn_colors(self) -> np.ndarray:

        ''' Default color map for Max Downdraft '''

        grays = cm.get_cmap('Greys', 2)([0])
        others = cm.get_cmap(self.vspec.get('cmap'), 18)(range(18, 1, -1), alpha=0.6)
        return np.concatenate((others, grays))

    @property
    def mean_vvel_colors(self) -> np.ndarray:

        ''' Default color map for Mean Vertical Velocity '''

        ctable = cm.get_cmap(self.vspec.get('cmap'), 128)(range(0, 114, 6))
        ctable[9] = [1, 1, 1, 1]
        return ctable

    @property
    def mup_colors(self) -> np.ndarray:

        ''' Default color map for Max Updraft '''

        grays = cm.get_cmap('Greys', 2)([0])
        others = cm.get_cmap(self.vspec.get('cmap'), 18)(range(1, 18, 1), alpha=0.6)
        return np.concatenate((grays, others))

    @property
    def pbl_colors(self) -> np.ndarray:

        ''' Default color map for PBL Height '''

        return ctables.colortables.get_colortable(self.vspec.get('cmap')) \
                          (range(15, 60, 3))

    @property
    def pcp_colors(self) -> np.ndarray:

        ''' Default color map for Hourly Precipitation '''

        grays = cm.get_cmap('Greys', 6)([0, 3])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([25, 50, 60, 70, 80, 85, 90, 115])
        return np.concatenate((grays, ncar))

    @property
    def ps_colors(self) -> np.ndarray:

        ''' Default color map for Surface Pressure '''

        grays = cm.get_cmap('Greys', 13)(range(13))
        segments = [[16, 53], [86, 105], [110, 151, 2], [172, 202, 2]]
        ncar = cm.get_cmap('gist_ncar', 200)(list(chain(*[range(*i) for i in segments])))
        return np.concatenate((grays, ncar))

    @property
    def pw_colors(self) -> np.ndarray:

        ''' Default color map for Precipitable Water '''

        grays = cm.get_cmap('Greys', 5)([1, 3])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([120, 100, 95, 85, 80, 70, 65, 50, 25, 22, 20, 17])
        bupu = cm.get_cmap('BuPu', 15)([13, 14])
        cool = cm.get_cmap('cool', 15)([10, 9, 12, 7, 5])
        return np.concatenate((grays, ncar, bupu, cool))

    @property
    def radiation_colors(self) -> np.ndarray:

        ''' Default color map for Longwave Radiation '''

        grays = cm.get_cmap('Greys', 2)([0])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          (range(0, 126, 5))
        return np.concatenate((grays, ncar))

    @property
    def radiation_bw_colors(self) -> np.ndarray:

        ''' Default grayscale map for Outgoing Shortwave Radiation '''

        return cm.get_cmap(self.vspec.get('cmap'), 128) \
                          (range(30, 110))

    @property
    def radiation_mix_colors(self) -> np.ndarray:

        ''' Default color map for Longwave Radiation '''

        ncar = ctables.colortables.get_colortable(self.vspec.get('cmap')) \
                            (range(0, 40))
        grays = cm.get_cmap('Greys', 100)(range(10, 100))
        return np.concatenate((ncar, grays))

    @property
    def rainbow12_colors(self) -> np.ndarray:

        ''' Default color map for ACPCP, ACSNOD, HLCY, RH, and SNOD '''

        grays = cm.get_cmap('Greys', 2)([0])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([15, 18, 20, 25, 50, 60, 70, 80, 85, 90, 100, 120])
        return np.concatenate((grays, ncar))

    @property
    def rainbow12_reverse(self) -> np.ndarray:

        ''' Default color map for min helicity '''

        return np.flip(self.rainbow12_colors, 0)

    @property
    def shear_colors(self) -> np.ndarray:

        ''' Default color map for Vertical Shear '''

        ctable = cm.get_cmap(self.vspec.get('cmap'), 16) \
                          (range(5, 15))
        ctable[9] = [1, 1, 1, 1]
        return ctable

    @property
    def smoke_colors(self) -> np.ndarray:

        ''' Default color map for smoke plots. '''

        white = cm.get_cmap('Greys', 2)([0])
        blues = cm.get_cmap('Blues', 6)(range(1, 5))
        green_yellow_red = cm.get_cmap('RdYlGn_r', 18)([1, 3, 5, 9, 12, 13, 14, 16, 18])
        purple = np.array([mpcolors.to_rgba('xkcd:vivid purple')])
        return np.concatenate((white,blues, green_yellow_red, purple))


    @property
    def snow_colors(self) -> np.ndarray:

        ''' Default color map for Snow fields '''

        grays = cm.get_cmap('Greys', 5)([0, 2])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([15, 18, 20, 25, 50, 60, 74, 81, 85, 90, 100])
        return np.concatenate((grays, ncar))

    @property
    def soilm_colors(self) -> np.ndarray:

        ''' Default color map for Soil Moisture Availability '''

        ncar = cm.get_cmap(self.vspec.get('cmap'), 128)(range(0, 122, 11))
        return ncar

    @property
    def soilw_colors(self) -> np.ndarray:

        ''' Default color map for Soil Moisture '''

        grays = cm.get_cmap('Greys', 2)([1])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 110) \
               ([0, 10, 20, 25, 35, 40, 60, 73, 80, 85, 95, 105])
        return np.concatenate((grays, ncar))

    @property
    def t_colors(self) -> np.ndarray:

        ''' Default color map for Upper-Air Temperature '''

        ncolors = len(self.clevs)
        return cm.get_cmap(self.vspec.get('cmap', 'jet'), ncolors)(range(ncolors))

    @property
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
    def terrain_colors(self) -> np.ndarray:

        ''' Default color map for Terrain '''

        ctable = ctables.colortables.get_colortable(self.vspec.get('cmap')) \
                    (range(0, 21, 1))
        return ctable

    @property
    def vis_colors(self) -> np.ndarray:

        ''' Default color map for Visibility

        section names are based on Aviation Flight Rule visibility categories
        LIFR (Low Instrument Flight Rules) -- less than 1 mile
        IFR (Instrument Flight Rules) -- 1 mile to less than 3 miles
        MVFR (Marginal Visual Flight Rules) -- 3 to 5 miles
        VFR (Visual Flight Rules) -- greater than 5 miles
        '''

        lifr = cm.get_cmap('RdPu_r', 20)(range(0, 11))
        ifr = cm.get_cmap('autumn', 30)(range(0, 30))
        mvfr = cm.get_cmap('Blues', 20)(range(10, 20))
        vfr1 = cm.get_cmap('YlGn_r', 60)(range(0, 50))
        vfr2 = cm.get_cmap('Reds', 15)(np.full(51, 1))

        return np.concatenate((lifr, ifr, mvfr, vfr1, vfr2))

    @property
    def vvel_colors(self) -> np.ndarray:

        ''' Default color map for Vetical Velocity '''

        ncar1 = cm.get_cmap(self.vspec.get('cmap'), 128)([15, 18, 20, 25])
        grays = cm.get_cmap('Greys', 2)([0])
        ncar2 = cm.get_cmap(self.vspec.get('cmap'), 128)([60, 70, 80, 85, 90, 100, 120])
        return np.concatenate((ncar1, grays, ncar2))

    @property
    def vort_colors(self) -> np.ndarray:

        ''' Default color map for Absolute Vorticity '''

        grays = cm.get_cmap('Greys', 2)([0])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128) \
                          ([15, 18, 20, 25, 50, 60, 70, 80, 83, 90, 100, 120])
        return np.concatenate((grays, ncar))

    @property
    def wind_colors(self) -> np.ndarray:

        ''' Default color map for Wind Speed '''

        low = cm.get_cmap(self.vspec.get('cmap'), 129)(range(129, 109, -5))
        high1 = cm.get_cmap(self.vspec.get('cmap'), 129)(range(16, 29, 3))
        high2 = cm.get_cmap(self.vspec.get('cmap'), 129)(range(48, 103, 6))
        return np.concatenate((low, high1, high2))

    @property
    def wind_colors_high(self) -> np.ndarray:

        ''' Default color map for High Wind Speed '''

        low = cm.get_cmap(self.vspec.get('cmap'), 129)(range(129, 108, -7))
        high1 = cm.get_cmap(self.vspec.get('cmap'), 129)(range(16, 29, 4))
        high2 = cm.get_cmap(self.vspec.get('cmap'), 129)(range(46, 95, 7))
        return np.concatenate((low, high1, high2))
