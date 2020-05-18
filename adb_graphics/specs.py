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
    def pw_colors(self) -> np.ndarray:

        ''' Default color map for Precipitable Water '''

        grays = cm.get_cmap('Greys', 5)([1,3])
        ncar = cm.get_cmap(self.vspec.get('cmap'), 128)([120,100,95,85,80,70,65,50,25,22,20,17])
        bupu = cm.get_cmap('BuPu', 15)([13,14])
        cool = cm.get_cmap('cool', 15)([10,9,12,7,5])
        return np.concatenate((grays, ncar, bupu, cool))

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
    def t_colors(self) -> np.ndarray:

        ''' Default color map for Temperature '''

        ncolors = len(self.clevs)
        return cm.get_cmap(self.vspec.get('cmap', 'jet'), ncolors)(range(ncolors))

    @property
    @lru_cache()
    def vvel_colors(self) -> np.ndarray:

        ''' Default color map for Vertical Velocity '''

        ncar1 = cm.get_cmap(self.vspec.get('cmap'), 128)([15,18,20,25])
        grays = cm.get_cmap('Greys', 2)([0])
        ncar2 = cm.get_cmap(self.vspec.get('cmap'), 128)([60,70,80,85,90,100,120])
        return np.concatenate((ncar1, grays, ncar2))

    @property
    @lru_cache()
    def vort_colors(self) -> np.ndarray:

        ''' Default color map for Absolute Vorticity '''

        ncar = cm.get_cmap(self.vspec.get('cmap'), 128)([15,18,20,25,50,60,70,80,85,90,100,120])
        return ncar

