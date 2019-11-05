from itertools import chain
from functools import lru_cache
from matplotlib import cm
import numpy as np

import yaml

class VarSpec():

    def __init__(self, config):

        with open(config, 'r') as cfg:
            self.yml = yaml.load(cfg, Loader=yaml.SafeLoader)

    @property
    @lru_cache()
    def ps_colors(self):
        ''' Default color map for Surface Pressure '''
        grays = cm.get_cmap('Greys', 13)(range(13)), cm.get_cmap('gist_ncar')
        segments = [[16, 53], [86, 105], [110, 151, 2], [172, 202, 2]]
        ncar = cm.get_cmap('gist_ncar', 200)(list(chain(*[range(*i) for i in segments])))
        return np.concatenate((grays[0], ncar))
