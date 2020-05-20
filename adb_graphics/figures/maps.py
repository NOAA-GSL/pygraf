# pylint: disable=invalid-name,too-few-public-methods

'''
Module contains classes relevant to plotting maps. The Map class handles all the
functionality related to a Basemap, and adding airports to a blank map. The
DataMap class takes as input a Map object and a DataHandler object (e.g.,
UPPData object) and creates a standard plot with shaded fields, contours, wind
barbs, and descriptive annotation.
'''

from functools import lru_cache

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

# REGIONS is a dict with predefined regions specifying the corners of the grid to be plotted.
#     Order: [lower left lat, upper right lat, lower left lon, upper right lon]

REGIONS = {
    'hrrr': [21.1381, 47.8422, 360-122.72, 360-60.9172],
    'fv3': [22.4140, 47.1024, -122.2141, -62.6567],
}


class Map():

    '''
    Class includes utilities needed to create a Basemap object, add airport
    locations, and draw the blank map.

        Required arguments:

          airport_fn    full path to airport file
          ax            figure axis

        Keyword arguments:

          region        string corresponding to REGIONS dict key
          map_proj      dict describing the map projection to use.
                        The only options currently are for lcc settings in
                        _get_basemap()
          corners       list of values lat and lon of lower left (ll) and upper
                        right(ur) corners:
                             ll_lat, ur_lat, ll_lon, ur_lon
    '''

    def __init__(self, airport_fn, ax, **kwargs):

        self.ax = ax
        self.corners = kwargs.get('corners', REGIONS[kwargs.get('region', 'hrrr')])
        self.m = self._get_basemap(**kwargs.get('map_proj', {}))
        self.airports = self.load_airports(airport_fn)

    def boundaries(self):

        ''' Draws map boundaries - coasts, states, countries. '''

        self.m.drawcoastlines()
        self.m.drawstates()
        self.m.drawcountries()

    def draw(self):

        ''' Draw a map with political boundaries and airports only. '''

        self.boundaries()
        self.draw_airports()

    def draw_airports(self):

        ''' Plot each of the airport locations on the map. '''

        lats = self.airports[:, 0]
        lons = 360 + self.airports[:, 1] # Convert to positive longitude
        x, y = self.m(lons, lats)
        self.m.plot(x, y, 'ko',
                    ax=self.ax,
                    color='w',
                    fillstyle='full',
                    markeredgecolor='k',
                    markeredgewidth=0.5,
                    markersize=4,
                    )

    def _get_basemap(self, center_lat=39.0, center_lon=262.5, lat_1=38.5, lat_2=38.5):

        ''' Wrapper around basemap creation '''

        return Basemap(ax=self.ax,
                       lat_0=center_lat,
                       lat_1=lat_1,
                       lat_2=lat_2,
                       llcrnrlat=self.corners[0],
                       llcrnrlon=self.corners[2],
                       lon_0=center_lon,
                       projection='lcc',
                       resolution='l',
                       urcrnrlat=self.corners[1],
                       urcrnrlon=self.corners[3],
                       )

    @staticmethod
    def load_airports(fn):

        ''' Load lat, lon pairs from a text file, return a list of lists. '''

        with open(fn, 'r') as f:
            data = f.readlines()
        return np.array([l.strip().split(',') for l in data], dtype=float)


class DataMap():

    '''
    Class that combines the input data and the chosen map to plot both together.

    Input:

        field             datahandler data object for data field to shade
        contour_field     dandhandler data object for data field to contour
        map               maps object

    '''

    def __init__(self, field, map_, contour_field=None):
        self.field = field
        self.contour_field = contour_field
        self.map = map_

    def _colorbar(self, cc, ax):

        ''' Internal method that plots the color bar for a contourf field. '''

        ticks = np.arange(np.amin(self.field.clevs),
                          np.amax(self.field.clevs+1), self.field.ticks)
        cbar = plt.colorbar(cc,
                            ax=ax,
                            orientation='horizontal',
                            pad=0.02,
                            shrink=1.0,
                            ticks=ticks,
                            )
        cbar.ax.set_xticklabels(ticks, fontsize=18)

    def draw(self, show=False):

        ''' Main method for creating the plot. Set show=True to display the
        figure from the command line. '''

        ax = self.map.ax

        # Draw a map and add the shaded field
        self.map.draw()
        cf = self._draw_field(field=self.field, func=self.map.m.contourf, \
                              colors=self.field.colors, extend='both', ax=ax)
        self._colorbar(cc=cf, ax=ax)

        # Contour a secondary field, if requested
        if self.contour_field is not None:
            cc = self._draw_field(field=self.contour_field, func=self.map.m.contour, ax=ax, \
                       colors=self.field.vspec.get('contour_colors', self.contour_field.colors))
            clab = plt.clabel(cc, self.contour_field.clevs[::4], fontsize=18, inline=1, fmt='%4.0f')
            _ = [txt.set_bbox(dict(facecolor='k', edgecolor='none', pad=0)) for txt in clab]

        # Add wind barbs, if requested
        add_wind = self.field.vspec.get('wind', False)
        if add_wind:
            self._wind_barbs(add_wind)

        # Finish with the title
        self._title()

        # Create a pop-up to display the figure, if show=True
        if show:
            plt.tight_layout()
            plt.show()

    def _draw_field(self, ax, field, func, **kwargs):

        '''
        Internal implementation that calls a matplotlib function.

        Input args:
            ax:      Figure axis
            field:   Field to be plotted
            func:    Matplotlib function to be called.

        Keyword args:
            Can be any of the keyword args accepted by original func in
            matplotlib.

        Return:
            The return from the function called.
        '''

        x, y = self._xy_mesh(field)
        return func(x, y, field.values(),
                    field.clevs,
                    ax=ax,
                    **kwargs,
                    )

    def _title(self):

        ''' Creates the standard annotation for a plot. '''

        f = self.field
        atime = f.date_to_str(f.anl_dt)
        vtime = f.date_to_str(f.valid_dt)

        # Create a descriptor string for the contoured field, if one exists
        contoured = ''
        if self.contour_field is not None:
            cf = self.contour_field
            contoured = f'{cf.field.long_name} ({cf.units}, contoured)'

        # Analysis time (top) and forecast hour (bottom) on the left
        plt.title(f"Analysis: {atime}\nFcst Hr: : {f.fhr}", loc='left', fontsize=16)

        # Atmospheric level and unit in the high center
        level, lev_unit = f.numeric_level
        plt.title(f"{level} {lev_unit}", position=(0.5, 1.04), fontsize=18)

        # Two lines for shaded data (top), and contoured data (bottom)
        plt.title(f"{f.field.long_name} ({f.units}, shaded)\n {contoured}",
                  loc='right',
                  fontsize=16,
                  )

        # X label shows forecast valid time.
        plt.xlabel(f"Valid time: {vtime}", fontsize=18, labelpad=100)

    def _wind_barbs(self, level):

        ''' Draws the wind barbs. '''

        u, v = self.field.wind(level)

        # Set the stride of the barbs to be plotted with a masked array.
        mask = np.ones_like(u)
        mask[::30, ::35] = 0

        mu, mv = [np.ma.masked_array(c, mask=mask) for c in [u, v]]
        x, y = self._xy_mesh(self.field)
        self.map.m.barbs(x, y, mu, mv,
                         barbcolor='k',
                         flagcolor='k',
                         length=6,
                         linewidth=0.3,
                         sizes={'spacing': 0.25},
                         )

    @lru_cache()
    def _xy_mesh(self, field):

        ''' Helper function to create mesh for various plot. '''

        lat, lon = field.latlons()
        return self.map.m(360+lon, lat)
