# pylint: disable=invalid-name,too-few-public-methods

'''
Module contains classes relevant to plotting maps. The Map class handles all the
functionality related to a Basemap, and adding airports to a blank map. The
DataMap class takes as input a Map object and a DataHandler object (e.g.,
UPPData object) and creates a standard plot with shaded fields, contours, wind
barbs, and descriptive annotation.
'''

import copy
from math import isnan
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.offsetbox as mpob
import matplotlib.patches as mpatches
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import shiftgrid
import numpy as np

# FULL_TILES is a list of strings that includes the labels GSL attaches to some of
# the wgrib2 cutouts used for larger domains like RAP, RRFS NA, and global.
FULL_TILES = [
    "AK",
    "CONUS",
    "conus",
    "full",
    "hrrr",
    "hrrrak",
    "NHemi",
    ]
# TILE_DEFS is a dict of dicts with predefined tiles specifying the corners of the grid
#     to be plotted, and the stride and length of the wind barbs.
# Order for corners: [lower left lat, upper right lat, lower left lon, upper right lon]

TILE_DEFS = {
    'NC': {'corners': [36, 51, -109, -85], 'stride': 10, 'length': 4},
    'NE': {'corners': [36, 48, -91, -62], 'stride': 10, 'length': 4},
    'NW': {'corners': [35, 52, -126, -102], 'stride': 10, 'length': 4},
    'SC': {'corners': [24, 41, -107, -86], 'stride': 10, 'length': 4},
    'SE': {'corners': [22, 37, -93.5, -72], 'stride': 10, 'length': 4},
    'SW': {'corners': [24.5, 45, -122, -103], 'stride': 10, 'length': 4},
    'Africa': {'corners': [-40, 40, -40, 60], 'stride': 7, 'length': 5},
    'AKZoom': {'corners': [52, 73, -162, -132], 'stride': 4, 'length': 5},
    'AKZoom2': {'corners': [37.9, 80.8, 180, -105.7], 'stride': 8, 'length': 5},
    'AKRange': {'corners': [59.722, 65.022, -153.583, -144.289], 'stride': 4, 'length': 4},
    'Anchorage': {'corners': [58.59, 62.776, -152.749, -146.218], 'stride': 4, 'length': 4},
    'ATL': {'corners': [31.2, 35.8, -87.4, -79.8], 'stride': 4, 'length': 4},
    'Beijing': {'corners': [25, 53, 102, 133], 'stride': 3, 'length': 5},
    'CA-NV': {'corners': [30, 45, -124, -114], 'stride': 10, 'length': 4},
    'Cambodia': {'corners': [0, 24, 90, 118], 'stride': 3, 'length': 5},
    'CentralCA': {'corners': [34.5, 40.5, -124, -118], 'stride': 4, 'length': 4},
    'CHI-DET': {'corners': [39, 44, -92, -83], 'stride': 4, 'length': 4},
    'DCArea': {'corners': [36.7, 40, -81, -72], 'stride': 4, 'length': 4},
    'EastCO': {'corners': [36.5, 41.5, -108, -101.8], 'stride': 4, 'length': 4},
    'EPacific': {'corners': [0, 60, 180, 300], 'stride': 10, 'length': 5},
    'Europe': {'corners': [15, 75, -30, 75], 'stride': 10, 'length': 5},
    'Florida': {'corners': [19.2305, 29.521, -86.1119, -73.8189], 'stride': 10, 'length': 5},
    'GreatLakes': {'corners': [37, 50, -96, -70], 'stride': 10, 'length': 4},
    'HI': {'corners': [16.6, 24.6, -157.6, -157.5], 'stride': 1, 'length': 4},
    'Juneau': {'corners': [55.741, 59.629, -140.247, -129.274], 'stride': 4, 'length': 4},
    'NW-large': {'corners': [29.5787, 52.6127, -121.666, -96.5617], 'stride': 4, 'length': 4},
    'NYC-BOS': {'corners': [39, 43.5, -77, -66.5], 'stride': 4, 'length': 4},
    'PuertoRico': {'corners': [15.5257, 24.0976, -74.6703, -61.848], 'stride': 10, 'length': 5},
    'SEA-POR': {'corners': [43, 50, -125, -119], 'stride': 4, 'length': 4},
    'SouthCA': {'corners': [31, 37, -120, -114], 'stride': 4, 'length': 4},
    'SouthFL': {'corners': [24, 28.5, -84, -77], 'stride': 4, 'length': 4},
    'Taiwan': {'corners': [19, 28, 116, 126], 'stride': 1, 'length': 5},
    'VortexSE': {'corners': [30, 37, -92.5, -82], 'stride': 4, 'length': 4},
    'WAtlantic': {'corners': [-0.25, 50.25, 261.75, 330.25], 'stride': 5, 'length': 5},
    'WPacific': {'corners': [-40, 50, 90, 240], 'stride': 10, 'length': 5},
}


class Map():
    # pylint: disable=too-many-instance-attributes

    '''
    Class includes utilities needed to create a Basemap object, add airport
    locations, and draw the blank map.

        Required arguments:

          airport_fn    full path to airport file
          ax            figure axis

        Keyword arguments:

          map_proj      dict describing the map projection to use.
                        The only options currently are for lcc settings in
                        _get_basemap()
          corners       list of values lat and lon of lower left (ll) and upper
                        right(ur) corners:
                             ll_lat, ur_lat, ll_lon, ur_lon
          model         model designation used to trigger higher resolution maps if needed
                        also used to turn off plotting of airports on global maps
          plot_airports bool to allow airport plotting to be turned off for
                        certain plots, default is True
          tile          a string corresponding to a pre-defined tile in the
                        TILE_DEFS dictionary
    '''

    def __init__(self, airport_fn, ax, **kwargs):

        self.ax = ax
        self.grid_info = kwargs.get('grid_info', {})
        self.model = kwargs.get('model')
        self.plot_airports = kwargs.get('plot_airports', True)
        self.tile = kwargs.get('tile', 'full')
        self.airports = self.load_airports(airport_fn)

        if self.tile in FULL_TILES:
            self.corners = self.grid_info.pop('corners')
        else:
            self.corners = self.get_corners()
            self.grid_info.pop('corners')

        # Some of Hawaii's smaller islands and islands in the Caribbean don't
        # show up with a larger threshold.
        area_thresh = 1000
        if self.tile in ['HI', 'Florida', 'PuertoRico'] or self.model in ['hrrrhi', 'hrrrcar']:
            area_thresh = 100

        self.m = self._get_basemap(area_thresh=area_thresh, **self.grid_info)

    def boundaries(self):

        ''' Draws map boundaries - coasts, states, countries. '''

        try:
            self.m.drawcoastlines(linewidth=0.5)
        except ValueError:
            self.m.drawcounties(color='gray',
                                linewidth=0.4,
                                zorder=2,
                                )
        else:
            if self.model not in ['global'] and self.tile not in FULL_TILES:
                self.m.drawcounties(antialiased=False,
                                    color='gray',
                                    linewidth=0.1,
                                    zorder=2,
                                    )
        self.m.drawstates()
        self.m.drawcountries()

    def draw(self):

        ''' Draw a map with political boundaries and airports only. '''

        self.boundaries()
        if self.plot_airports and 'global' not in self.model: # airports are too dense in global
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

        del x
        del y

    def _get_basemap(self, **get_basemap_kwargs):

        ''' Wrapper around basemap creation '''

        basemap_args = dict(
            ax=self.ax,
            resolution='i',
            )
        corners = self.corners
        if corners is not None:
            basemap_args.update(dict(
                llcrnrlat=corners[0],
                llcrnrlon=corners[2],
                urcrnrlat=corners[1],
                urcrnrlon=corners[3],
                ))

        basemap_args.update(get_basemap_kwargs)

        return Basemap(**basemap_args)

    def get_corners(self):

        '''
        Gather the corners for a specific tile. Corners are supplied in the
        following format:

        lat and lon of lower left (ll) and upper right(ur) corners:
             ll_lat, ur_lat, ll_lon, ur_lon
        '''

        return TILE_DEFS[self.tile]["corners"]

    @staticmethod
    def load_airports(fn):

        ''' Load lat, lon pairs from a text file, return a list of lists. '''

        with open(fn, 'r') as f:
            data = f.readlines()
        return np.array([l.strip().split(',') for l in data], dtype=float)


class DataMap():
    #pylint: disable=too-many-arguments

    '''
    Class that combines the input data and the chosen map to plot both together.

    Input:

        field             datahandler data object for data field to shade
        contour_fields    list of datahandler object fields to contour
        hatch_fields      list of datahandler object fields to hatch over shaded
                          fields
        map               maps object

    '''

    #pylint: disable=unused-argument
    def __init__(self, map_fields, map_, model_name=None, **kwargs):

        self.field = map_fields.main_field
        self.contour_fields = map_fields.contours
        self.hatch_fields = map_fields.hatches
        self.map = map_
        self.model_name = model_name
        self.plot_scatter = map_fields.fields_spec.get('plot_scatter', False)

    @staticmethod
    def add_logo(ax):

        ''' Puts the NOAA logo at the bottom left of the matplotlib axes. '''

        logo = mpimg.imread('static/noaa-logo-50x50.png')

        imagebox = mpob.OffsetImage(logo)
        ab = mpob.AnnotationBbox(
            imagebox,
            (0, 0),
            box_alignment=(-0.2, -0.2),
            frameon=False,
            xycoords='axes points',
            )

        ax.add_artist(ab)


    def _colorbar(self, cc, ax):

        ''' Internal method that plots the color bar for a contourf field.
            If ticks is set to zero, use a user-defined list of clevs from default_specs
            If ticks is less than zero, use abs(ticks) as the step for labeling clevs '''

        if self.field.ticks > 0:
            ticks = np.arange(np.amin(self.field.clevs),
                              np.amax(self.field.clevs+1), self.field.ticks)
        elif self.field.ticks == 0:
            ticks = self.field.clevs
        else:
            ticks = self.field.clevs[0:len(self.field.clevs):-self.field.ticks]
        ticks = np.around(ticks, 4)

        cbar = plt.colorbar(cc,
                            ax=ax,
                            orientation='horizontal',
                            pad=0.02,
                            shrink=1.0,
                            ticks=ticks,
                            )

        if self.field.short_name == 'flru':
            ticks = [label.rjust(30) for label in ['VFR', 'MVFR', 'IFR', 'LIFR']]

        cbar.ax.set_xticklabels(ticks, fontsize=12)

    def draw(self, show=False):

        ''' Main method for creating the plot. Set show=True to display the
        figure from the command line. '''

        cf = self._draw_panel()

        # Draw colorbar
        self._colorbar(ax=self.map.ax, cc=cf)

        # Finish with the title
        self._title()

        # Create a pop-up to display the figure, if show=True
        if show:
            plt.tight_layout()
            plt.show()

        self.add_logo(self.map.ax)

    def _draw_panel(self, wind_barbs=True): # pylint: disable=too-many-locals, too-many-branches

        ax = self.map.ax

        # Draw a map and add the shaded field
        self.map.draw()
        cf = self._draw_field(ax=ax,
                              colors=self.field.colors,
                              extend='both',
                              field=self.field,
                              func=self.map.m.contourf,
                              levels=self.field.clevs,
                              )

        not_labeled = [self.field.short_name]
        if self.hatch_fields:
            not_labeled.extend([h.short_name for h in self.hatch_fields])

        # Contour secondary fields, if requested
        if self.contour_fields:
            self._draw_contours(ax, not_labeled)

        # Add hatched fields, if requested
        if self.hatch_fields:
            self._draw_hatches(ax)

        # Add wind barbs, if requested
        add_wind = self.field.vspec.get('wind', False)
        if add_wind and wind_barbs:
            self._wind_barbs(add_wind)

        # Add field values at airports
        annotate = self.field.vspec.get('annotate', False)
        if annotate and 'global' not in self.map.model: # too dense in global
            self._draw_field_values(ax)

        # Add scatter plot, if requested
        if self.plot_scatter:
            self._draw_scatter(ax)

        return cf

    def _draw_contours(self, ax, not_labeled):

        ''' Draw the contour fields requested. '''

        for contour_field in self.contour_fields:
            levels = contour_field.contour_kwargs.pop('levels',
                                                      contour_field.clevs)

            cc = self._draw_field(ax=ax,
                                  field=contour_field,
                                  func=self.map.m.contour,
                                  levels=levels,
                                  **contour_field.contour_kwargs,
                                  )
            if contour_field.short_name not in not_labeled:
                try:
                    clab = plt.clabel(cc, levels[::4],
                                      colors='w',
                                      fmt='%1.0f',
                                      fontsize=10,
                                      inline=1,
                                      )
                    # Set the background color for the line labels to black
                    _ = [txt.set_bbox(dict(color='k')) for txt in clab]

                except ValueError:
                    print(f'Cannot add contour labels to map for {self.field.short_name} \
                            {self.field.level}')

    def _draw_scatter(self, ax):

        ''' Plot dots at locations on the map that meet a threshold. '''

        field = self.field
        levels = self.field.clevs
        colors = self.field.colors
        vals = self.field.values()

        value_to_color = np.full_like(vals, colors[0], dtype='object')
        num_levels = len(levels)
        for i in range(num_levels):
            if i != num_levels - 1:
                value_to_color = np.where((vals > levels[i]) & \
                    (vals <= levels[i+1]), colors[i+1], value_to_color)
            else:
                value_to_color = np.where(vals > levels[i], colors[i+1], value_to_color)

        vtc1d = np.ravel(value_to_color)
        self._draw_field(ax=ax,
                         field=field,
                         alpha=1.0,
                         c=vtc1d,
                         func=self.map.m.scatter,
                         **field.contour_kwargs,
                         )

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
        vals = field.data

        # Scatter plot dots are sized by value. Doing this here alters the size
        # without altering the colors.
        if self.plot_scatter:
            vals = np.full_like(vals, np.log10(vals) * 20, dtype='float')

        # For global lat-lon models, make 2D arrays for x and y
        # Shift the map and data if needed
        if self.map.model in ['global']:
            tile = self.map.tile
            if tile in ['Africa', 'Europe']:
                vals, x = shiftgrid(180., vals, x, start=False)
            y, x = np.meshgrid(y, x, sparse=False, indexing='ij')

        ret = func(x, y, vals,
                   ax=ax,
                   **kwargs,
                   )

        del x
        del y
        try:
            vals.close()
        except AttributeError:
            del vals
            print(f'CLOSE ERROR: {field.short_name} {field.level}')
        return ret

    def _draw_field_values(self, ax):

        ''' Add the text value of the field at airport locations. '''
        annotate_decimal = self.field.vspec.get('annotate_decimal', 0)
        lats = self.map.airports[:, 0]
        lons = 360 + self.map.airports[:, 1]
        x, y = self.map.m(lons, lats)
        data_values = self.field.data
        crnrs = copy.copy(self.map.corners)
        if crnrs[2] < 0:
            crnrs[2] = 360 + crnrs[2]
        if crnrs[3] < 0:
            crnrs[3] = 360 + crnrs[3]
        for i, lat in enumerate(lats):
            if crnrs[1] > lat > crnrs[0] and \
               crnrs[3] > lons[i] > crnrs[2]:
                xgrid, ygrid = self.field.get_xypoint(lat, lons[i])
                data_value = data_values[xgrid, ygrid].values.item()
                if xgrid > 0 and ygrid > 0:
                    if (not isnan(data_value)) and (data_value != 0.):
                        ax.annotate(f"{data_value:.{annotate_decimal}f}", \
                                    xy=(x[i], y[i]), fontsize=10)
        data_values.close()

    def _draw_hatches(self, ax):

        ''' Draw the hatched regions requested. '''

        # Levels should be included in the settings dict here since they don't
        # correspond to a full field of contours.
        handles = []
        for field in self.hatch_fields:
            colors = field.contour_kwargs.get('colors', 'k')
            hatches = field.contour_kwargs.get('hatches', '----')
            labels = field.contour_kwargs.get('labels', 'XXXX')
            linewidths = field.contour_kwargs.get('linewidths', 0.1)
            handles.append(mpatches.Patch(edgecolor=colors[-1], facecolor='lightgrey', \
                           label=labels, hatch=hatches[-1]))

            cf = self._draw_field(ax=ax,
                                  extend='both',
                                  field=field,
                                  func=self.map.m.contourf,
                                  **field.contour_kwargs,
                                  )

            # For each level, we set the color of its hatch
            for collection in cf.collections:
                collection.set_edgecolor(colors)
                collection.set_facecolor(['None'])
                collection.set_linewidth(linewidths)

        # Create legend for precip type field
        if self.field.short_name == 'ptyp':
            plt.legend(handles=handles, loc=[0.25, 0.03])

    def _set_overlay_string(self):

        ''' Creates the main title of the plot with select hatched and
        contoured fields defined. '''

        f = self.field

        # Create a descriptor string for the first hatched field, if one exists
        contoured = []
        contoured_units = []
        not_labeled = [f.short_name]
        if self.hatch_fields:
            cf = self.hatch_fields[0]
            not_labeled.extend([h.short_name for h in self.hatch_fields])
            if not any(list(set(cf.short_name).intersection(['pres']))):
                title = cf.vspec.get('title', cf.field.long_name)
                contoured.append(f'{title} ({cf.units}, hatched)')

        # Add descriptor string for the important contoured fields
        if self.contour_fields:
            for cf in self.contour_fields:
                if cf.short_name not in not_labeled:
                    title = cf.vspec.get('title', cf.field.long_name)
                    title = title.replace("Geopotential", "Geop.")
                    contoured.append(f'{title}')
                    contoured_units.append(f'{cf.units}')

        contoured = '\n'.join(contoured)  # Make 'contoured' a string with linefeeds
        if contoured_units:
            contoured = f"{contoured} ({', '.join(contoured_units)}, contoured)"

        return contoured

    def _title(self):

        ''' Draw the title for a map. '''

        f = self.field
        atime = f.date_to_str(f.anl_dt)
        vtime = f.date_to_str(f.valid_dt)

        # Analysis time (top) and forecast hour with Valid Time (bottom) on the left
        plt.title(f"{self.model_name}: {atime}\nFcst Hr: {f.fhr}, Valid Time {vtime}",
                  alpha=None,
                  fontsize=14,
                  loc='left',
                  )

        level, lev_unit = f.numeric_level(index_match=False)
        if f.vspec.get('print_units', True):
            units = f'({f.units}, shaded)'
        else:
            units = f''

        # Title or Atmospheric level and unit in the high center
        if f.vspec.get('title'):
            title = f"{f.vspec.get('title')} {units}"
        else:
            level = level if not isinstance(level, list) else level[0]
            title = f'{level} {lev_unit} {f.field.long_name} {units}'
        plt.title(f"{title}", position=(0.5, 1.08), fontsize=18)

        # Two lines for hatched data (top), and contoured data (bottom) on the right
        contoured = self._set_overlay_string()
        plt.title(f"{contoured}",
                  loc='right',
                  fontsize=14,
                  )

    def _wind_barbs(self, level):

        ''' Draws the wind barbs. A decent stride can be found if you divide the
            number of grid points on the shorter side by 35. Subdomains are defined
            by lat,lon so the stride is set in the TILE_DEFS. For the globalCONUS
            subdomains, further dividing by 2.5 works well. '''

        u, v = self.field.wind(level)

        tile = self.map.tile

        full_tile = tile in FULL_TILES

        # Set the stride and size of the barbs to be plotted with a masked array.
        if full_tile:
            if u.shape[0] < u.shape[1]:
                stride = int(round(u.shape[0] / 35))
            else:
                stride = int(round(u.shape[1] / 35))
            length = 5
        else:
            stride = TILE_DEFS[tile]["stride"]
            length = TILE_DEFS[tile]["length"]
            if self.map.model == 'globalCONUS':
                stride = int(round(stride / 2.5))
                length = 5

        mask = np.ones_like(u)
        mask[::stride, ::stride] = 0

        x, y = self._xy_mesh(self.field)

        # For global lat-lon models, make 2D arrays for x and y
        # Shift the map and data if needed
        if self.map.m.projection == 'cyl':
            if tile in ['Africa', 'Europe']:
                savex = x
                u, x = shiftgrid(180., u, x, start=False)
                v, savex = shiftgrid(180., v, savex, start=False)
            y, x = np.meshgrid(y, x, sparse=False, indexing='ij')
        mu, mv = [np.ma.masked_array(c, mask=mask) for c in [u, v]]

        self.map.m.barbs(x, y, mu, mv,
                         barbcolor='k',
                         flagcolor='k',
                         length=length,
                         linewidth=0.2,
                         sizes={'spacing': 0.25},
                         )

    def _xy_mesh(self, field):

        ''' Helper function to create mesh for various plot. '''

        lat, lon = field.latlons()
        if self.map.model == 'obs':
            lat, lon = np.meshgrid(lat, lon, sparse=False, indexing='ij')

        adjust = 360 if np.any(lon < 0) else 0
        return self.map.m(adjust + lon, lat)

class DiffMap(DataMap):
    '''
    Extends DataMap for handling difference plots, which need different titles,
    and will not plot overlays and such.
    '''

    def _colorbar(self, cc, ax):

        ''' Set the colorbar for a difference field. '''

        plt.colorbar(
            cc,
            ax=ax,
            orientation='horizontal',
            pad=0.02,
            shrink=1.0,
            )

    def _draw_panel(self, wind_barbs=False):

        ''' Draw a map of the difference field. '''

        ax = self.map.ax

        # Draw a map and add the shaded field
        self.map.draw()

        # The number of levels (nlev) here, should be the same number as is used
        # in the linspace call in self._eq_contours. 21 seems reasonable, but is
        # arbitrary.
        colors = self.field.centered_diff(cmap='Spectral_r', nlev=21)
        cf = self._draw_field(ax=ax,
                              colors=colors,
                              extend='both',
                              field=self.field,
                              func=self.map.m.contourf,
                              levels=self._eq_contours(),
                              )
        return cf

    def _eq_contours(self):
        ''' Center the contours based on the data min/max '''

        minval = np.amin(self.field.data)
        maxval = np.amax(self.field.data)
        if minval == maxval == 0:
            return np.array([-1, 0, 1])
        maxval = max(abs(minval), abs(maxval))
        return np.linspace(-maxval, maxval, 21)

    def _title(self):
        ''' Draw the title for a map. '''

        f = self.field
        atime = f.date_to_str(f.anl_dt)
        vtime = f.date_to_str(f.valid_dt)

        # Analysis time (top) and forecast hour with Valid Time (bottom) on the left
        plt.title(f"{self.model_name}: {atime}\nFcst Hr: {f.fhr}, Valid Time {vtime}",
                  alpha=None,
                  fontsize=14,
                  loc='left',
                  )

        level, lev_unit = f.numeric_level(index_match=False)
        if f.vspec.get('print_units', True):
            units = f'({f.units}, shaded)'
        else:
            units = f''

        # Title or Atmospheric level and unit in the high center
        if f.vspec.get('title'):
            title = f"Diff: {f.vspec.get('title')} {units}"
        else:
            level = level if not isinstance(level, list) else level[0]
            title = f'Diff: {level} {lev_unit} {f.field.long_name} {units}'
        plt.title(f"{title}", position=(0.5, 1.08), fontsize=18)



class MultiPanelDataMap(DataMap):
    '''
    Class that extends a DataMap for handling multiple panels.

    Keyword arguments:
        last_panel        flag for multipanel plots to designate last panel drawn
    '''

    def __init__(self, map_fields, map_, member, model_name=None, **kwargs):

        super().__init__(map_fields, map_, model_name=model_name)

        self.last_panel = kwargs.get('last_panel', False)
        self.member = str(member)

    def draw(self, show=False):

        ''' Main method for creating the plot. Set show=True to display the
        figure from the command line. '''

        cf = self._draw_panel(wind_barbs=False)

        self._label_member()

        # Finish with the colorbar on the last panel only
        # Plot it on the full figure scale.
        if self.last_panel:
            cax = plt.axes([0.0, 0.0, 1.0, 0.2])
            self._colorbar(ax=cax, cc=cf)
            cax.axis('off')

        # Create a pop-up to display the figure, if show=True
        if show:
            plt.tight_layout()
            plt.show()

        return cf

    def _label_member(self):

        ''' Add the member label to the top left of the plot '''

        ax = self.map.ax
        ax.text(
            0.05, 0.90,
            self.member,
            fontsize=18,
            fontweight='bold',
            backgroundcolor="white",
            transform=ax.transAxes,
            )

    def title(self):

        ''' Draw the title for a map. '''

        f = self.field
        atime = f.date_to_str(f.anl_dt)
        vtime = f.date_to_str(f.valid_dt)
        ax = self.map.ax

        # Analysis time (top) and forecast hour with Valid Time (bottom) on the left
        ax.text(0.0, 0.5,
                f"{self.model_name}: {atime}\nFcst Hr: {f.fhr}, Valid Time {vtime}",
                alpha=None,
                fontsize=14,
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes,
                )

        level, lev_unit = f.numeric_level(index_match=False)
        if f.vspec.get('print_units', True):
            units = f'({f.units}, shaded)'
        else:
            units = f''

        # Title or Atmospheric level and unit in the high center
        if f.vspec.get('title'):
            title = f"{f.vspec.get('title')} {units}"
        else:
            level = level if not isinstance(level, list) else level[0]
            title = f'{level} {lev_unit} {f.field.long_name} {units}'
        ax.text(0, 0.7,
                f"{title}",
                horizontalalignment='left',
                verticalalignment='top',
                fontsize=16,
                transform=ax.transAxes,
                )

        # Two lines for hatched data (top), and contoured data (bottom) on the right
        contoured = self._set_overlay_string()
        ax.text(0, 0.6,
                f"{contoured}",
                horizontalalignment='left',
                verticalalignment='top',
                fontsize=14,
                transform=ax.transAxes,
                )


class MapFields():
    ''' Class that packages all the field objects need for producing
    desired map content, i.e. an object that contains all filled
    contours, hatched spaces, and overlayed contours needed for a full
    product. '''

    def __init__(self, main_field, fields_spec=None, map_type=None,
                 **kwargs):

        self.main_field = main_field
        self.fields_spec = fields_spec if fields_spec is not None else {}
        self.map_type = map_type
        self.model = kwargs.get('model')
        self.tile = kwargs.get('tile', 'full')

    @property
    def contours(self):
        ''' Return the list of contour fieldData objects'''

        # We won't plot contours on multipanel plots, or full global
        # plots.
        if self.map_type == 'enspanel':
            return []

        if self.model in ['global'] and self.tile in ['full']:
            return []

        return self._overlay_fields('contours')

    @property
    def hatches(self):
        ''' Return the list of hatch fieldData objects'''

        return self._overlay_fields('hatches')

    def _overlay_fields(self, spec_sect):

        ''' Generate a list of fieldData objects for the specified type
        of overlay -- hatches or contours '''
        overlays = self.fields_spec.get(spec_sect)
        overlay_fields = []
        if overlays is not None:
            for overlay, overlay_kwargs in overlays.items():
                if '_' in overlay:
                    var, lev = overlay.split('_')
                else:
                    var, lev = overlay, self.main_field.level

                # Make a copy of the main object, and change the
                # attributes to match the overlay field
                overlay_obj = copy.deepcopy(self.main_field)
                overlay_obj.contour_kwargs = overlay_kwargs
                overlay_obj.short_name = var
                overlay_obj.level = lev

                overlay_fields.append(overlay_obj)
        return overlay_fields
