# pylint: disable=invalid-name,
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

# REGIONS is a dict with predefined regions specifying the corners of the grid to be plotted.
#     Order: [lower left lat, upper right lat, lower left lon, upper right lon]

REGIONS = {
    'hrrr': [21.1381, 47.8422, 360-122.72, 360-60.9172],
    'fv3': [22.4140, 47.1024, -122.2141,-62.6567],
}

class Map():

    '''
    Class includes utilities needed to create a Basemap object, add airport
    locations, and draw the blank map.

        Required arguments:

          airport_fn    full path to airport file
          ax            figure axis

        Optional arguments:

          region        string corresponding to REGIONS dict key
          map_proj      dict describing the map projection to use.
                        The only options currently are for lcc settings in
                        _get_basemap()
          corners       list of values lat and lon of lower left (ll) and upper
                        right(ur) corners:
                             ll_lat, ur_lat, ll_lon, ur_lon
    '''

    def __init__(self, airport_fn, ax, region='hrrr', **kwargs):

        self.ax = ax
        self.corners = kwargs.get('corners', REGIONS[region])
        self.m = self._get_basemap(**kwargs.get('map_proj', {}))
        self.airports = self.load_airports(airport_fn)

    @property
    def boundaries(self):
        self.m.drawcoastlines()
        self.m.drawstates()
        self.m.drawcountries()

    def draw(self):
        self.boundaries # pylint: disable=pointless-statement
        self.draw_airports()

    def draw_airports(self):
        lats = self.airports[:, 0]
        lons = 360 + self.airports[:, 1] # Convert to positive longitude
        x, y = self.m(lons, lats)
        self.boundaries # pylint: disable=pointless-statement
        self.m.plot(x, y, 'ko', markersize=4, fillstyle='full', color='w',
                markeredgecolor='k', markeredgewidth=0.5, ax=self.ax)

    def _get_basemap(self, center_lat=39.0, center_lon=262.5, lat_1=38.5, lat_2=38.5):
        return Basemap(projection='lcc',
                       llcrnrlat=self.corners[0],
                       urcrnrlat=self.corners[1],
                       llcrnrlon=self.corners[2],
                       urcrnrlon=self.corners[3],
                       lat_0=center_lat,
                       lat_1=lat_1,
                       lat_2=lat_2,
                       lon_0=center_lon,
                       resolution='l',
                       ax=self.ax,
                       )

    @staticmethod
    def load_airports(fn):
        with open(fn, 'r') as f:
            data = f.readlines()
        return np.array([l.strip().split(',') for l in data], dtype=float)

class DataMap():

    def __init__(self, field, map_, contour_field=None, draw_barbs=True):
        self.field = field
        self.contour_field = contour_field
        self.map = map_
        self.draw_barbs = draw_barbs

    def _colorbar(self, cc, ax):
        ticks = range(int(min(self.field.clevs)),
                      int(max(self.field.clevs))+1, self.field.ticks)
        cbar = plt.colorbar(cc,
                            orientation='horizontal',
                            shrink=1.0,
                            ax=ax,
                            ticks=ticks,
                            pad=0.02,
                           )
        cbar.ax.set_xticklabels(ticks)

    def draw(self, show=False):
        ax = self.map.ax
        self.map.draw()
        cf = self._draw_field(field=self.field, func=self.map.m.contourf, ax=ax)
        if self.contour_field is not None:
            cc = self._draw_field(field=self.contour_field, func=self.map.m.contour,
                    ax=ax)
            clab = plt.clabel(cc, self.contour_field.clevs[::4], fontsize=18, inline=1, fmt= '%4.0f')
            [txt.set_bbox(dict(facecolor='k', edgecolor='none', pad=0)) for txt in clab]

        self._colorbar(cc=cf, ax=ax)
        if self.draw_barbs:
            self._wind_barbs()
        self._title()
        if show:
            plt.show()


    def _draw_field(self, ax, field, func, **kwargs):
        x, y = self._xy_mesh(field)
        return func(x, y, field.values,
                    field.clevs,
                    colors=field.colors,
                    ax=ax,
                    **kwargs,
                    )

    def _filled(self):
        pass

    def _title(self):
        f = self.field
        atime = f.date_to_str(f.anl_dt)
        vtime = f.date_to_str(f.valid_dt)

        contoured = ''
        if self.contour_field is not None:
            cf = self.contour_field
            contoured = f'{cf.data.name} ({cf.units}, contoured)'

        plt.title(f"Analysis: {atime}\nFcst Hr: : {f.fhr}", loc='left',
                fontsize=16)
        plt.title(f"{f.level} {f.lev_unit}", position=(0.5, 1.04), fontsize=18)
        plt.title(f"{f.data.name} ({f.units}, shaded)\n {contoured}",
                loc='right', fontsize=16)

        plt.xlabel(f"Valid time: {vtime}", fontsize=18, labelpad=100)

    def _wind_barbs(self):
        u, v = self.field.wind
        mask = np.ones_like(u.values)
        mask[::30, ::35] = 0

        mu, mv = [np.ma.masked_array(c.values, mask=mask) for c in [u, v]]
        x, y = self._xy_mesh(self.field)
        self.map.m.barbs(x, y, mu, mv, barbcolor='k', flagcolor='k', length=6,
                linewidth=0.3, sizes={'spacing': 0.25} )

    def _xy_mesh(self, field):
        lat, lon = field.data.latlons()
        return self.map.m(360+lon, lat)
