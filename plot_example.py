import matplotlib.pyplot as plt

import adb_graphics.datahandler.grib as grib
import adb_graphics.figures.maps as maps

filename = 'HRRR.t12z.bgdawp06.tm12'
airports = 'static/Airports_locs.txt'

temp = grib.UPPData(filename=filename, short_name='t', level=500, lev_type='isobaricInhPa', season='warm')
height = grib.UPPData(filename=filename, short_name='gh', level=500, lev_type='isobaricInhPa', season='warm')

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
m = maps.Map(airports, ax, corners=temp.corners, region='fv3')
dm = maps.DataMap(field=temp, contour_field=height, map_=m)
dm.draw(show=True)
