#pylint: disable=invalid-name

''' Example script for creating a figure with the adb_graphics package. '''

import matplotlib.pyplot as plt

import adb_graphics.datahandler.grib as grib
import adb_graphics.figures.maps as maps

filename = '../data/wrfnat_hrconus_12.grib2'
airports = 'static/Airports_locs.txt'

var = grib.fieldData(filename=filename,
                     level='2ds',
                     short_name='temp',
                     )
contour_var = grib.fieldData(filename=filename,
                             level='500mb',
                             short_name='gh',
                             )

#contour_var = None

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
m = maps.Map(airports, ax, corners=var.corners, region='fv3')
dm = maps.DataMap(field=var, contour_field=contour_var, map_=m)
dm.draw(show=True)
