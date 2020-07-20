# pylint: disable=too-many-locals,invalid-name
''' Driver for creating all the graphics needed for a specific input dataset. '''

# pylint: disable=wrong-import-position, wrong-import-order
import matplotlib as mpl
mpl.use('Agg')
# pylint: enable=wrong-import-position, wrong-import-order

import argparse
import os

import matplotlib.pyplot as plt
import yaml

from adb_graphics.datahandler import grib
import adb_graphics.errors as errors
from adb_graphics.figures import maps
import adb_graphics.utils as utils


AIRPORTS = 'static/Airports_locs.txt'


def main(cla):

    '''
    Loads a set of images to be plotted, then creates them from grib input files.
    '''

    # Load image list
    with open(cla.image_list, 'r') as fn:
        images = yaml.load(fn, Loader=yaml.Loader)[cla.image_set]

    # Locate input grib file
    str_start_time = utils.from_datetime(cla.start_time)
    grib_file = images['input_files']['hrrr_prs'].format(FCST_TIME=cla.fcst_hour)
    grib_path = os.path.join(cla.data_root, str_start_time, grib_file)


    if not os.path.exists(grib_path):
        raise IOError(f"{grib_path} not found!")

    # Create working directory
    workdir = os.path.join(cla.output_path, str_start_time + f"{cla.fcst_hour:02d}")
    os.makedirs(workdir, exist_ok=True)

    # Load default specs configuration
    spec_file = 'adb_graphics/default_specs.yml'
    with open(spec_file, 'r') as fn:
        specs = yaml.load(fn, Loader=yaml.Loader)

    print((('-' * 120)+'\n') * 2)
    print(f'Creating graphics for input file: {grib_path}')
    print(f'Output graphics directory: {workdir}')
    print(f'Graphics specification follows: {spec_file}')
    print()
    print((('-' * 120)+'\n') * 2)

    tile = ''
    # Create plot for each figure in image list
    for variable, levels in images.get('variables').items():
        for level in levels:

            # Load the spec for the current variable
            spec = specs.get(variable, {}).get(level)

            if not spec:
                msg = f'graphics: {variable} {level}'
                raise errors.NoGraphicsDefinitionForVariable(msg)

            field = grib.fieldData(
                filename=grib_path,
                level=level,
                short_name=variable,
                )

            contour_field = spec.get('contour')
            if contour_field is not None:
                contour_field = grib.fieldData(
                    filename=grib_path,
                    level=level,
                    short_name=contour_field,
                    )

            _, ax = plt.subplots(1, 1, figsize=(12, 12))

            m = maps.Map(
                airport_fn=AIRPORTS,
                ax=ax,
                corners=field.corners,
                )

            dm = maps.DataMap(
                field=field,
                contour_field=contour_field,
                map_=m,
                )

            dm.draw(show=True)

            png_suffix = level if level != 'ua' else ''
            png_file = f'{variable}{"_" + tile}{png_suffix}'
            png_path = os.path.join(workdir, png_file)

            print('*' * 120)
            print(f"Creating image file: {png_path}")
            print('*' * 120)

            # pylint: disable=duplicate-code
            plt.savefig(
                png_path,
                bbox_inches='tight',
                dpi='figure',
                format='png',
                orientation='landscape',
                )

def webname(prefix, tile='', suffix=''):
    ''' Return the filename expected for the web graphic. '''
    return f"{prefix}{'_' + tile}{'_' + suffix}"

def parse_args():

    ''' Set up argparse command line arguments, and return the Namespace
    containing the settings. '''

    parser = argparse.ArgumentParser(description='Script to drive the creation of graphics.')

    parser.add_argument('-d', '--data_root',
                        help='Cycle-independant data directory location.',
                        required=True,
                        )
    parser.add_argument('-f', '--fcst_hour',
                        help='Forecast hour',
                        required=True,
                        type=int,
                        )
    parser.add_argument('--image_list',
                        help='Path to YAML config file specifying which graphics to create.',
                        required=True,
                        )
    parser.add_argument('-m', '--image_set',
                        choices=['hourly'],
                        help='Name of top level key in image_list',
                        required=True,
                        )
    parser.add_argument('-o', '--output_path',
                        help='Directory location desired for the output graphics files.',
                        required=True,
                        )
    parser.add_argument('-s', '--start_time',
                        help='Start time in YYYYMMDDHH format',
                        required=True,
                        type=utils.to_datetime,
                        )
    parser.add_argument('--subh_freq',
                        default=60,
                        help='Sub-hourly frequency in minutes.',
                        )
    parser.add_argument('-t', '--num_threads',
                        default=1,
                        help='Number of threads to use.',
                        )
    return parser.parse_args()

if __name__ == '__main__':
    CLARGS = parse_args()
    main(CLARGS)
