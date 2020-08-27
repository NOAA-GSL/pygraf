# pylint: disable=too-many-locals,invalid-name,too-many-statements
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

    # Create an empty zip file
    zipf = None
    if cla.zip_dir:
        os.makedirs(cla.zip_dir, exist_ok=True)
        zipf = os.path.join(cla.zip_dir, 'files.zip')

    # Load image list
    with open(cla.image_list, 'r') as fn:
        images = yaml.load(fn, Loader=yaml.Loader)[cla.image_set]

    # Locate input grib file
    grib_path = os.path.join(cla.data_root,
                             cla.file_tmpl.format(FCST_TIME=cla.fcst_hour))

    if not os.path.exists(grib_path):
        raise IOError(f"{grib_path} not found!")

    # Create working directory
    workdir = os.path.join(cla.output_path,
                           f"{utils.from_datetime(cla.start_time)}{cla.fcst_hour:02d}")
    os.makedirs(workdir, exist_ok=True)

    # Load default specs configuration
    spec_file = 'adb_graphics/default_specs.yml'
    with open(spec_file, 'r') as fn:
        specs = yaml.load(fn, Loader=yaml.Loader)

    print((('-' * 120)+'\n') * 2)
    print(f'Creating graphics for input file: {grib_path}')
    print(f'Output graphics directory: {workdir}')
    print(f'Graphics specification follows: {spec_file}\n')
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
                fhr=cla.fcst_hour,
                filename=grib_path,
                level=level,
                short_name=variable,
                )

            # Create a list of fieldData objects for each contour requested
            contours = spec.get('contours')
            contour_fields = []
            if contours is not None:
                for contour, contour_kwargs in contours.items():
                    if '_' in contour:
                        var, lev = contour.split('_')
                    else:
                        var, lev = contour, level

                    contour_fields.append(grib.fieldData(
                        fhr=cla.fcst_hour,
                        filename=grib_path,
                        level=lev,
                        contour_kwargs=contour_kwargs,
                        short_name=var,
                        ))

            # Create a list of fieldData objects for each hatched area requested
            hatches = spec.get('hatches')
            hatch_fields = []
            if hatches is not None:
                for hatch, hatch_kwargs in hatches.items():
                    var, lev = hatch.split('_')
                    hatch_fields.append(grib.fieldData(
                        fhr=cla.fcst_hour,
                        filename=grib_path,
                        level=lev,
                        contour_kwargs=hatch_kwargs,
                        short_name=var,
                        ))

            _, ax = plt.subplots(1, 1, figsize=(12, 12))

            m = maps.Map(
                airport_fn=AIRPORTS,
                ax=ax,
                corners=field.corners,
                )

            dm = maps.DataMap(
                field=field,
                contour_fields=contour_fields,
                hatch_fields=hatch_fields,
                map_=m,
                )

            dm.draw(show=True)

            png_suffix = level if level != 'ua' else ''
            png_file = f'{variable}{"_" + tile}{png_suffix}.png'
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

            plt.close()

def webname(prefix, tile='', suffix=''):
    ''' Return the filename expected for the web graphic. '''
    return f"{prefix}{'_' + tile}{'_' + suffix}"

def parse_args():

    ''' Set up argparse command line arguments, and return the Namespace
    containing the settings. '''

    parser = argparse.ArgumentParser(description='Script to drive the creation of graphics.')

    # Short args
    parser.add_argument('-d',
                        dest='data_root',
                        help='Cycle-independant data directory location.',
                        required=True,
                        )
    parser.add_argument('-f',
                        dest='fcst_hour',
                        help='A list describing forecast hours.' +
                        'If one argument, one fhr will be processed.' +
                        'If 2 or 3 arguments, a sequence of forecast' +
                        ' hours [start, stop, [increment]] will be ' +
                        'processed. If more than 3 arguments, the list ' +
                        'is processed as-is.',
                        nargs='+',
                        required=True,
                        type=int,
                        )
    parser.add_argument('-n',
                        default=1,
                        dest='nprocs',
                        help='Number of processes to use for parallelization.',
                        )
    parser.add_argument('-o',
                        dest='output_path',
                        help='Directory location desired for the output graphics files.',
                        required=True,
                        )
    parser.add_argument('-s',
                        dest='start_time',
                        help='Start time in YYYYMMDDHH format',
                        required=True,
                        type=utils.to_datetime,
                        )
    parser.add_argument('--subh_freq',
                        default=60,
                        help='Sub-hourly frequency in minutes.',
                        )
    parser.add_argument('-z',
                        dest='zip_dir',
                        help='Full path to zip directory.',
                        )

    # Long args
    parser.add_argument('--file_tmpl',
                        default='wrfnat_hrconus_{FCST_TIME:02d}.grib2',
                        help='File naming convention',
                        )
    parser.add_argument('--file_type',
                        choices=('nat', 'prs'),
                        default='prs',
                        help='Type of levels contained in grib file.',
                        )
    parser.add_argument('--image_list',
                        help='Path to YAML config file specifying which graphics to create.',
                        required=True,
                        )
    parser.add_argument('--image_set',
                        choices=['hourly'],
                        help='Name of top level key in image_list',
                        required=True,
                        )
    return parser.parse_args()

if __name__ == '__main__':
    CLARGS = parse_args()
    CLARGS.fcst_hour = utils.fhr_list(CLARGS.fcst_hour)

    main(CLARGS)
