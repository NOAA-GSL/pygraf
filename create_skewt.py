'''
Driver for creating all the SkewT diagrams needed for a specific input dataset.
'''

import argparse
import datetime as dt
import os

import matplotlib.pyplot as plt
import yaml

from adb_graphics.datahandler import grib
import adb_graphics.errors as errors
from adb_graphics.figures import skewt
import adb_graphics.utils as utils

def parse_args():

    ''' Set up argparse command line arguments, and return the Namespace
        containing the settings. '''

    parser = argparse.ArgumentParser(description='Script to drive the \
                                     creation of SkewT diagrams.')

    parser.add_argument('-d',
                        dest='data_root',
                        help='Cycle-independant data directory location.',
                        required=True,
                        )
    parser.add_argument('-f',
                        dest='fcst_hour',
                        help='Forecast hour',
                        required=True,
                        type=int,
                        )
    parser.add_argument('--file_tmpl',
                        default='wrfnat_hrconus_{FCST_TIME:02d}.grib2',
                        help='File naming convention',
                        )
    parser.add_argument('--sites',
                        help='Path to a sites file.',
                        type=utils.file_exists,
                        )
    parser.add_argument('--max_plev',
                        help='Maximum pressure level to plot for profiles.',
                        type=int,
                        )
    parser.add_argument('-n',
                        default=1,
                        dest='nprocs',
                        help='Number of processes to use for parallelization.',
                        type=int,
                       )
    parser.add_argument('-o',
                        dest='output_path',
                        help='Directory location desired for the output graphics files.',
                        required=True,
                        )
    parser.add_argument('-s', '--start_time',
                        dest='start_time',
                        help='Start time in YYYYMMDDHH format',
                        required=True,
                        type=utils.to_datetime,
                        )

    return parser.parse_args()


def prepare_skewt(cla):

    # Locate input grib file
    str_start_time = utils.from_datetime(cla.start_time)
    grib_file = cla.file_tmpl.format(FCST_TIME=cla.fcst_hour)
    grib_path = os.path.join(cla.data_root, grib_file)

    # Create the working directory
    workdir = os.path.join(cla.output_path, f"{str_start_time}{cla.fcst_hour:02d}")
    os.makedirs(workdir, exist_ok=True)

    # Load sites
    with open(cla.sites, 'r') as sites_file:
        sites = sites_file.readlines()

    print((('-' * 120)+'\n') * 2)
    print(f'Creating graphics for input file: {grib_path}')
    print(f'Output graphics directory: {workdir}')
    print()
    print((('-' * 120)+'\n') * 2)

    for site in sites:
        skew = skewt.SkewTDiagram(filename=grib_path, loc=site, max_plev=cla.max_plev)
        skew.create_diagram()
        outfile = f"skewt_{skew.site_code}_{skew.site_num}_f{cla.fcst_hour:02d}.png"
        png_path = os.path.join(workdir, outfile)

        print('*' * 120)
        print(f"Creating image file: {png_path}")
        print('*' * 120)

        plt.savefig(
            png_path,
            bbox_inches='tight',
            dpi='figure',
            format='png',
            orientation='landscape',
            )


if __name__ == '__main__':

    CLARGS = parse_args()
    prepare_skewt(CLARGS)
