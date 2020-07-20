'''
Driver for creating all the SkewT diagrams needed for a specific input dataset.
'''

# pylint: disable=wrong-import-position, wrong-import-order
import matplotlib as mpl
mpl.use('Agg')
# pylint: enable=wrong-import-position, wrong-import-order

import argparse
import glob
from multiprocessing import Pool
import os
import zipfile

import matplotlib.pyplot as plt

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
    parser.add_argument('-s',
                        dest='start_time',
                        help='Start time in YYYYMMDDHH format',
                        required=True,
                        type=utils.to_datetime,
                        )
    parser.add_argument('--sites',
                        help='Path to a sites file.',
                        type=utils.file_exists,
                        )
    parser.add_argument('-z',
                        dest='zip_dir',
                        help='Full path to zip directory.',
                        )

    return parser.parse_args()

def parallel_skewt(cla, grib_path, site, workdir):

    '''
    Function that creates a single SkewT plot. Can be used in parallel.
    Input:

      cla        command line arguments Namespace object
      grib_path  the full path to the grib file
      site       the string representation of the site from the sites file
      workdir    output directory
    '''

    skew = skewt.SkewTDiagram(filename=grib_path, loc=site, max_plev=cla.max_plev)
    skew.create_diagram()
    outfile = f"skewt_{skew.site_code}_{skew.site_num}_f{cla.fcst_hour:02d}.png"
    png_path = os.path.join(workdir, outfile)

    print('*' * 80)
    print(f"Creating image file: {png_path}")
    print('*' * 80)

    # pylint: disable=duplicate-code
    plt.savefig(
        png_path,
        bbox_inches='tight',
        dpi='figure',
        format='png',
        orientation='landscape',
        )
    plt.close()

@utils.timer
def prepare_skewt(cla):

    '''
    Function that interprets the command line arguments to locate the input grib
    file, create the output directory, and set up creating Skew-T diagrams in
    parallel.

    Input:

      cla         Namespace object containing command line arguments.

    '''

    # Locate input grib file
    str_start_time = utils.from_datetime(cla.start_time)
    grib_file = cla.file_tmpl.format(FCST_TIME=cla.fcst_hour)
    grib_path = os.path.join(cla.data_root, grib_file)

    if not os.path.exists(grib_path):
        raise IOError(f"{grib_path} not found!")

    # Create the working directory
    workdir = os.path.join(cla.output_path, f"{str_start_time}{cla.fcst_hour:02d}")
    os.makedirs(workdir, exist_ok=True)

    # Create an empty zip file
    zipf = None
    if cla.zip_dir:
        os.makedirs(cla.zip_dir, exist_ok=True)
        zip_path = os.path.join(cla.zip_dir, 'files.zip')
        zipf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)

    # Load sites
    with open(cla.sites, 'r') as sites_file:
        sites = sites_file.readlines()

    print((('-' * 80)+'\n') * 2)
    print()
    print(f'Creating graphics for input file: {grib_path}')
    print(f'Output graphics directory: {workdir}')
    print()
    print((('-' * 80)+'\n') * 2)

    skewt_args = []
    for site in sites:
        skewt_args.append((cla, grib_path, site, workdir))

    with Pool(processes=cla.nprocs) as pool:
        pool.starmap(parallel_skewt, skewt_args)

    # Zip png files and remove the original
    if zipf:
        png_files = glob.glob(os.path.join(workdir, '*.png'))
        for png_file in png_files:
            zipf.write(png_file)
            os.remove(png_file)


if __name__ == '__main__':

    CLARGS = parse_args()
    prepare_skewt(CLARGS)
