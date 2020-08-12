'''
Driver for creating all the SkewT diagrams needed for a specific input dataset.
'''

# pylint: disable=wrong-import-position, wrong-import-order
import matplotlib as mpl
mpl.use('Agg')
# pylint: enable=wrong-import-position, wrong-import-order

import argparse
import copy
import glob
from multiprocessing import Pool
import os
import time
import zipfile

import matplotlib.pyplot as plt

from adb_graphics.figures import skewt
import adb_graphics.utils as utils

def fhr_list(args):

    '''
    Given an arg list, return the sequence of forecast hours to process.

    The length of the list will determine what forecast hours are returned:

      Length = 1:   A single fhr is to be processed
      Length = 2:   A sequence of start, stop with increment 1
      Length = 3:   A sequence of start, stop, increment
      Length > 3:   List as is

    argparse should provide a list of at least one item (nargs='+').

    Must ensure that the list contains integers.
    '''

    args = args if isinstance(args, list) else [args]
    arg_len = len(args)
    if arg_len in (2, 3):
        return list(range(*args))

    return args

def parse_args():

    ''' Set up argparse command line arguments, and return the Namespace
        containing the settings. '''

    parser = argparse.ArgumentParser(description='Script to drive the \
                                     creation of SkewT diagrams.')

    # Short args
    parser.add_argument('-d',
                        dest='data_root',
                        help='Cycle-independant data directory location.',
                        required=True,
                        type=utils.path_exists,
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
                        default='nat',
                        help='Type of levels contained in grib file.',
                        )
    parser.add_argument('--max_plev',
                        help='Maximum pressure level to plot for profiles.',
                        type=int,
                        )
    parser.add_argument('--sites',
                        help='Path to a sites file.',
                        type=utils.path_exists,
                        )

    return parser.parse_args()

def parallel_skewt(cla, fhr, grib_path, site, workdir):

    '''
    Function that creates a single SkewT plot. Can be used in parallel.
    Input:

      cla        command line arguments Namespace object
      grib_path  the full path to the grib file
      site       the string representation of the site from the sites file
      workdir    output directory
    '''

    skew = skewt.SkewTDiagram(
        filename=grib_path,
        filetype=cla.file_type,
        loc=site,
        max_plev=cla.max_plev,
        )
    skew.create_diagram()
    outfile = f"skewt_{skew.site_code}_{skew.site_num}_f{fhr:02d}.png"
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

    # Create an empty zip file
    zipf = None
    if cla.zip_dir:
        os.makedirs(cla.zip_dir, exist_ok=True)
        zipf = os.path.join(cla.zip_dir, 'files.zip')
        if os.path.exists(zipf):
            os.remove(zipf)

    # Load sites
    with open(cla.sites, 'r') as sites_file:
        sites = sites_file.readlines()

    fcst_hours = copy.deepcopy(cla.fcst_hour)

    # Initialize a timer used for killing the program
    timer_end = time.time()

    # Allow this task to run concurrently with UPP by continuing to check for
    # new files as they become available.
    while fcst_hours:
        timer_sleep = time.time()
        for fhr in sorted(fcst_hours):
            grib_path = os.path.join(cla.data_root,
                                     cla.file_tmpl.format(FCST_TIME=fhr))

            # UPP is most likely done writing if it hasn't written in 3 mins
            if os.path.exists(grib_path) and utils.old_enough(3, grib_path):
                fcst_hours.remove(fhr)
            else:
                # Try next forecast hour
                continue

            # Create the working directory
            workdir = os.path.join(cla.output_path,
                                   f"{utils.from_datetime(cla.start_time)}{fhr:02d}")
            os.makedirs(workdir, exist_ok=True)

            print((('-' * 80)+'\n') * 2)
            print()
            print(f'Graphics will be created for input file: {grib_path}')
            print(f'Output graphics directory: {workdir}')
            print()
            print((('-' * 80)+'\n') * 2)

            skewt_args = [(cla, fhr, grib_path, site, workdir) for site in
                          sites]

            with Pool(processes=cla.nprocs) as pool:
                pool.starmap(parallel_skewt, skewt_args)

            # Zip png files and remove the originals
            if zipf:
                png_files = glob.glob(os.path.join(workdir, '*.png'))
                with zipfile.ZipFile(zipf, 'a', zipfile.ZIP_DEFLATED) as zfile:
                    for png_file in png_files:
                        zfile.write(png_file, os.path.basename(png_file))
                        os.remove(png_file)
                # Directory is empty now -- rmdir is fine.
                os.rmdir(workdir)

            # Keep track of last time we did something useful
            timer_end = time.time()

        # Give up trying to process remaining forecast hours after waiting 10
        # arbitrary mins since doing something useful.
        if time.time() - timer_end > 600:
            print(f"Exiting with forecast hours remaining: {fcst_hours}")
            print((('-' * 80)+'\n') * 2)
            break

        # Wait for a bit if it's been < 2 minutes (about the length of time UPP
        # takes) since starting last loop
        if fcst_hours and time.time() - timer_sleep < 120:
            print(f"Waiting for a minute before forecast hours: {fcst_hours}")
            print((('-' * 80)+'\n') * 2)
            time.sleep(60)


if __name__ == '__main__':

    CLARGS = parse_args()
    CLARGS.fcst_hour = fhr_list(CLARGS.fcst_hour)

    print(f"Running script with args: ")
    print((('-' * 80)+'\n') * 2)

    for name, val in CLARGS.__dict__.items():
        print(f"{name:>15s}: {val}")

    prepare_skewt(CLARGS)
