# pylint: disable=invalid-name
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
import random
import string
import subprocess
import time

import yaml

from adb_graphics.datahandler import gribfile
import adb_graphics.errors as errors
from adb_graphics.figure_builders import parallel_maps, parallel_skewt
from adb_graphics.figures import maps
import adb_graphics.utils as utils


AIRPORTS = 'static/Airports_locs.txt'

COMBINED_FN = 'combined_{fhr:03d}_{uniq}.grib2'
TMP_FN = 'combined_{fhr:03d}_{uniq}.tmp.grib2'

LOG_BREAK = f"{('-' * 80)}\n{('-' * 80)}"

def check_file(cla, fhr, mem=None):
    ''' Given the command line arguments, the forecast hour, and a potential
    ensemble member, build a full path to the file and ensure it exists. '''

    grib_path = os.path.join(cla.data_root[0], cla.file_tmpl[0])
    if mem is not None:
        grib_path = grib_path.format(FCST_TIME=fhr, mem=mem)
    else:
        grib_path = grib_path.format(FCST_TIME=fhr)

    print(f'Checking on file {grib_path}')
    old_enough = utils.old_enough(cla.data_age, grib_path) if \
        os.path.exists(grib_path) else False
    return grib_path, old_enough

def create_skewt(cla, fhr, grib_path, workdir):

    ''' Generate arguments for parallel processing of Skew T graphics,
    and generate a pool of workers to complete the tasks. '''

    # Create the file object to load the contents
    gfile = gribfile.GribFile(grib_path)

    args = [(cla, fhr, gfile.contents, site, workdir) for site in cla.sites]

    print(f'Queueing {len(args)} Skew Ts')
    with Pool(processes=cla.nprocs) as pool:
        pool.starmap(parallel_skewt, args)

def create_maps(cla, fhr, gribfiles, workdir):

    ''' Generate arguments for parallel processing of plan-view maps and
    generate a pool of workers to complete the task. '''


    model = cla.images[0]
    for tile in cla.tiles:
        args = []
        for variable, levels in cla.images[1].items():
            for level in levels:

                # Load the spec for the current variable
                spec = cla.specs.get(variable, {}).get(level)

                if not spec:
                    msg = f'graphics: {variable} {level}'
                    raise errors.NoGraphicsDefinitionForVariable(msg)

                args.append((cla, fhr, gribfiles.contents, level, model, spec,
                             variable, workdir, tile))

        print(f'Queueing {len(args)} maps')
        with Pool(processes=cla.nprocs) as pool:
            pool.starmap(parallel_maps, args)

def gather_gribfiles(cla, fhr, filename, gribfiles):

    ''' Returns the appropriate gribfiles object for the type of graphics being
    generated -- whether it's for a single forecast time or all forecast lead
    times. '''

    filenames = {'01fcst': [], 'free_fcst': []}

    fcst_hour = int(fhr)

    first_fcst = 6 if 'global' in cla.images[0] else 1
    if fcst_hour <= first_fcst:
        filenames['01fcst'].append(filename)
    else:
        filenames['free_fcst'].append(filename)

    if gribfiles is None or not cla.all_leads:

        # Create a new GribFiles object, include all hours, or just this one,
        # depending on command line argument flag

        gribfiles = gribfile.GribFiles(
            coord_dims={'fcst_hr': [fhr]},
            filenames=filenames,
            filetype=cla.file_type,
            model=cla.images[0],
            )
    else:

        # Append a single forecast hour to the existing GribFiles object.
        gribfiles.coord_dims.get('fcst_hr').append(fhr)
        gribfiles.append(filenames)

    return gribfiles

def generate_tile_list(arg_list):

    ''' Given the input arguments -- a list if the argument is provided, return
    the list. If no arg is provided, defaults to the full domain, and if 'all'
    is provided, the full domain, and all subdomains are plotted. '''

    if not arg_list:
        return ['full']

    if ',' in arg_list[0]:
        arg_list = arg_list[0].split(',')

    hrrr_ak_only = ('Anchorage', 'AKRange', 'Juneau')
    rap_only = ('AK', 'AKZoom', 'conus', 'HI')
    if 'all' in arg_list:
        all_list = ['full'] + list(maps.TILE_DEFS.keys())
        return [tile for tile in all_list if tile not in hrrr_ak_only + rap_only]

    return arg_list

def load_images(arg):

    ''' Check that input image file exists, and that it contains the
    requested section. Return a 2-list (required by argparse) of the
    file path and dictionary of images to be created.
    '''

    # Agument is expected to be a 2-list of file name and internal
    # section name.

    image_file = arg[0]
    image_set = arg[1]

    # Check that the file exists
    image_file = utils.path_exists(image_file)

    # Load yaml file
    with open(image_file, 'r') as fn:
        images = yaml.load(fn, Loader=yaml.Loader)[image_set]

    return [images.get('model'), images.get('variables')]

def parse_args():

    ''' Set up argparse command line arguments, and return the Namespace
        containing the settings. '''

    parser = argparse.ArgumentParser(description='Script to drive the \
                                     creation of graphices files.')

    # Positional argument
    parser.add_argument(
        'graphic_type',
        choices=['maps', 'skewts', 'enspanel'],
        help='The type of graphics to create.',
        )

    # Short args
    parser.add_argument(
        '-r',
        dest='img_res',
        default=72,
        required=False,
        help='Resolution of output images in DPI. Recommended to stay below 1000. Default = 72',
        type=int,
        )
    parser.add_argument(
        '-a',
        dest='data_age',
        default=3,
        help='Age in minutes required for data files to be complete. Default = 3',
        type=int,
        )
    parser.add_argument(
        '-d',
        dest='data_root',
        help='Cycle-independant data directory location. Provide more than one \
        data path if data input files should be combined. When providing \
        multiple options, the same number of options is required for the \
        --file_tmpl flag.',
        nargs='+',
        required=True,
        )
    parser.add_argument(
        '-f',
        dest='fcst_hour',
        help='A list describing forecast hours.  If one argument, \
        one fhr will be processed.  If 2 or 3 arguments, a sequence \
        of forecast hours [start, stop, [increment]] will be \
        processed.  If more than 3 arguments, the list is processed \
        as-is.',
        nargs='+',
        required=True,
        type=int,
        )
    parser.add_argument(
        '-m',
        default='Unnamed Experiment',
        dest='model_name',
        help='string to use in title of graphic.',
        type=str,
        )
    parser.add_argument(
        '-n',
        default=1,
        dest='nprocs',
        help='Number of processes to use for parallelization.',
        type=int,
        )
    parser.add_argument(
        '-o',
        dest='output_path',
        help='Directory location desired for the output graphics files.',
        required=True,
        )
    parser.add_argument(
        '-s',
        dest='start_time',
        help='Start time in YYYYMMDDHH format',
        required=True,
        type=utils.to_datetime,
        )
    parser.add_argument(
        '-w',
        dest='wait_time',
        default=10,
        help='Time in minutes to wait on data files to be available. Default = 10',
        type=int,
        )
    parser.add_argument(
        '-z',
        dest='zip_dir',
        help='Full path to zip directory.',
        )

    # Long args
    parser.add_argument(
        '--all_leads',
        action='store_true',
        help='Use --all_leads to accumulate all forecast lead times.',
        )
    parser.add_argument(
        '--file_tmpl',
        default='wrfnat_hrconus_{FCST_TIME:02d}.grib2',
        nargs='+',
        help='File naming convention. Use FCST_TIME to indicate forecast hour. \
        Provide more than one template when data files should be combined. \
        When providing multiple options, the same number of options is required \
        for the -d flag.', \
        )
    parser.add_argument(
        '--file_type',
        choices=('nat', 'prs'),
        default='nat',
        help='Type of levels contained in grib file.',
        )

    # SkewT-specific args
    skewt_group = parser.add_argument_group('SkewT Arguments')
    skewt_group.add_argument(
        '--max_plev',
        help='Maximum pressure level to plot for profiles.',
        type=int,
        )
    skewt_group.add_argument(
        '--sites',
        help='Path to a sites file.',
        type=utils.load_sites,
        )

    # Map-specific args
    map_group = parser.add_argument_group('Map Arguments')
    map_group.add_argument(
        '--images',
        help='Path to YAML config file specifying which \
        variables to map and the top-level section to use.',
        metavar=('[FILE,', 'SECTION]'),
        nargs=2,
        )
    map_group.add_argument(
        '--obs_file_path',
        help='Path to an observation file. Currently this \
        feature is only supported for ensemble panel plots and \
        composite reflectivity.',
        type=utils.path_exists,
        )
    map_group.add_argument(
        '--specs',
        default='adb_graphics/default_specs.yml',
        help='Path to the specs YAML file.',
        )
    map_group.add_argument(
        '--subh_freq',
        default=60,
        help='Sub-hourly frequency in minutes.',
        )
    map_group.add_argument(
        '--tiles',
        default=['full'],
        help='The domains to plot. Choose from any of those listed. Special ' \
        'choices: full is full model output domain, and all is the full domain, ' \
        'plus all of the sub domains. ' \
        f'Choices: {["full", "all"] + maps.FULL_TILES + list(maps.TILE_DEFS.keys())}',
        nargs='+',
        )

    # Ensemble panel-specific args
    map_group = parser.add_argument_group('Ensemble Panel Arguments')
    map_group.add_argument(
        '--ens_size',
        default=10,
        help='Number of ensemble members.',
        type=int,
        )
    return parser.parse_args()

def pre_proc_grib_files(cla, fhr):

    ''' Use the command line argument object (cla) to determine the grib file
    loaction at a given forecast hour. If multiple data input paths and file
    templates are provided by user, concatenate the files and remove the
    duplicates. Return the file path of the file to be used by the graphics data
    handler, and whether the file is old enough. Files making it through the
    combined process here are assumed to be old enough.

    Input:
        cla     Program command line arguments in a  Namespace datastructure
        fhr     Forecast hour; integer

    Output
        grib_path    path to data used in plotting
        old_enough   bool stating whether the file is old enough as defined by
                     user settings. Combined files here are presumed old enough
                     by default
    '''

    if len(cla.data_root) == 1 and len(cla.file_tmpl) == 1:
        # Nothing to do, return the original file location
        return check_file(cla, fhr)

    # Generate a list of files to be joined.
    file_list = [os.path.join(*path).format(FCST_TIME=fhr) for path in
                 zip(cla.data_root, cla.file_tmpl)]
    for file_path in file_list:
        if not os.path.exists(file_path) \
            or not utils.old_enough(cla.data_age, file_path):
            return file_path, False

    print(f'Combining input files: ')
    for fn in file_list:
        print(f'  {fn}')

    file_rand = ''.join([random.choice(string.ascii_letters + string.digits) \
        for _ in range(8)])
    combined_fp = os.path.join(cla.output_path,
                               COMBINED_FN.format(fhr=fhr, uniq=file_rand))
    tmp_fp = os.path.join(cla.output_path,
                          TMP_FN.format(fhr=fhr, uniq=file_rand))

    cmd = f'cat {" ".join(file_list)} > {tmp_fp}'
    output = subprocess.run(cmd,
                            capture_output=True,
                            check=True,
                            shell=True,
                            )
    if output.returncode != 0:
        msg = f'{cmd} returned exit status: {output.returncode}!'
        raise OSError(msg)

    # Gather all grib2 entries from combined file
    cmd = f'wgrib2 {tmp_fp} -submsg 1'
    output = subprocess.run(cmd,
                            capture_output=True,
                            check=True,
                            shell=True,
                            )
    wgrib2_list = output.stdout.decode("utf-8").split('\n')

    # Create a unique list of grib fields.
    wgrib2_list = utils.uniq_wgrib2_list(wgrib2_list)

    # Remove duplicate grib2 entries in grib file
    cmd = f'wgrib2 -i {tmp_fp} -GRIB {combined_fp}'
    input_arg = '\n'.join(wgrib2_list).encode("utf-8")

    output = subprocess.run(cmd,
                            capture_output=True,
                            check=True,
                            input=input_arg,
                            shell=True,
                            )
    if output.returncode != 0:
        msg = f'{cmd} returned exit status: {output.returncode}'
        raise OSError(msg)
    os.remove(f'{tmp_fp}')

    return f'{combined_fp}', True

def remove_accumulated_images(cla):

    ''' Searches for all images that correspond with specs that have the
    accumulate entry set to True and removes them from the list of images to
    create. '''

    for variable, levels in cla.images[1].items():
        for level in levels:
            spec = cla.specs.get(variable, {}).get(level)
            if not spec:
                msg = f'graphics: {variable} {level}'
                raise errors.NoGraphicsDefinitionForVariable(msg)
            accumulate = spec.get('accumulate', False)

            if accumulate:
                print(f'Will not plot {variable}:{level}')
                cla.images[1][variable].remove(level)
                if not cla.images[1][variable]:
                    del cla.images[1][variable]

def remove_proc_grib_files(cla):

    ''' Find all processed grib files produced by this script and remove them.
    '''

    # Prepare template with all viable forecast hours -- glob accepts *
    combined_fn = COMBINED_FN.format(fhr=999, uniq=999).replace('999', '*')
    combined_fp = os.path.join(cla.output_path, combined_fn)

    combined_files = glob.glob(combined_fp)

    if combined_files:
        print(f'Removing combined files: ')
        for file_path in combined_files:
            print(f'  {file_path}')
            os.remove(file_path)

def stage_zip_files(tiles, zip_dir):

    ''' Stage the zip files in the appropriate directory for each tile to be
    plotted. Return the dictionary of zipfile paths.

    Input:

        tiles    list of subregions to plot from larger domain. becomes the
                 subdirectory under the zip_dir
        zip_dir  the top level zip file directory where files are expected to
                 show up

    Returns:
        zipfiles   dictionary of tile keys, and zip directory values.

    '''
    zipfiles = {}
    for tile in tiles:
        tile_zip_dir = os.path.join(zip_dir, tile)
        tile_zip_file = os.path.join(tile_zip_dir, 'files.zip')
        print(f"checking for {tile_zip_file}")
        if os.path.isfile(tile_zip_file):
            os.remove(tile_zip_file)
            print(f"{tile_zip_file} found and removed")
        os.makedirs(tile_zip_dir, exist_ok=True)
        zipfiles[tile] = tile_zip_file
    return zipfiles

@utils.timer
def graphics_driver(cla):

    # pylint: disable=too-many-statements
    # This whole script has likely reached the point of neededing refactoring
    # into an object oriented design....each graphics type is it's own object
    # sharing a base class.

    '''
    Function that interprets the command line arguments to locate the input grib
    file, create the output directory, and call the graphic-specifc function.

    Input:

      cla         Namespace object containing command line arguments.

    '''

    # pylint: disable=too-many-branches, too-many-locals

    # Create an empty zip file
    if cla.zip_dir:
        tiles = cla.tiles if cla.graphic_type == "maps" else ['skewt']
        if 'skewt' in tiles:
            tiles.append('skewt_csv')
        zipfiles = stage_zip_files(tiles, cla.zip_dir)

    fcst_hours = copy.deepcopy(cla.fcst_hour)

    # Initialize a timer used for killing the program
    timer_end = time.time()

    gribfiles = None

    # When accummulating variables for preparing a single lead time,
    # load all of those into gribfiles up front.
    # This is not an operational feature. Exit if files don't exist.

    if cla.graphic_type == 'maps':
        first_fcst = 6 if 'global' in cla.images[0] else 0
        fcst_inc = 6 if 'global' in cla.images[0] else 1
        if len(cla.fcst_hour) == 1 and cla.all_leads:
            for fhr in range(first_fcst, int(cla.fcst_hour[0]), fcst_inc):
                grib_path, old_enough = pre_proc_grib_files(cla, fhr)
                if not os.path.exists(grib_path) or not old_enough:
                    msg = (f'File {grib_path} does not exist! Cannot accumulate',
                           f'data for this forecast lead time!')
                    remove_proc_grib_files(cla)
                    raise FileNotFoundError(' '.join(msg))
                gribfiles = gather_gribfiles(cla, fhr, grib_path, gribfiles)


    # Allow this task to run concurrently with UPP by continuing to check for
    # new files as they become available.
    while fcst_hours:
        timer_sleep = time.time()
        old_enough = False
        for fhr in sorted(fcst_hours):
            if cla.graphic_type == 'enspanel':
                # Expand template to create a list of ensemble member files and
                # check if they exist and that they're old enough
                grib_paths = []
                ens_members = list(range(cla.ens_size))
                for mem in ens_members:
                    mem_path, mem_old_enough = check_file(cla, fhr, mem=mem)
                    if mem_old_enough:
                        grib_paths.append(mem_path)
                    old_enough = len(grib_paths) == cla.ens_size
            else:
                grib_path, old_enough = pre_proc_grib_files(cla, fhr)

            # UPP is most likely done writing if it hasn't written in data_age
            # mins (default is 3 to address most CONUS-sized domains)
            if old_enough:
                fcst_hours.remove(fhr)
            else:
                if cla.all_leads:
                    # Wait on the missing file for an arbitrary 90% of wait time
                    if time.time() - timer_end > cla.wait_time * 60 * .9:
                        print(f"Giving up waiting on {grib_path}. \n",
                              f"Removing accumulated variables from image list \n",
                              f"{LOG_BREAK}\n")
                        remove_accumulated_images(cla)
                        # Explicitly set -all_leads to False
                        cla.all_leads = False
                    else:
                        # Break out of loop, wait for the desired period, and start
                        # back at this forecast hour.
                        print(f'Waiting for {grib_path} to be available.')
                        break
                # It's safe to continue on processing the next forecast hour
                print(f'Cannot find specified file(s), continuing to check on \n \
                    next forecast hour.')
                continue

            # Create the working directory
            workdir = os.path.join(cla.output_path,
                                   f"{utils.from_datetime(cla.start_time)}{fhr:02d}")
            os.makedirs(workdir, exist_ok=True)

            print(f'{LOG_BREAK}\n',
                  f'Graphics will be created for input files\n',
                  f'Output graphics directory: {workdir} \n'
                  f'{LOG_BREAK}')

            if cla.graphic_type == 'skewts':
                create_skewt(cla, fhr, grib_path, workdir)
            elif cla.graphic_type == 'maps':
                gribfiles = gather_gribfiles(cla, fhr, grib_path, gribfiles)
                create_maps(cla,
                            fhr=fhr,
                            gribfiles=gribfiles,
                            workdir=workdir,
                            )
            else:
                gribfiles = gribfile.GribFiles(
                    coord_dims={'ens_mem': ens_members},
                    filenames={'free_fcst': grib_paths},
                    filetype=cla.file_type,
                    model=cla.images[0],
                    )
                create_maps(cla,
                            fhr=fhr,
                            gribfiles=gribfiles,
                            workdir=workdir,
                            )

            # Zip png files and remove the originals in a subprocess
            if cla.zip_dir:
                utils.zip_products(fhr, workdir, zipfiles)

            # Keep track of last time we did something useful
            timer_end = time.time()

        # Give up trying to process remaining forecast hours after waiting
        # wait_time mins. This accounts for slower UPP processes. Default for
        # most CONUS-sized domains is 10 mins.
        if time.time() - timer_end > cla.wait_time * 60:
            print(f"Exiting with forecast hours remaining: {fcst_hours}",
                  f"{LOG_BREAK}")
            break

        # Wait for a bit if it's been < 2 minutes (about the length of time UPP
        # takes) since starting last loop
        if fcst_hours and time.time() - timer_sleep < 120:
            print(f"Waiting for a minute for forecast hours: {fcst_hours}",
                  f"{LOG_BREAK}")
            time.sleep(60)

        remove_proc_grib_files(cla)

if __name__ == '__main__':

    CLARGS = parse_args()
    CLARGS.fcst_hour = utils.fhr_list(CLARGS.fcst_hour)

    # Check that the same number of entries exists in -d and --file_tmpl
    if len(CLARGS.data_root) != len(CLARGS.file_tmpl):
        errmsg = 'Must specify the same number of arguments for -d and --file_tmpl'
        print(errmsg)
        raise argparse.ArgumentError

    # Ensure wgrib command is available in environment before getting too far
    # down this path...
    if len(CLARGS.data_root) > 1:
        retcode = subprocess.run('which wgrib2', shell=True, check=True)
        if retcode.returncode != 0:
            errmsg = 'Could not find wgrib2, please make sure it is loaded \n \
            in your environment.'
            raise OSError(errmsg)

    # Only need to load the default in memory if we're making maps.
    if CLARGS.graphic_type in ['maps', 'enspanel']:
        CLARGS.specs = utils.load_specs(CLARGS.specs)

        CLARGS.images = load_images(CLARGS.images)
        CLARGS.tiles = generate_tile_list(CLARGS.tiles)

    print(f"Running script for {CLARGS.graphic_type} with args: ",
          f"{LOG_BREAK}")

    for name, val in CLARGS.__dict__.items():
        if name not in ['specs', 'sites']:
            print(f"{name:>15s}: {val}")
    graphics_driver(CLARGS)
