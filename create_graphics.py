# pylint: disable=invalid-name
'''
                create_maps(cla,
                            fhr=fhr,
                            gribfiles=gribfiles,
                            workdir=workdir,
                            )
Driver for creating all the SkewT diagrams needed for a specific input dataset.
'''

# pylint: disable=wrong-import-position, wrong-import-order
import matplotlib as mpl
mpl.use('Agg')
# pylint: enable=wrong-import-position, wrong-import-order

import argparse
import copy
import gc

import glob
from multiprocessing import Pool, Process
import os
import random
import string
import subprocess
import sys
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import yaml

from adb_graphics.datahandler import gribfile
from adb_graphics.datahandler import gribdata
import adb_graphics.errors as errors
from adb_graphics.figures import maps
from adb_graphics.figures import skewt
import adb_graphics.utils as utils


AIRPORTS = 'static/Airports_locs.txt'

COMBINED_FN = 'combined_{fhr:03d}_{uniq}.grib2'
TMP_FN = 'combined_{fhr:03d}_{uniq}.tmp.grib2'

LOG_BREAK = f"{('-' * 80)}\n{('-' * 80)}"

def add_obs_panel(ax, model_name, obs_file, proj_info, short_name, tile):

    ''' Plot observation data provided by the obs_file
    path using the assigned projection. '''

    gribobs = gribfile.GribFile(filename=obs_file)
    field = gribdata.fieldData(
        ds=gribobs.contents,
        fhr=0,
        level='obs',
        model='obs',
        short_name=short_name,
        )
    map_fields = maps.MapFields(main_field=field)
    m = maps.Map(
        airport_fn=AIRPORTS,
        ax=ax,
        grid_info=proj_info,
        model='obs',
        tile=tile,
        )
    dm = maps.DataMap(
        map_fields=map_fields,
        map_=m,
        model_name=model_name,
        multipanel=True,
        )

    # Draw the map
    dm.draw(show=True)

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

def create_zip(png_files, zipf):

    ''' Create a zip file. Use a locking mechanism -- write a lock file to disk. '''

    lock_file = f'{zipf}._lock'
    retry = 2
    count = 0
    while True:
        if not os.path.exists(lock_file):
            fd = open(lock_file, 'w')
            print(f'Writing to zip file {zipf} for files like: {png_files[0][-10:]}')

            try:
                with zipfile.ZipFile(zipf, 'a', zipfile.ZIP_DEFLATED) as zfile:
                    for png_file in png_files:
                        if os.path.exists(png_file):
                            zfile.write(png_file, os.path.basename(png_file))
            except: # pylint: disable=bare-except
                print(f'Error on writing zip file! {sys.exc_info()[0]}')
                count += 1
                if count >= retry:
                    raise
            else:
                # When zipping is successful, remove png_files
                for png_file in png_files:
                    if os.path.exists(png_file):
                        os.remove(png_file)
            finally:
                fd.close()
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            break
        # Wait before trying to obtain the lock on the file
        time.sleep(5)

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

def load_sites(arg):

    ''' Check that the sites file exists, and return its contents. '''

    # Check that the file exists
    path = utils.path_exists(arg)

    with open(path, 'r') as sites_file:
        sites = sites_file.readlines()
    return sites

def load_specs(arg):

    ''' Check to make sure arg file exists. Return its contents. '''

    spec_file = utils.path_exists(arg)

    with open(spec_file, 'r') as fn:
        specs = yaml.load(fn, Loader=yaml.Loader)

    return specs

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
        type=load_sites,
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


def parallel_maps(cla, fhr, ds, level, model, spec, variable, workdir,
                  tile='full'):

    # pylint: disable=too-many-arguments,too-many-locals

    '''
    Function that creates plan-view maps, either a single panel, or
    multipanel for a forecast ensemble. Can be used in parallel.

    Input:

      fhr        forecast hour
      ds         xarray dataset from the grib file
      level      the vertical level of the variable to be plotted
                 corresponding to a key in the specs file
      model      model name: rap, hrrr, hrrre, rrfs, rtma
      spec       the dictionary of specifications for the given variable
                 and level
      variable   the name of the variable section in the specs file
      workdir    output directory
    '''

    fig, axes = set_figure(cla.model_name, cla.graphic_type)

    for index, current_ax in enumerate(axes):

        mem = None
        if cla.graphic_type == 'enspanel':
            # Don't put data in the top left or bottom left panels.
            if index in (0, 8):
                current_ax.axis('off')
                continue
            # Shenanigans to match ensemble member to panel index
            mem = 0 if index == 4 else index
            mem = mem if mem < 4 else index - 1
            mem = mem if mem < 8 else index - 2

        # Object to be plotted on the map in filled contours.
        field = gribdata.fieldData(
            ds=ds,
            fhr=fhr,
            filetype=cla.file_type,
            level=level,
            member=mem,
            model=model,
            short_name=variable,
            )

        try:
            field.field
        except errors.GribReadError:
            print(f'Cannot find grib2 variable for {variable} at {level}. Skipping.')
            return

        map_fields = maps.MapFields(main_field=field, fields_spec=spec)

        # Generate a map object
        m = maps.Map(
            airport_fn=AIRPORTS,
            ax=current_ax,
            grid_info=field.grid_info(),
            model=model,
            plot_airports=spec.get('plot_airports', True),
            tile=tile,
            )

        # Send all objects (map_field, contours, hatches) to a DataMap object
        dm = maps.DataMap(
            map_fields=map_fields,
            map_=m,
            model_name=cla.model_name,
            multipanel=cla.graphic_type == 'enspanel',
            )

        # Draw the map
        dm.draw()

    # Add observation panel to lower left. Currently only supported for
    # composite reflectivity.
    if cla.graphic_type == 'enspanel' and spec.get('include_obs', False):
        add_obs_panel(
            ax=axes[8],
            model_name=cla.model_name,
            obs_file=cla.obs_file_path,
            proj_info=field.grid_info(),
            short_name=variable,
            tile=tile,
            )

    if cla.graphic_type == 'enspanel':
        # once all the subplots are ready, adjust to remove white space and make room for color bar
        plt.subplots_adjust(bottom=0.15, top=0.90, wspace=0, hspace=0)
        # add the color bar (based on last contour, but all should be the same)
        cax = plt.axes([0.15, 0.040, 0.70, 0.041])
#        plt.colorbar(contour, orientation='horizontal', cax=cax)

        # plot title
        title = "Ensemble plot"
        unit = "dbZ"
        fig.suptitle(f'{title} ({unit})', fontsize=18)

    # Build the output path
    png_file = f'{variable}_{tile}_{level}_f{fhr:03d}.png'
    png_file = png_file.replace("__", "_")
    png_path = os.path.join(workdir, png_file)

    print('*' * 120)
    print(f"Creating image file: {png_path}")
    print('*' * 120)

    # Save the png file to disk
    plt.savefig(
        png_path,
        bbox_inches='tight',
        dpi=cla.img_res,
        format='png',
        orientation='landscape',
        pil_kwargs={'optimize': True},
        )


    fig.clear()
    # Clear the current axes.
    plt.cla()
    # Clear the current figure.
    plt.clf()
    # Closes all the figure windows.
    plt.close('all')
    del field
    del m
    gc.collect()

def parallel_skewt(cla, fhr, ds, site, workdir):

    '''
    Function that creates a single SkewT plot. Can be used in parallel.
    Input:

      cla        command line arguments Namespace object
      ds         the XArray dataset
      fhr        the forecast hour integer
      site       the string representation of the site from the sites file
      workdir    output directory
    '''

    skew = skewt.SkewTDiagram(
        ds=ds,
        fhr=fhr,
        filetype=cla.file_type,
        loc=site,
        max_plev=cla.max_plev,
        model_name=cla.model_name,
        )
    skew.create_diagram()
    outfile = f"{skew.site_code}_{skew.site_num}_skewt_f{fhr:03d}.png"
    png_path = os.path.join(workdir, outfile)

    print('*' * 80)
    print(f"Creating image file: {png_path}")
    print('*' * 80)

    # pylint: disable=duplicate-code
    plt.savefig(
        png_path,
        bbox_inches='tight',
        dpi=cla.img_res,
        format='png',
        orientation='landscape',
        )
    plt.close()

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
    wgrib2_list = uniq_wgrib2_list(wgrib2_list)

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

def set_figure(model_name, graphic_type):

    ''' Create the figure and subplots appropriate for the model and
    graphics type. Return the figure handle and list of axes. '''

    if model_name == "HRRR-HI":
        inches = 12.2
    else:
        inches = 10

    # A 12 panel plot to accommodate 10 ensemble members, or a single panel
    if graphic_type == 'enspanel':
        nrows = 3
        ncols = 4
        inches = 20
    else:
        nrows = 1
        ncols = 1

    # Create a rectangle shape
    fig, ax = plt.subplots(nrows, ncols, figsize=(inches, 0.8*inches),)
    # Flatten the 2D array and number panel axes from top left to bottom right
    # sequentially
    ax = ax.flatten() if isinstance(ax, np.ndarray) else [ax]

    return fig, ax

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

def uniq_wgrib2_list(inlist):

    ''' Given a list of wgrib2 output fields, returns a uniq list of fields for
    simplifying a grib2 dataset. Uniqueness is defined by the wgrib output from
    field 3 (colon delimted) onward, although the original full grib record must
    be included in the wgrib2 command below.
    '''

    uniq_field_set = set()
    uniq_list = []
    for infield in inlist:
        infield_info = infield.split(':')
        if len(infield_info) <= 3:
            continue
        infield_str = ':'.join(infield_info[3:])
        if infield_str not in uniq_field_set:
            uniq_list.append(infield)
        uniq_field_set.add(infield_str)

    return uniq_list

def zip_pngs(fhr, workdir, zipfiles):

    ''' Spin up a subprocess to zip all the png files into the staged zip files.

    Input:

        fhr         integer forecast hour
        workdir     path to the png files
        zipfiles    dictionary of tile keys, and zip directory values.

    Output:
        None
    '''

    for tile, zipf in zipfiles.items():
        png_files = glob.glob(os.path.join(workdir, f'*_{tile}_*{fhr:02d}.png'))
        zip_proc = Process(group=None,
                           target=create_zip,
                           args=(png_files, zipf),
                           )
        zip_proc.start()
        zip_proc.join()

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
                print(gribfiles.contents)

            # Zip png files and remove the originals in a subprocess
            if cla.zip_dir:
                zip_pngs(fhr, workdir, zipfiles)

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
        CLARGS.specs = load_specs(CLARGS.specs)

        CLARGS.images = load_images(CLARGS.images)
        CLARGS.tiles = generate_tile_list(CLARGS.tiles)

    print(f"Running script for {CLARGS.graphic_type} with args: ",
          f"{LOG_BREAK}")

    for name, val in CLARGS.__dict__.items():
        if name not in ['specs', 'sites']:
            print(f"{name:>15s}: {val}")
    graphics_driver(CLARGS)
