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
import time
import zipfile

import matplotlib.pyplot as plt
from PIL import Image
import yaml

from adb_graphics.datahandler import grib
import adb_graphics.errors as errors
from adb_graphics.figures import maps
from adb_graphics.figures import skewt
import adb_graphics.utils as utils


AIRPORTS = 'static/Airports_locs.txt'


def add_logo(fpath, position):

    ''' Add the NOAA logo to a PNG image file, fpath '''

    logo_file = 'static/noaa-logo-100x100.png'
    logo_image = Image.open(logo_file)

    image = Image.open(fpath)
    image.paste(logo_image,
                position,
                logo_image)
    image.save(fpath)

def create_skewt(cla, fhr, grib_path, workdir):

    ''' Generate arguments for parallel processing of Skew T graphics,
    and generate a pool of workers to complete the tasks. '''

    args = [(cla, fhr, grib_path, site, workdir) for site in cla.sites]

    with Pool(processes=cla.nprocs) as pool:
        pool.starmap(parallel_skewt, args)

def create_maps(cla, fhr, grib_path, workdir):

    ''' Generate arguments for parallel processing of plan-view maps and
    generate a pool of workers to complete the task. '''

    args = []

    for tile in cla.tiles:
        for variable, levels in cla.images[1].items():
            for level in levels:

                # Load the spec for the current variable
                spec = cla.specs.get(variable, {}).get(level)

                if not spec:
                    msg = f'graphics: {variable} {level}'
                    raise errors.NoGraphicsDefinitionForVariable(msg)

                args.append((fhr, grib_path, level, spec,
                             variable, workdir, tile))

    with Pool(processes=cla.nprocs) as pool:
        pool.starmap(parallel_maps, args)

def generate_tile_list(arg_list):

    ''' Given the input arguments -- a list if the argument is provided, return
    the list. If no arg is provided, defaults to the full domain, and if 'all'
    is provided, the full domain, and all subdomains are plotted. '''

    if not arg_list:
        return ['full']

    if 'all' in arg_list:
        return ['full'] + list(maps.TILE_DEFS.keys())

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

    return [image_file, images.get('variables')]

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
        choices=['maps', 'skewts'],
        help='The type of graphics to create.',
        )

    # Short args
    parser.add_argument(
        '-d',
        dest='data_root',
        help='Cycle-independant data directory location.',
        required=True,
        type=utils.path_exists,
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
        '-z',
        dest='zip_dir',
        help='Full path to zip directory.',
        )

    # Long args
    parser.add_argument(
        '--file_tmpl',
        default='wrfnat_hrconus_{FCST_TIME:02d}.grib2',
        help='File naming convention',
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
        choices=['full', 'all'] + list(maps.TILE_DEFS.keys()),
        default=['full'],
        help='The domains to plot. Choose from any of those listed. Special \
        choices: full is full model output domain, and all is the full domain, \
        plus all of the sub domains.',
        nargs='+',
        )
    return parser.parse_args()

def parallel_maps(fhr, grib_path, level, spec, variable, workdir,
                  tile='full'):

    # pylint: disable=too-many-arguments,too-many-locals

    '''
    Function that creates a single plan-view map. Can be used in
    parallel.

    Input:

      fhr        forecast hour
      grib_path  the full path to the grib file
      level      the vertical level of the variable to be plotted
                 corresponding to a key in the specs file
      spec       the dictionary of specifications for the given variable
                 and level
      variable   the name of the variable section in the specs file
      workdir    output directory
    '''

    # Object to be plotted on the map in filled contours.
    field = grib.fieldData(
        fhr=fhr,
        filename=grib_path,
        level=level,
        short_name=variable,
        )

    # Create a list of fieldData objects for each contour field requested
    # These will show up as line contours on the plot.
    contours = spec.get('contours')
    contour_fields = []
    if contours is not None:
        for contour, contour_kwargs in contours.items():
            if '_' in contour:
                var, lev = contour.split('_')
            else:
                var, lev = contour, level

            contour_fields.append(grib.fieldData(
                fhr=fhr,
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
                fhr=fhr,
                filename=grib_path,
                level=lev,
                contour_kwargs=hatch_kwargs,
                short_name=var,
                ))

    _, ax = plt.subplots(1, 1, figsize=(12, 12))


    corners = field.corners if tile == 'full' else None

    # Generate a map object
    m = maps.Map(
        airport_fn=AIRPORTS,
        ax=ax,
        corners=corners,
        tile=tile,
        )

    # Send all objects (map, field, contours, hatches) to a DataMap object
    dm = maps.DataMap(
        field=field,
        contour_fields=contour_fields,
        hatch_fields=hatch_fields,
        map_=m,
        )

    # Draw the mape
    dm.draw(show=True)

    # Build the output path
    png_suffix = level if level != 'ua' else ''
    tile_label = f'_{tile}' if tile != 'full' else ''
    png_file = f'{variable}{tile_label}{png_suffix}_f{fhr:03d}.png'
    png_path = os.path.join(workdir, png_file)

    print('*' * 120)
    print(f"Creating image file: {png_path}")
    print('*' * 120)

    # Save the png file to disk
    plt.savefig(
        png_path,
        bbox_inches='tight',
        dpi='figure',
        format='png',
        orientation='landscape',
        )

    plt.close()

    add_logo(png_path, (20, 645))


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
        fhr=fhr,
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

    add_logo(png_path, (125, 875))

@utils.timer
def graphics_driver(cla):

    '''
    Function that interprets the command line arguments to locate the input grib
    file, create the output directory, and call the graphic-specifc function.

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



            if cla.graphic_type == 'skewts':
                create_skewt(cla, fhr, grib_path, workdir)
            else:
                create_maps(cla, fhr, grib_path, workdir)

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
    CLARGS.fcst_hour = utils.fhr_list(CLARGS.fcst_hour)

    # Only need to load the default in memory if we're making maps.
    if CLARGS.graphic_type == 'maps':
        CLARGS.specs = load_specs(CLARGS.specs)

        CLARGS.images = load_images(CLARGS.images)
        CLARGS.tiles = generate_tile_list(CLARGS.tiles)

    print(f"Running script for {CLARGS.graphic_type} with args: ")
    print((('-' * 80)+'\n') * 2)

    for name, val in CLARGS.__dict__.items():
        print(f"{name:>15s}: {val}")


    graphics_driver(CLARGS)
