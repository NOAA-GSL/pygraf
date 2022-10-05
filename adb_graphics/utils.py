# pylint: disable=invalid-name
'''
A set of generic utilities available to all the adb_graphics components.
'''

import argparse
import datetime as dt
import functools
import glob
import importlib as il
from math import atan2, degrees
from multiprocessing import Process
import os
import subprocess
import sys
import time

import numpy as np
import yaml


def create_zip(files_to_zip, zipf):

    ''' Create a zip file. Use a locking mechanism -- write a lock file to disk. '''

    lock_file = f'{zipf}._lock'
    retry = 2
    count = 0
    while True:
        if not os.path.exists(lock_file):
            # Create the lock
            fd = open(lock_file, 'w')
            print(f'Writing to zip file {zipf} for files like: {files_to_zip[0][-10:]}')

            cmd = f'zip -uj {zipf} {" ".join(files_to_zip)}'
            print(f'Running command: {cmd}')
            try:
                subprocess.run(cmd,
                               check=True,
                               shell=True,
                               )
            except: # pylint: disable=bare-except
                print(f'Error on writing zip file! {sys.exc_info()[0]}')
                count += 1
                if count >= retry:
                    raise
            else:
                # Zipping was successful. Remove files that were zipped
                for file_to_zip in files_to_zip:
                    if os.path.exists(file_to_zip):
                        os.remove(file_to_zip)
            finally:
                # Remove the lock
                fd.close()
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            break
        # Wait before trying to obtain the lock on the file
        time.sleep(5)

def fhr_list(args):

    '''
    Given an argparse list argument, return the sequence of forecast hours to
    process.

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
        args[1] += 1
        return list(range(*args))

    return args

def from_datetime(date):
    ''' Return a string like YYYYMMDDHH given a datetime object. '''
    return dt.datetime.strftime(date, '%Y%m%d%H')

def get_func(val: str):

    '''
    Given an input string, val, returns the corresponding callable function.
    This function is borrowed from stackoverflow.com response to "Python: YAML
    dictionary of functions: how to load without converting to strings."
    '''

    if '.' in val:
        module_name, fun_name = val.rsplit('.', 1)
    else:
        module_name = '__main__'
        fun_name = val

    mod_spec = il.util.find_spec(module_name, package='adb_graphics')
    if mod_spec is None:
        mod_spec = il.util.find_spec('.' + module_name, package='adb_graphics')

    try:
        __import__(mod_spec.name)
    except ImportError as exc:
        print(f'Could not load {module_name} while trying to locate function in get_func')
        raise exc
    module = sys.modules[mod_spec.name]
    fun = getattr(module, fun_name)
    return fun

# pylint: disable=unused-argument
def join_ranges(loader, node):

    '''
    Merge two or more different ranges into a single array for color bar clevs.

    e.g.: in default_specs.yml, clevs for visibility can be assigned as

        clevs: !join_ranges [[0, 10, 0.1], [10, 51, 1.0]]

    The join_ranges method concatenates these ranges into a single array of levels.
    This can be useful for plots where one part of the color ranges requires higher
    resolution than the rest, while keeping the colorbar from looking squished.

    Note that a "yaml.add_constructor" is required, as shown after the method.
    '''

    list_ = []
    for seq_node in node.value:
        range_args = []
        for scalar_node in seq_node.value:
            range_args.append(float(scalar_node.value))

        list_.append(np.arange(*range_args))

    return np.concatenate(list_, axis=0)

yaml.add_constructor("!join_ranges", join_ranges, Loader=yaml.Loader)

# pylint: disable=invalid-name, too-many-locals
def label_line(ax, label, segment, **kwargs):

    '''
    Label a single line with line2D label data.

    Input:

      ax        the SkewT object axis
      label     label to be used for the current line
      segment   a list (array) of values for the current line

    Key Word Arguments

      align     optional bool to enable the rotation of the label to line angle
      end       the end of the line at which to put the label. 'bottom' or 'top'
      offset    index to use for the "end" of the array

      Any kwargs accepted by matplotlib's text box.
    '''

    # Strip non-text-box key word arguments and set default if they don't exist
    align = kwargs.pop('align', True)
    end = kwargs.pop('end', 'bottom')
    offset = kwargs.pop('offset', 0)

    # Label location
    if end == 'bottom':
        x, y = segment[0 + offset, :]
        ip = 1 + offset
    elif end == 'top':
        x, y = segment[-1 - offset, :]
        ip = -1 - offset

    if align:
        #Compute the slope
        dx = segment[ip, 0] - segment[ip-1, 0]
        dy = segment[ip, 1] - segment[ip-1, 1]
        ang = degrees(atan2(dy, dx))

        #Transform to screen co-ordinates
        pt = np.array([x, y]).reshape((1, 2))
        trans_angle = ax.transData.transform_angles(np.array((ang, )), pt)[0]

        if end == 'top':
            trans_angle -= 180

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'fontsize' not in kwargs:
        kwargs['fontsize'] = 'larger'

    if 'fontweight' not in kwargs:
        kwargs['fontweight'] = 'bold'

    # Larger value (e.g., 2.0) to move box in front of other diagram elements
    if 'zorder' not in kwargs:
        kwargs['zorder'] = 1.50

    # Place the text box label on the line.
    ax.text(x, y, label, rotation=trans_angle, **kwargs)

def label_lines(ax, lines, labels, offset=0, **kwargs):

    '''
    Plots labels on a set of lines from SkewT.

    Input:

      ax      the SkewT object axis
      lines   the SkewT object special lines
      labels  list of labels to be used
      offset  index to use for the "end" of the array

    Key Word Arguments

      color   line color

      Along with any other kwargs accepted by matplotlib's text box.
    '''

    if 'color' not in kwargs:
        kwargs['color'] = lines.get_color()[0]

    for i, line in enumerate(lines.get_segments()):
        label = int(labels[i])
        label_line(ax, label, line, align=True, offset=offset, **kwargs)

def load_sites(arg):

    ''' Check that the sites file exists, and return its contents. '''

    # Check that the file exists
    path = path_exists(arg)

    with open(path, 'r') as sites_file:
        sites = sites_file.readlines()
    return sites

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

def load_specs(arg):

    ''' Check to make sure arg file exists. Return its contents. '''

    spec_file = path_exists(arg)

    with open(spec_file, 'r') as fn:
        specs = yaml.load(fn, Loader=yaml.Loader)

    return specs

def old_enough(age, file_path):

    '''
    Helper function to test the age of a file.

    Input:

      age         desired age in minutes
      file_path   full path to file to check

    Output:

      bool    whether the file is at least age minutes old
    '''

    file_time = dt.datetime.fromtimestamp(os.path.getctime(file_path))
    max_age = dt.datetime.now() - dt.timedelta(minutes=age)

    return file_time < max_age

def path_exists(path: str):

    ''' Checks whether a file exists, and returns the path if it does. '''

    if not os.path.exists(path):
        msg = f'{path} does not exist!'
        raise argparse.ArgumentTypeError(msg)

    return path

def timer(func):

    ''' Decorator function that provides an elapsed time for a method. '''

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"{func.__name__} Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer

def to_datetime(string):
    ''' Return a datetime object give a string like YYYYMMDDHH. '''

    return dt.datetime.strptime(string, '%Y%m%d%H')

@timer
def zip_products(fhr, workdir, zipfiles):

    ''' Spin up a subprocess to zip all the product files into the staged zip files.

    Input:

        fhr         integer forecast hour
        workdir     path to the product files
        zipfiles    dictionary of tile keys, and zip directory values.

    Output:
        None
    '''

    for tile, zipf in zipfiles.items():
        if tile == 'skewt_csv':
            file_tmpl = f'*.skewt.*_f{fhr:03d}.csv'
        else:
            file_tmpl = f'*_{tile}_*{fhr:02d}.png'
        product_files = glob.glob(os.path.join(workdir, file_tmpl))
        if product_files:
            zip_proc = Process(group=None,
                               target=create_zip,
                               args=(product_files, zipf),
                               )
            zip_proc.start()
            zip_proc.join()
