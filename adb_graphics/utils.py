'''
A set of generic utilities available to all the adb_graphics components.
'''

import argparse
import importlib as il
from math import atan2, degrees
import os
import sys

import numpy as np


def file_exists(filename: str):

    ''' Checks whether a file exists, and returns the path if it does. '''

    if not os.path.exists(filename):
        msg = f'{filename} does not exist!'
        raise argparse.ArgumentTypeError(msg)

    return filename

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


# pylint: disable=invalid-name, too-many-locals
def label_line(line, x, label=None, align=True, **kwargs):

    ''' Label line with line2D label data '''

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i, xd in enumerate(xdata):
        if x < xd:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy, dx))

        #Transform to screen co-ordinates
        pt = np.array([x, y]).reshape((1, 2))
        trans_angle = ax.transData.transform_angles(np.array((ang, )), pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x, y, label, rotation=trans_angle, **kwargs)

def label_lines(lines, align=True, xvals=None, **kwargs):

    '''
    retrieved from
    https://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib
    '''

    #ax = lines[0].axes
    ax = kwargs.get('ax', lines.axes)
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines.get_segments():
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin, xmax = ax.get_xlim()
        xvals = np.linspace(xmin, xmax, len(labLines)+2)[1:-1]

    for line, x, label in zip(labLines, xvals, labels):
        label_line(line, x, label, align, **kwargs)
