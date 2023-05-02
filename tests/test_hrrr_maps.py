#pylint: disable=unused-variable
''' Tests for create_graphics driver '''
from create_graphics import create_graphics
import os
import os.path

data_loc = os.environ.get("data_loc")
output_loc = os.environ.get("output_loc")


def test_parse_args():
    ''' Test parse_args for basic parsing success. '''
    args = ['create_graphics', 'maps', '-d', str(data_loc), '-f', '0', '12', '1', '-o', str(output_loc),\
         '-s', '2021052315', '--file_tmpl', 'hrrr.t00z.wrfprsf{FCST_TIME:02d}.grib2', \
            '--images', './image_lists/hrrr_test.yml', 'hourly', '--all_leads', '--file_type=prs']
    test_args = parse_args(args)
    assert test_args.graphic_type == 'maps'


def test_maps():
    ''' Test function to build HRRR 12-hour accumulated maps. '''
    args = ['maps', '-d', str(data_loc), '-f', '0', '12', '1', '-o', str(output_loc),\
         '-s', '2021052315', '--file_tmpl', 'hrrr.t00z.wrfprsf{FCST_TIME:02d}.grib2', \
            '--images', './image_lists/hrrr_test.yml', 'hourly', '--all_leads', '--file_type=prs']
    create_graphics(args)


def test_existence():
    ''' Test function to check for existence of output folders and files. '''
    folder_0 = '/202303150100'
    full_path_0 = output_loc + folder_0
    p0 = os.path.isdir(full_path_0)
    print("Full path:", flul_path_0)
    assert p0 == True