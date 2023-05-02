#pylint: disable=unused-variable
''' Tests for create_graphics driver '''
from create_graphics import create_graphics
from create_graphics import parse_args
import os
import os.path

data_loc = os.environ.get("data_loc")
output_loc = os.environ.get("output_loc")


def test_parse_args():
    ''' Test parse_args for basic parsing success. 
        Checks if parse_args returns 'maps' in the graphic_type field.
    '''
    args = ['maps', '-d', str(data_loc), '-f', '0', '12', '1', '-o', str(output_loc),\
         '-s', '2021052315', '--file_tmpl', 'hrrr.t00z.wrfprsf{FCST_TIME:02d}.grib2', \
            '--images', './image_lists/hrrr_test.yml', 'hourly', '--all_leads', '--file_type=prs']
    test_args = parse_args(args)
    assert test_args.graphic_type == 'maps'


def test_existence():
    ''' Test function to build HRRR 12-hour accumulated maps. '''
    # Build Maps
    args = ['maps', '-d', str(data_loc), '-f', '0', '12', '1', '-o', str(output_loc),\
         '-s', '2023031500', '--file_tmpl', 'hrrr.t00z.wrfprsf{FCST_TIME:02d}.grib2', \
            '--images', './image_lists/hrrr_test.yml', 'hourly', '--all_leads', '--file_type=prs']
    create_graphics(args)

    # Test for existence of output folders
    folder_0 = "/202303150100"
    full_path_0 = output_loc + folder_0
    p0 = os.path.isdir(full_path_0)
    print("Full path:", full_path_0)
    assert p0 == True