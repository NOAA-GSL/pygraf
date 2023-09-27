#pylint: disable=unused-variable
''' Tests for create_graphics driver '''
import os
import pytest
from create_graphics import create_graphics
from create_graphics import parse_args

DATA_LOC = os.environ.get("data_loc")
OUTPUT_LOC = os.environ.get("data_loc")

@pytest.fixture(name="_setup")
def build_maps():
    ''' Builds HRRR 12-hour accumulated maps '''
    args = ['maps', '-d', DATA_LOC, '-f', '0', '12', '1', '-o', OUTPUT_LOC,\
         '-s', '2023031500', '--file_tmpl', 'hrrr.t00z.wrfprsf{FCST_TIME:02d}.grib2', \
            '--images', './image_lists/hrrr_test.yml', 'hourly', '--all_leads', '--file_type=prs']
    create_graphics(args)


def test_parse_args():
    ''' Test parse_args for basic parsing success.
        Checks if parse_args returns 'maps' in the graphic_type field.
    '''
    args = ['maps', '-d', DATA_LOC, '-f', '0', '12', '1', '-o', OUTPUT_LOC,\
         '-s', '2021052315', '--file_tmpl', 'hrrr.t00z.wrfprsf{FCST_TIME:02d}.grib2', \
            '--images', './image_lists/hrrr_test.yml', 'hourly', '--all_leads', '--file_type=prs']
    test_args = parse_args(args)
    assert test_args.graphic_type == 'maps'


def test_folder_existence(_setup):
    ''' Tests for existence of folders.
        Can be extended to cover multiple folders.
    '''
    folder = "/202303150000"
    full_path = OUTPUT_LOC + folder
    file_path = os.path.isdir(full_path)
    assert file_path


def test_file_count(_setup):
    ''' Test for file count in directory.
        Can be extended to cover multiple folders.
    '''
    # Based on the hrrr_test.yml file, only 6 maps will be created
    map_count = 6
    count = 0
    folder = "/202303150000/"
    for file_name in os.listdir(OUTPUT_LOC + folder):
        if os.path.isfile(OUTPUT_LOC + folder + file_name):
            count += 1
    assert count == map_count
