#pylint: disable=unused-variable
''' Tests for create_graphics driver '''
from create_graphics import create_graphics
import os
import os.path

data_loc = os.environ.get("data_loc")
output_loc = os.environ.get("output_loc")

def test_parse_args():
    ''' Test function for existence of HRRR 12-hour accumulated maps. '''
    args = ['create_graphics', 'maps', '-d', str(data_loc), '-f', '0 12 1', '-o', str(output_loc),\
         '-s', '2021052315', '--file_tmpl', 'hrrr.t00z.wrfprsf{FCST_TIME:02d}.grib2', \
            '--images', 'image_lists/hrrr_test.yml', 'hourly', '--all_leads', '--file_type=prs']
    create_graphics(args)
    assert test_args.graphic_type == 'skewts'   # this of course will fail. Holdover from initial test. To be replaced.

def test_nonsense():
    assert 5 == 5

def test_existence():
    f0 = '/202303150100'
    path = output_loc + f0
    p0 = os.path.isdir(path)
    assert p0 == True