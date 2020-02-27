import argparse
from string import digits
import yaml

def main(cla):

    # Load image list
    images = yaml.load(cla.image_list, loader=SafeLoader)[cla.image_set]

    # Locate grib file
    str_start_time = from_datetime(cla.start_time)
    grb_file = images[input_files][hrrr].format(FCST_TIME=cla.fcst_hour)
    grb_file = os.path.join(cla.data_root, str_start_time, grib_file)

    if not os.path.exists(grb_file):
        raise IOError(f"{grb_file} not found!")

    # Create working directory
    workdir = os.path.join(cla.data_root, str_start_time + f"{cla.fcst_hour:02d}")
    os.makedirs(workdir, exist_ok=True)

    # Load default specs configuration
    specs = yaml.load('adb_graphics.default_specs.yml', loader=SafeLoader)

    # Create plot for each figure in image list
    for variable, levels in images.get(variables).items():
        for level in levels:
            spec = specs.get(variable).get(level)

            # Strip all non-numbers from the level
            lev_num = 0 if level == 'sfc' else ''.join(c for c in level if c in digits)
            field = grib.UPPData(
                        filename=grb_file,
                        lev_type=specs.get('lev_type'),
                        short_name=specs.get('short_name'),
                        )




def to_datetime(string):
    return dt.strptime(string, '%Y%m%d%H')


def from_datetime(date):
    return dt.strftime(date, '%Y%m%d%H')

def webname(prefix, tile='', suffix=''):
    name = f"{prefix}{'_' + tile}{'_' + suffix}"
    return name

def parse_args():
    parser = argparse.ArgumentParser(description='Script to drive the creation of graphics.')

    parser.add_argument('-d', '--data_root',
                        help='Cycle-independant data directory location.',
                        required=True,
                        )
    parser.add_argument('-f', '--fcst_hour',
                        help='Forecast hour',
                        required=True,
                        )
    parser.add_argument('--image_list',
                        help='Path to YAML config file specifying which graphics to create.',
                        required=True,
                        )
    parser.add_argument('-m', '--image_set',
                        choices=['hourly'],
                        help='Name of top level key in image_list',
                        required=True,
                        )
    parser.add_argument('-o', '--output_path',
                        help='Directory location desired for the output graphics files.',
                        required=True,
                        )
    parser.add_argument('-s', '--start_time',
                        help='Start time in YYYYMMDDHH format',
                        required=True,
                        type=to_datetime,
                        )
    parser.add_argument('--subh_freq',
                        default=60,
                        help='Sub-hourly frequency in minutes.',
                        )
    parser.add_argument('-t', '--num_threads',
                        default=1,
                        help='Number of threads to use.',
                        )
    return parser.parse_args()

if __name__ == '__main__':
    clargs = parse_args()
    main(clargs)


