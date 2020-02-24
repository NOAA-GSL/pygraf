import argparse
import yaml

def main():
    pass

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
    parser.add_argument('-o', '--output_path',
                        help='Directory location desired for the output graphics files.',
                        required=True,
                        )
    parser.add_argument('-s', '--start_time',
                        help='Start time in YYYYMMDDHH format',
                        required=True,
                        )
    parser.add_argument('--subh_freq',
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


