import argparse
import yaml

def main():
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='Script to drive the creation of graphics.')

    parser.add_argument('-c', '--config',
                        help='Path to YAML config file specifying which graphics to create.',
                        )
    parser.add_argument('-d', '--input_data',
                        help='Path to input data file. Supported types: grib.',
                        )
    parser.add_argument('-o', '--output_path',
                        help='Directory location desired for the output graphics files.',
                        )
    return parser.parse_args()

if __name__ == '__main__':
    clargs = parse_args()
    main(clargs)
