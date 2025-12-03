"""Tests for create_graphics driver."""

import os

from pytest import fixture

from create_graphics import create_graphics, parse_args

DATA_LOC = os.environ.get("DATA_LOC")


@fixture
def maps_args(tmp_path) -> list:
    """Builds HRRR 12-hour accumulated maps."""
    return [
        "maps",
        "-a",
        "1",
        "-d",
        DATA_LOC,
        "-f",
        "0",
        "6",
        "1",
        "-o",
        str(tmp_path / "output"),
        "-s",
        "2023031500",
        "--file_tmpl",
        "hrrr.t00z.wrfprsf{FCST_TIME:02d}.grib2",
        "--images",
        "./image_lists/hrrr_test.yml",
        "hourly",
        "--all_leads",
        "--file_type=prs",
    ]


def test_hrrr_maps_parse_args(tmp_path):
    """
    Test parse_args for basic parsing success.
    Checks if parse_args returns 'maps' in the graphic_type field.
    """
    args = [
        "maps",
        "-d",
        DATA_LOC,
        "-f",
        "0",
        "12",
        "1",
        "-o",
        str(tmp_path / "output"),
        "-s",
        "2021052315",
        "--file_tmpl",
        "hrrr.t00z.wrfprsf{FCST_TIME:02d}.grib2",
        "--images",
        "./image_lists/hrrr_test.yml",
        "hourly",
        "--all_leads",
        "--file_type=prs",
    ]
    test_args = parse_args(args)
    assert test_args.graphic_type == "maps"


def test_hrrr_maps_file_count(maps_args, tmp_path):
    """
    Test for file count in directory.
    Can be extended to cover multiple folders.
    """
    # Based on the hrrr_test.yml file, only 6 maps will be created
    create_graphics(maps_args)
    map_count = 6
    count = 0
    output = tmp_path / "output" / "202303150003"
    for file_name in output.iterdir():
        if (output / file_name).is_file():
            count += 1
    assert count == map_count
