from pathlib import Path

from pytest import mark
from xarray import Dataset

from adb_graphics.datahandler import gribfile


def test_gribfile(prsfile):
    gf = gribfile.GribFile(
        filename=Path(prsfile),
        cfgrib_config={
            "shortName": "sp",
            "typeOfLevel": "surface",
        },
    )
    assert isinstance(gf.contents, Dataset)
    assert len(gf.contents.data_vars) == 1
    assert len(gf.contents.data_vars["sp"].shape) == 2


@mark.skip(reason="This test requires test data that is not yet available.")
def test_gribfiles():
    paths = [
        "/Users/cholt/work/pygraf_cfgrib/sample_data/rrfs_a/2025101312/rrfs.t12z.prslev.3km.f016.conus.grib2",
        "/Users/cholt/work/pygraf_cfgrib/sample_data/rrfs_a/rrfs.t12z.prslev.3km.f016.conus.grib2",
    ]
    gribfiles = [Path(f) for f in paths]
    gf = gribfile.GribFiles(
        filenames=gribfiles,
        cfgrib_config={
            "shortName": "sp",
            "typeOfLevel": "surface",
        },
        model="rrfs",
    )
    assert isinstance(gf.contents, Dataset)
    assert len(gf.contents.data_vars) == 1
    assert len(gf.contents.data_vars["sp"].shape) == 3
