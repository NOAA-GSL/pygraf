"""Test suite for grib datahandler."""

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import colors as mcolors

from adb_graphics.datahandler import gribdata, gribfile

DATAARRAY = xr.core.dataarray.DataArray


def test_UPPData(natfile, prsfile):
    """Test the UPPData class methods on both types of input files."""

    nat_ds = gribfile.GribFile(natfile, var_config={})
    prs_ds = gribfile.GribFile(prsfile, var_config={})

    class UPP(gribdata.UPPData):
        """Test class needed to define the values as an abstract class."""

        def values(self, level=None, name=None, **kwargs):  # noqa: ARG002
            return 1

    upp_nat = UPP(nat_ds.contents, fhr=2, filetype="nat", short_name="temp", spec={})
    upp_prs = UPP(prs_ds.contents, fhr=2, short_name="temp", spec={})

    cycle = datetime(2025, 10, 2, 17)

    # Ensure appropriate typing and size (where applicable)
    for upp in [upp_nat, upp_prs]:
        assert isinstance(upp.anl_dt, datetime)
        assert isinstance(upp.clevs, np.ndarray)
        assert isinstance(upp.date_to_str(cycle), str)
        assert isinstance(upp.fhr, str)
        assert isinstance(upp.field, DATAARRAY)
        assert isinstance(upp.latlons(), list)
        assert isinstance(upp.lev_descriptor, str)
        assert isinstance(upp.numeric_level(), tuple)
        assert isinstance(upp.spec, dict)
        assert isinstance(upp.valid_dt, datetime)
        assert isinstance(upp.vspec, dict)
        # Test for appropriate date formatting
        test_date = datetime(2020, 12, 5, 12)
        assert upp.date_to_str(test_date) == "20201205 12 UTC"


def test_FieldData(prsfile):
    """Test the FieldData class methods on a prs file."""

    prs_ds = gribfile.GribFile(prsfile, var_config={})
    field = gribdata.FieldData(prs_ds.contents, fhr=2, level="500mb", short_name="temp")

    assert isinstance(field.cmap, mcolors.Colormap)
    assert isinstance(field.colors, np.ndarray)
    assert isinstance(field.corners, list)
    assert isinstance(field.ticks, int)
    assert isinstance(field.units, str)
    assert isinstance(field.values(), DATAARRAY)
    assert isinstance(field.aviation_flight_rules(field.values()), DATAARRAY)
    assert isinstance(field.wind(True), list)
    assert len(field.corners) == 4
    assert len(field.wind(True)) == 2
    assert len(field.wind("850mb")) == 2
    for component in field.wind(True):
        assert isinstance(component, DATAARRAY)

    # Test retrieving other values
    assert np.array_equal(field.values(), field.values(name="temp", level="500mb"))

    # Return zeros by subtracting same field
    diff = field.field_diff(field.values(), variable2="temp", level2="500mb")
    assert isinstance(diff, DATAARRAY)
    assert not np.any(diff)

    # Test transform
    assert np.array_equal(
        field.get_transform("conversions.k_to_f", field.values()),
        (field.values() - 273.15) * 9 / 5 + 32,
    )

    field2 = gribdata.FieldData(prs_ds.contents, fhr=2, level="ua", short_name="ceil")
    transforms = field2.vspec.get("transform")
    assert np.array_equal(
        field2.get_transform(transforms, field2.values()),
        field2.field_diff(field2.values(), variable2="gh", level2="sfc") / 304.8,
    )

    # Expected size of values
    assert len(np.shape(field.values())) == 2
    assert len(np.shape(field.values(name="u"))) == 2
    assert len(np.shape(field.values(name="u", level="850mb"))) == 2


def test_profile_data(natfile: Path):
    """Test the ProfileData class methods on a nat file."""

    nat_ds = gribfile.GribFile(natfile, var_config={})
    loc = " BNA   9999 99999  36.12  86.69  597 Nashville, TN\n"
    profile = gribdata.ProfileData(
        nat_ds.contents,
        fhr=2,
        filetype="nat",
        loc=loc,
        short_name="temp",
    )

    assert isinstance(profile.get_xypoint(40.0, -100.0), tuple)
    assert isinstance(profile.values(), DATAARRAY)

    # The values should return a single number (0) or a 1D array (1)
    assert len(np.shape(profile.values(level="best", name="li"))) == 0
    assert len(np.shape(profile.values(name="temp"))) == 1
