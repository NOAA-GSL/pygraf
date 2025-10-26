from datetime import datetime

import numpy as np
from pytest import fixture, mark
from xarray import DataArray, ones_like, zeros_like

from adb_graphics import utils
from adb_graphics.datahandler import gribdata, gribfile


class ConcreteUPPData(gribdata.UPPData):
    def values(self, level: str | None = None, name: str | None = None, **kwargs) -> DataArray:  # noqa: ARG002
        return self.ds.to_dataarray().squeeze()


@fixture
def hrrr_data(prsfile):
    return gribfile.GribFile(
        prsfile,
        var_config={
            "shortName": "t",
            "typeOfLevel": "isobaricInhPa",
        },
    ).contents


@fixture
def spec(spec_file):
    return utils.load_yaml(spec_file)


@fixture
def uppdata_obj(hrrr_data, prsfile, spec):
    return ConcreteUPPData(
        ds=hrrr_data,
        short_name="temp",
        spec=spec,
        fhr=15,
        grib_path=prsfile,
    )


@fixture
def uppdata_multilev_obj(prsfile, spec):
    ds = gribfile.GribFile(
        prsfile,
        var_config={
            "shortName": "t",
            "typeOfLevel": "isobaricInhPa",
        },
    ).contents
    return ConcreteUPPData(
        ds=ds,
        short_name="temp",
        spec=spec,
        fhr=15,
    )


def test_uppdata_anl_dt(uppdata_obj):
    dt = uppdata_obj.anl_dt
    assert dt == datetime(2020, 10, 9, 8)


def test_uppdata_clevs_array(uppdata_obj):
    assert np.array_equal(uppdata_obj.clevs, np.arange(-40, 40, 2.5))


def test_uppdata_clevs_list(uppdata_obj):
    uppdata_obj.spec["temp"]["ua"]["clevs"] = [1, 2, 3]
    assert np.array_equal(uppdata_obj.clevs, np.asarray([1, 2, 3]))


def test_uppdata_date_to_str(uppdata_obj):
    assert uppdata_obj.date_to_str(uppdata_obj.anl_dt) == "20201009 08 UTC"


def test_uppdata_field(uppdata_obj):
    assert np.array_equal(uppdata_obj.field, uppdata_obj.ds.t)


def test_uppdata_field_column_max(uppdata_multilev_obj):
    assert np.array_equal(
        uppdata_multilev_obj.field_column_max(), uppdata_multilev_obj.ds.t.max(axis=0)
    )
    assert uppdata_multilev_obj.field_column_max().shape == (1059, 1799)


def test_uppdata_field_diff(uppdata_obj):
    summed_field = uppdata_obj.field_diff(values=uppdata_obj.field, variable2="temp", level2="sfc")
    assert np.array_equal(summed_field, uppdata_obj.ds.t * 0)


def test_uppdata_field_mean(uppdata_multilev_obj):
    levels = ["500mb", "800mb"]
    mean = uppdata_multilev_obj.field_mean(values=uppdata_multilev_obj.field, levels=levels)
    assert np.array_equal(
        mean, uppdata_multilev_obj.ds.t.sel(isobaricInhPa=[500, 800]).mean("isobaricInhPa")
    )
    assert mean.shape == (1059, 1799)


def test_uppdata_field_sum(uppdata_obj):
    summed_field = uppdata_obj.field_sum(values=uppdata_obj.field, variable2="temp", level2="sfc")
    assert np.array_equal(summed_field, uppdata_obj.ds.t * 2)


def test_uppdata__get_data_levels(uppdata_multilev_obj):
    assert np.array_equal(
        uppdata_multilev_obj._get_data_levels("isobaricInhPa"),
        uppdata_multilev_obj.ds.coords["isobaricInhPa"].to_numpy(),
    )


def test_uppdata__get_field(prsfile, uppdata_obj):
    spec = {"shortName": "t", "typeOfLevel": "isobaricInhPa", "level": 500}
    field = uppdata_obj._get_field(spec=spec)
    ds = gribfile.GribFile(
        prsfile,
        var_config=spec,
    ).contents
    assert np.array_equal(field, ds.t)


@mark.parametrize(
    "transforms",
    [
        "conversions.percent",
        ["conversions.percent", "opposite"],
        {"funcs": "field_diff", "kwargs": {"variable2": "temp", "level2": "sfc"}},
    ],
)
def test_uppdata_get_transform(transforms, uppdata_obj):
    val = ones_like(uppdata_obj.ds.t) if not isinstance(transforms, dict) else uppdata_obj.ds.t
    field = uppdata_obj.get_transform(transforms, val)
    expected = 0
    match transforms:
        case dict():
            expected = zeros_like(uppdata_obj.ds.t)
        case list():
            expected = val * -100.0
        case str():
            expected = val * 100.0
    assert np.array_equal(field, expected)


@mark.parametrize(
    ("lat", "lon", "expected"),
    [(40.019, 360 - 105.2747, (595, 679)), (25.7617, 360 - 80.1918, (109, 1487))],
)
def test_uppdata_get_xypoint(expected, lat, lon, uppdata_obj):
    assert uppdata_obj.get_xypoint(lat, lon) == expected


@mark.parametrize(("lat", "lon"), [(88.0, 270.0), (40, 180), (10, 330), (30, 345)])
def test_uppdata_get_xypoint_outside(lat, lon, uppdata_obj):
    assert uppdata_obj.get_xypoint(lat, lon) == (-1, -1)


def test_uppdata_latlons(uppdata_obj):
    lats = uppdata_obj.ds.coords["latitude"].to_numpy()
    lons = uppdata_obj.ds.coords["longitude"].to_numpy()
    assert [
        np.array_equal(act, exp)
        for act, exp in zip(uppdata_obj.latlons(), [lats, lons], strict=True)
    ]


@mark.parametrize("factor", [1, -1, 0, -20.0, 6543.0])
def test_uppdata_opposite(factor, uppdata_obj):
    ds = ones_like(uppdata_obj.field) * factor
    assert np.array_equal(uppdata_obj.opposite(ds), -ds)


def test_uppdata_valid_dt(uppdata_obj):
    assert uppdata_obj.valid_dt == datetime(2020, 10, 9, 23)


def test_uppdata_vspec(uppdata_obj):
    expected = {
        "cfgrib": {"shortName": "t", "typeOfLevel": "isobaricInhPa"},
        "clevs": np.arange(-40, 40, 2.5),
        "cmap": "jet",
        "colors": "ua_temp_colors",
        "contours": {
            "pres_sfc": {"levels": [0, 500], "colors": "k", "linewidths": 0.6},
            "gh": {"colors": "grey"},
        },
        "hatches": {"pres_sfc": {"hatches": ["", "..."], "levels": [0, 500]}},
        "ncl_name": {"prs": "TMP_P0_L100_{grid}", "nat": "TMP_P0_L105_{grid}"},
        "ticks": 5,
        "transform": "conversions.k_to_c",
        "unit": "C",
        "wind": True,
    }
    vspec = uppdata_obj.vspec
    # Can't test the array items with ==, so check them separately and then remove.
    assert np.array_equal(vspec["clevs"], expected["clevs"])
    vspec.pop("clevs")
    expected.pop("clevs")
    assert uppdata_obj.vspec == expected
