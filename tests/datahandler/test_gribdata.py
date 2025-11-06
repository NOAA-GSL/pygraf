from datetime import datetime

import numpy as np
from matplotlib.pyplot import get_cmap
from pytest import fixture, mark, raises
from xarray import DataArray, ones_like, zeros_like

from adb_graphics import errors, utils
from adb_graphics.datahandler import gribdata, gribfile


class ConcreteUPPData(gribdata.UPPData):
    def values(self, level: str | None = None, name: str | None = None, **kwargs) -> DataArray:  # noqa: ARG002
        return self.ds.to_dataarray().squeeze()


@fixture
def hrrr_data(prsfile):
    return gribfile.GribFile(
        prsfile,
        cfgrib_config={
            "shortName": "t",
            "typeOfLevel": "surface",
        },
    ).contents


@fixture
def fielddata_obj(hrrr_data, prsfile, spec):
    return gribdata.FieldData(
        ds=hrrr_data,
        fhr=15,
        grib_path=prsfile,
        level="sfc",
        short_name="temp",
        spec=spec,
    )


@fixture
def profiledata_obj(natfile, spec):
    ds = gribfile.GribFile(
        natfile,
        cfgrib_config={
            "shortName": "t",
            "typeOfLevel": "hybrid",
        },
    )
    return gribdata.ProfileData(
        ds=ds.contents,
        fhr=15,
        grib_path=natfile,
        loc=" DNR  23062 72469  39.77 104.88 1611 Denver, CO",
        short_name="temp",
        spec=spec,
    )


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
        cfgrib_config={
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
    assert dt == datetime(2025, 10, 6, 0)


def test_uppdata_clevs_array(uppdata_obj):
    assert np.array_equal(uppdata_obj.clevs, np.arange(-40, 40, 2.5))


def test_uppdata_clevs_list(uppdata_obj):
    uppdata_obj.spec["temp"]["ua"]["clevs"] = [1, 2, 3]
    assert np.array_equal(uppdata_obj.clevs, np.asarray([1, 2, 3]))


def test_uppdata_date_to_str(uppdata_obj):
    assert uppdata_obj.date_to_str(uppdata_obj.anl_dt) == "20251006 00 UTC"


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
        cfgrib_config=spec,
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


def test_uppdata_latlons_lats_flipped(uppdata_obj):
    # Test a 1D latitude option (like in Global, etc.)
    ds = uppdata_obj.ds.sel(y=500)
    lats = ds.coords["latitude"].to_numpy()
    ds.coords["latitude"] = (("x"), lats[::-1])
    lons = ds.coords["longitude"].to_numpy()
    uppdata_obj.ds = ds
    assert [
        np.array_equal(act, exp)
        for act, exp in zip(uppdata_obj.latlons(), [lats, lons], strict=True)
    ]


@mark.parametrize("factor", [1, -1, 0, -20.0, 6543.0])
def test_uppdata_opposite(factor, uppdata_obj):
    ds = ones_like(uppdata_obj.field) * factor
    assert np.array_equal(uppdata_obj.opposite(ds), -ds)


def test_uppdata_valid_dt(uppdata_obj):
    assert uppdata_obj.valid_dt == datetime(2025, 10, 6, 15)


def test_uppdata_vector_magnitude(prsfile, spec):
    ds = gribfile.GribFile(
        prsfile,
        cfgrib_config={
            "shortName": "u",
            "typeOfLevel": "isobaricInhPa",
            "level": 250,
        },
    )
    fd = ConcreteUPPData(
        ds=ds.contents,
        fhr=15,
        grib_path=prsfile,
        level="250mb",
        short_name="u",
        spec=spec,
    )
    vm = fd.vector_magnitude(field1=fd.ds.u, field2_id="v_250mb")
    assert not np.array_equal(vm, ds.contents.u)


def test_uppdata_vspec(uppdata_obj):
    expected = {
        "cfgrib": {"shortName": "t", "typeOfLevel": "hybrid"},
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
    assert np.array_equal(vspec["clevs"], np.asarray(expected["clevs"]))
    vspec.pop("clevs")
    expected.pop("clevs")
    assert uppdata_obj.vspec == expected


def test_uppdata_vspec_bad(uppdata_obj):
    uppdata_obj.short_name = "foo"
    with raises(errors.NoGraphicsDefinitionForVariableError):
        uppdata_obj.vspec  # noqa: B018


def test_fielddata_aviation_flight_rules(prsfile, spec):
    ds = gribfile.GribFile(
        prsfile,
        cfgrib_config={
            "shortName": "gh",
            "typeOfLevel": "cloudCeiling",
        },
    )
    fd = gribdata.FieldData(
        ds=ds.contents,
        fhr=15,
        grib_path=prsfile,
        level="sfc",
        short_name="flru",
        spec=spec,
    )

    flru = fd.aviation_flight_rules(fd.field)
    assert flru.max() == 3.01
    assert flru.min() == 0.0


def test_fielddata_cmap(fielddata_obj):
    assert fielddata_obj.cmap == get_cmap("gist_ncar")


@mark.parametrize("color_def", ["aod_colors", "shear_colors", "vvel_colors"])
def test_fielddata_colors(color_def, fielddata_obj):
    fielddata_obj.vspec["colors"] = color_def
    assert fielddata_obj.colors.min() == 0.0
    assert fielddata_obj.colors.max() == 1.0


def test_fielddata_colors_undefined(fielddata_obj):
    del fielddata_obj.vspec["colors"]
    with raises(errors.NoGraphicsDefinitionForVariableError):
        fielddata_obj.colors  # noqa: B018


def test_fielddata_colors_bad(fielddata_obj):
    fielddata_obj.vspec["colors"] = "foo"
    with raises(AttributeError) as e:
        fielddata_obj.colors  # noqa: B018
    assert "There is no color definition named foo" in str(e.value)


def test_fielddata_corners(fielddata_obj):
    assert fielddata_obj.corners == [
        21.138123,
        47.842195,
        237.280472,
        299.082807,
    ]


def test_fielddata_corners_single_dim(fielddata_obj):
    # Remove one dimension for the purposes of the test
    fielddata_obj.ds.coords["latitude"] = fielddata_obj.ds.coords["latitude"][:, 0]
    assert fielddata_obj.corners == [
        21.138123,
        47.838623,
        237.280472,
        225.904520,
    ]


def test_fielddata_data_getter_and_setter(fielddata_obj):
    assert np.array_equal(fielddata_obj.data, fielddata_obj.values())
    new_data = ones_like(fielddata_obj.ds.t)
    fielddata_obj.data = new_data
    assert np.array_equal(fielddata_obj.data, new_data)


def test_fielddata_fire_weather_index(prsfile, spec):
    ds = gribfile.GribFile(
        prsfile,
        cfgrib_config={
            "shortName": "vgtyp",
            "typeOfLevel": "surface",
        },
    )
    fd = gribdata.FieldData(
        ds=ds.contents,
        fhr=15,
        grib_path=prsfile,
        level="sfc",
        short_name="firewxtransform",
        spec=spec,
    )

    firewx = fd.fire_weather_index(fd.field)
    assert firewx.max() <= 100
    assert firewx.min() == 0


def test_fielddata_grid_info_lambert(fielddata_obj):
    grid_info = fielddata_obj.grid_info()
    assert grid_info == {
        "corners": [21.138123, 47.842195, 237.280472, 299.082807],
        "lat_0": 39.0,
        "lat_1": 38.5,
        "lat_2": 38.5,
        "lon_0": 262.5,
        "projection": "lcc",
    }


def test_fielddata_icing_adjust_trace(prsfile, spec):
    ds = gribfile.GribFile(
        prsfile,
        cfgrib_config={
            "shortName": "gh",
            "typeOfLevel": "cloudCeiling",
        },
    )
    fd = gribdata.FieldData(
        ds=ds.contents,
        fhr=15,
        grib_path=prsfile,
        level="sfc",
        short_name="flru",
        spec=spec,
    )
    field = ones_like(fd.field) * 4
    icing_adjust_trace = fd.icing_adjust_trace(field)
    assert np.array_equal(icing_adjust_trace, ones_like(field) * 0.5)


def test_fielddata_supercooled_liquid_water(natfile, spec):
    ds = gribfile.GribFile(
        natfile,
        cfgrib_config={
            "shortName": "t",
            "typeOfLevel": "surface",
        },
    )
    fd = gribdata.FieldData(
        ds=ds.contents,
        fhr=15,
        grib_path=natfile,
        level="sfc",
        short_name="slw",
        spec=spec,
    )
    slw = fd.supercooled_liquid_water()
    assert not np.array_equal(slw, ds.contents.t)


def test_fielddata_ticks_default(fielddata_obj):
    assert fielddata_obj.ticks == 10


def test_fielddata_ticks_in_vspec(fielddata_obj):
    ticks = 22
    fielddata_obj.vspec["ticks"] = ticks
    assert fielddata_obj.ticks == ticks


def test_fielddata_units_default(fielddata_obj):
    assert fielddata_obj.units == "F"


def test_fielddata_units_in_vspec(fielddata_obj):
    units = "foo"
    fielddata_obj.vspec["unit"] = units
    assert fielddata_obj.units == units


@mark.parametrize(
    ("var", "lev"), [("pres", "sfc"), ("1ref", "1000m"), ("acsnw", "sfc"), ("rh", "500mb")]
)
def test_fielddata_values_args_no_transform(fielddata_obj, lev, var):
    fielddata_obj.vspec["transform"] = None
    fielddata_obj.model = "hrrr"
    assert not np.array_equal(fielddata_obj.values(level=lev, name=var), fielddata_obj.ds.t)


def test_fielddata_values_args_transform(fielddata_obj):
    fielddata_obj.vspec["transform"] = "opposite"
    fielddata_obj.model = "hrrr"
    assert np.array_equal(fielddata_obj.values(level="sfc", name="temp"), -fielddata_obj.ds.t)


def test_fielddata_values_no_args_no_transform(fielddata_obj):
    field = ones_like(fielddata_obj.ds)
    fielddata_obj.ds = field
    fielddata_obj.vspec["transform"] = None
    assert np.array_equal(fielddata_obj.values(), field.t)


def test_fielddata_values_no_args_transform(fielddata_obj):
    field = ones_like(fielddata_obj.ds)
    fielddata_obj.ds = field
    fielddata_obj.vspec["transform"] = "opposite"
    assert np.array_equal(fielddata_obj.values(), -field.t)


def test_fielddata_values_bad_name_level(fielddata_obj):
    with raises(errors.NoGraphicsDefinitionForVariableError):
        fielddata_obj.values(level="foo", name="temp")
    with raises(errors.NoGraphicsDefinitionForVariableError):
        fielddata_obj.values(level="sfc", name="foo")
    with raises(errors.NoGraphicsDefinitionForVariableError):
        fielddata_obj.values(level="bar", name="foo")


def test_profiledata_values(profiledata_obj):
    assert profiledata_obj.values().shape == (50,)


def test_profiledata_values_bad_name_level(profiledata_obj):
    with raises(errors.NoGraphicsDefinitionForVariableError):
        profiledata_obj.values(level="foo", name="temp")
    with raises(errors.NoGraphicsDefinitionForVariableError):
        profiledata_obj.values(level="sfc", name="foo")
    with raises(errors.NoGraphicsDefinitionForVariableError):
        profiledata_obj.values(level="bar", name="foo")


def test_profiledata_values_one_level(profiledata_obj):
    value = profiledata_obj.values(name="hlcy", level="sr01")
    assert value.shape == ()  # A single number
    assert value == 47.7
