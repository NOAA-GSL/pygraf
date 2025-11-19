from datetime import datetime

import numpy as np
from matplotlib.pyplot import get_cmap
from pytest import fixture, mark, raises
from uwtools.api.config import get_yaml_config
from xarray import DataArray, ones_like, zeros_like

from adb_graphics import errors, utils
from adb_graphics.datahandler import gribdata


class ConcreteUPPData(gribdata.UPPData):
    def values(
        self,
        level: str | None = None,  # noqa: ARG002
        name: str | None = None,  # noqa: ARG002
        do_transform: bool = True,  # noqa: ARG002
    ) -> DataArray:
        return self.ds.to_dataarray().squeeze()  # type: ignore[no-any-return]


@fixture
def fielddata_obj(prsfile, spec):
    spec = get_yaml_config(spec)
    spec.dereference(context={"file_type": "prs"})
    return gribdata.FieldData(
        fhr=16,
        grib_paths=[prsfile],
        level="sfc",
        model="hrrr",
        short_name="temp",
        spec=spec,
    )


@fixture(scope="module")
def fielddata_obj_ro(prsfile, spec):
    spec = get_yaml_config(spec)
    spec.dereference(context={"file_type": "prs"})
    return gribdata.FieldData(
        fhr=16,
        grib_paths=[prsfile],
        level="sfc",
        model="hrrr",
        short_name="temp",
        spec=spec,
    )


@fixture(scope="module")
def profiledata_obj(natfile, spec):
    spec = get_yaml_config(spec)
    spec.dereference(context={"file_type": "nat"})
    return gribdata.ProfileData(
        fhr=16,
        grib_paths=[natfile],
        loc=" DNR  23062 72469  39.77 104.88 1611 Denver, CO",
        model="hrrr",
        short_name="temp",
        spec=spec,
    )


@fixture(scope="module")
def spec(spec_file):
    spec = utils.load_yaml(spec_file)
    spec.dereference(context={"fhr": 16})
    return spec


@fixture
def uppdata_obj(natfile, spec):
    spec = get_yaml_config(spec)
    spec.dereference(context={"file_type": "nat"})
    return ConcreteUPPData(
        level="ua",
        model="hrrr",
        short_name="temp",
        spec=spec,
        fhr=16,
        grib_paths=[natfile],
    )


@fixture(scope="module")
def uppdata_obj_ro(natfile, spec):
    spec = get_yaml_config(spec)
    spec.dereference(context={"file_type": "nat"})
    return ConcreteUPPData(
        level="ua",
        model="hrrr",
        short_name="temp",
        spec=spec,
        fhr=16,
        grib_paths=[natfile],
    )


@fixture
def uppdata_multilev_obj(natfile, spec):
    spec = get_yaml_config(spec)
    spec.dereference(context={"file_type": "nat"})
    return ConcreteUPPData(
        model="hrrr",
        short_name="temp",
        spec=spec,
        fhr=16,
        grib_paths=[natfile],
    )


@fixture
def uppdata_multilev_prs_obj(prsfile, spec):
    spec = get_yaml_config(spec)
    spec.dereference(context={"file_type": "prs"})
    return ConcreteUPPData(
        level="500mb",
        model="hrrr",
        short_name="temp",
        spec=spec,
        fhr=16,
        grib_paths=[prsfile],
    )


def test_uppdata_anl_dt(uppdata_obj_ro):
    dt = uppdata_obj_ro.anl_dt
    assert dt == datetime(2025, 10, 6, 0)


def test_uppdata_clevs_array(uppdata_obj_ro):
    assert np.array_equal(uppdata_obj_ro.clevs, np.arange(-40, 40, 2.5))


def test_uppdata_clevs_list(uppdata_obj):
    uppdata_obj.spec["temp"]["ua"]["clevs"] = [1, 2, 3]
    assert np.array_equal(uppdata_obj.clevs, np.asarray([1, 2, 3]))


def test_uppdata_date_to_str(uppdata_obj_ro):
    assert uppdata_obj_ro.date_to_str(uppdata_obj_ro.anl_dt) == "20251006 00 UTC"


def test_uppdata_field(uppdata_obj_ro):
    assert np.array_equal(uppdata_obj_ro.field, uppdata_obj_ro.ds.t.squeeze())


def test_uppdata_field_column_max(prsfile, spec):
    spec = get_yaml_config(spec)
    spec.dereference(context={"file_type": "prs"})
    fd = gribdata.FieldData(
        fhr=16,
        grib_paths=[prsfile],
        level="ua",
        model="hrrr",
        short_name="temp",
        spec=spec,
    )
    column_max = fd.field_column_max(values=fd.ds.t)
    assert np.array_equal(column_max, fd.ds.t.squeeze().max(axis=0))
    assert column_max.shape == (1059, 1799)


def test_uppdata_field_diff(fielddata_obj_ro):
    summed_field = fielddata_obj_ro.field_diff(
        values=fielddata_obj_ro.field, variable2="temp", level2="sfc", do_transform=False
    )
    assert np.array_equal(summed_field, fielddata_obj_ro.ds.t.squeeze() * 0)


def test_uppdata_field_mean(prsfile, spec):
    dataobj = gribdata.FieldData(
        level="mean",
        model="hrrr",
        short_name="rh",
        spec=spec,
        fhr=16,
        grib_paths=[prsfile],
    )
    levels = ["500mb", "800mb"]
    mean = dataobj.field_mean(values=dataobj.field, levels=levels)
    assert np.array_equal(
        mean, dataobj.ds.r.squeeze().sel(isobaricInhPa=[500, 800]).mean("isobaricInhPa")
    )
    assert mean.shape == (1059, 1799)


def test_uppdata_field_sum(fielddata_obj_ro):
    summed_field = fielddata_obj_ro.field_sum(
        values=fielddata_obj_ro.field, variable2="temp", level2="sfc", do_transform=False
    )
    assert np.array_equal(summed_field, fielddata_obj_ro.ds.t.squeeze() * 2)


def test_uppdata__get_data_levels(uppdata_multilev_prs_obj):
    assert np.array_equal(
        uppdata_multilev_prs_obj._get_data_levels("isobaricInhPa"),
        uppdata_multilev_prs_obj.ds.coords["isobaricInhPa"].to_numpy(),
    )


def test_uppdata__get_field(uppdata_multilev_prs_obj):
    spec = {"shortName": "t", "typeOfLevel": "isobaricInhPa", "level": 500}
    field = uppdata_multilev_prs_obj._get_field(cfgribspec=spec)
    assert np.array_equal(field, uppdata_multilev_prs_obj.ds.t.squeeze())


@mark.parametrize(
    "transforms",
    [
        "conversions.percent",
        ["conversions.percent", "opposite"],
        {
            "funcs": "field_diff",
            "kwargs": {"variable2": "temp", "level2": "sfc", "do_transform": False},
        },
    ],
)
def test_uppdata_get_transform(fielddata_obj_ro, transforms):
    temp = fielddata_obj_ro.ds.t
    val = ones_like(temp) if not isinstance(transforms, dict) else temp
    field = fielddata_obj_ro.get_transform(transforms, val)
    expected = 0
    match transforms:
        case dict():
            expected = zeros_like(temp)
        case list():
            expected = val * -100.0
        case str():
            expected = val * 100.0
    assert np.array_equal(field, expected)


@mark.parametrize(
    ("lat", "lon", "expected"),
    [(40.019, 360 - 105.2747, (595, 679)), (25.7617, 360 - 80.1918, (109, 1487))],
)
def test_uppdata_get_xypoint(expected, lat, lon, uppdata_obj_ro):
    assert uppdata_obj_ro.get_xypoint(lat, lon) == expected


@mark.parametrize(("lat", "lon"), [(88.0, 270.0), (40, 180), (10, 330), (30, 345)])
def test_uppdata_get_xypoint_outside(lat, lon, uppdata_obj_ro):
    assert uppdata_obj_ro.get_xypoint(lat, lon) == (-1, -1)


def test_uppdata_latlons(uppdata_obj_ro):
    lats = uppdata_obj_ro.ds.coords["latitude"].to_numpy()
    lons = uppdata_obj_ro.ds.coords["longitude"].to_numpy()
    assert [
        np.array_equal(act, exp)
        for act, exp in zip(uppdata_obj_ro.latlons(), [lats, lons], strict=True)
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
def test_uppdata_opposite(factor, uppdata_obj_ro):
    ds = ones_like(uppdata_obj_ro.field) * factor
    assert np.array_equal(uppdata_obj_ro.opposite(ds), -ds)


def test_uppdata_valid_dt(uppdata_obj_ro):
    assert uppdata_obj_ro.valid_dt == datetime(2025, 10, 6, 16)


def test_uppdata_vector_magnitude(prsfile, spec):
    fd = ConcreteUPPData(
        model="hrrr",
        level="250mb",
        fhr=16,
        grib_paths=[prsfile],
        short_name="u",
        spec=spec,
    )
    vm = fd.vector_magnitude(field1=fd.ds.u, field2_id="v_250mb")
    assert not np.array_equal(vm, fd.ds.u)


def test_uppdata_vspec(uppdata_obj_ro):
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
    vspec = uppdata_obj_ro.vspec
    # Can't test the array items with ==, so check them separately.
    actual_clevs = vspec.pop("clevs")
    expected_clevs = expected.pop("clevs")
    assert np.array_equal(actual_clevs, np.asarray(expected_clevs))
    assert uppdata_obj_ro.vspec == expected


def test_uppdata_vspec_bad(uppdata_obj):
    uppdata_obj.short_name = "foo"
    with raises(errors.NoGraphicsDefinitionForVariableError):
        uppdata_obj.vspec  # noqa: B018


def test_fielddata_aviation_flight_rules(prsfile, spec):
    fd = gribdata.FieldData(
        fhr=16,
        grib_paths=[prsfile],
        level="sfc",
        model="hrrr",
        short_name="flru",
        spec=spec,
    )

    flru = fd.aviation_flight_rules(fd.field)
    assert flru.max() == 3.01
    assert flru.min() == 0.0


def test_fielddata_cmap(fielddata_obj_ro):
    assert fielddata_obj_ro.cmap == get_cmap("gist_ncar")


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
        21.13812299999999,
        47.84219502248864,
        237.28047200000003,
        299.08280722816215,
    ]


def test_fielddata_corners_single_dim(fielddata_obj):
    # Remove one dimension for the purposes of the test
    fielddata_obj.ds.coords["latitude"] = fielddata_obj.ds.coords["latitude"][:, 0]
    assert fielddata_obj.corners == [
        21.13812299999999,
        47.83862349881542,
        237.28047200000003,
        225.90452026573686,
    ]


def test_fielddata_data_getter_and_setter(fielddata_obj):
    assert np.array_equal(fielddata_obj.data, fielddata_obj.values())
    new_data = ones_like(fielddata_obj.ds.t)
    fielddata_obj.data = new_data
    assert np.array_equal(fielddata_obj.data, new_data)


def test_fielddata_fire_weather_index(prsfile, spec):
    fd = gribdata.FieldData(
        fhr=16,
        grib_paths=[prsfile],
        level="sfc",
        model="hrrr",
        short_name="firewxtransform",
        spec=spec,
    )

    firewx = fd.fire_weather_index(fd.field)
    assert firewx.max() <= 100
    assert firewx.min() == 0


def test_fielddata_grid_info_lambert(fielddata_obj_ro):
    grid_info = fielddata_obj_ro.grid_info()
    assert grid_info == {
        "corners": [21.13812299999999, 47.84219502248864, 237.28047200000003, 299.08280722816215],
        "lat_0": 39.0,
        "lat_1": 38.5,
        "lat_2": 38.5,
        "lon_0": 262.5,
        "projection": "lcc",
    }


def test_fielddata_icing_adjust_trace(prsfile, spec):
    spec = get_yaml_config(spec)
    spec.dereference(context={"file_type": "prs"})
    fd = gribdata.FieldData(
        fhr=16,
        grib_paths=[prsfile],
        level="sfc",
        model="hrrr",
        short_name="flru",
        spec=spec,
    )
    field = ones_like(fd.field) * 4
    icing_adjust_trace = fd.icing_adjust_trace(field)
    assert np.array_equal(icing_adjust_trace, ones_like(field) * 0.5)


def test_fielddata_supercooled_liquid_water(natfile, spec):
    spec = get_yaml_config(spec)
    spec.dereference(context={"file_type": "nat"})
    fd = gribdata.FieldData(
        fhr=16,
        grib_paths=[natfile],
        level="sfc",
        model="hrrr",
        short_name="slw",
        spec=spec,
    )
    slw = fd.supercooled_liquid_water()
    assert not np.array_equal(slw, fd.ds.t.squeeze())


def test_fielddata_ticks_default(fielddata_obj_ro):
    assert fielddata_obj_ro.ticks == 10


def test_fielddata_ticks_in_vspec(fielddata_obj):
    ticks = 22
    fielddata_obj.vspec["ticks"] = ticks
    assert fielddata_obj.ticks == ticks


def test_fielddata_units_default(fielddata_obj_ro):
    assert fielddata_obj_ro.units == "F"


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
    assert not np.array_equal(
        fielddata_obj.values(level=lev, name=var), fielddata_obj.ds.t.squeeze()
    )


def test_fielddata_values_args_transform(fielddata_obj):
    fielddata_obj.vspec["transform"] = "opposite"
    fielddata_obj.model = "hrrr"
    assert np.array_equal(
        fielddata_obj.values(level="sfc", name="temp"), -fielddata_obj.ds.t.squeeze()
    )


def test_fielddata_values_no_args_no_transform(fielddata_obj):
    field = ones_like(fielddata_obj.ds)
    fielddata_obj.ds = field
    fielddata_obj.vspec["transform"] = None
    assert np.array_equal(fielddata_obj.values(), field.t.squeeze())


def test_fielddata_values_no_args_transform(fielddata_obj):
    field = ones_like(fielddata_obj.ds)
    fielddata_obj.ds = field
    fielddata_obj.vspec["transform"] = "opposite"
    assert np.array_equal(fielddata_obj.values(), -field.t.squeeze())


def test_fielddata_values_bad_name_level(fielddata_obj_ro):
    with raises(errors.NoGraphicsDefinitionForVariableError):
        fielddata_obj_ro.values(level="foo", name="temp")
    with raises(errors.NoGraphicsDefinitionForVariableError):
        fielddata_obj_ro.values(level="sfc", name="foo")
    with raises(errors.NoGraphicsDefinitionForVariableError):
        fielddata_obj_ro.values(level="bar", name="foo")


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
