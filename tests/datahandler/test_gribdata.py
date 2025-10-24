from datetime import datetime

import numpy as np
from pytest import fixture
from xarray import DataArray

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
            "typeOfLevel": "surface",
        },
    ).contents


@fixture
def spec(spec_file):
    return utils.load_yaml(spec_file)


@fixture
def uppdata_obj(hrrr_data, spec):
    return ConcreteUPPData(
        ds=hrrr_data,
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


def test_uppdata_field_column_max(prsfile, spec):
    ds = gribfile.GribFile(
        prsfile,
        var_config={
            "shortName": "t",
            "typeOfLevel": "isobaricInhPa",
        },
    ).contents
    uppdata_obj = ConcreteUPPData(
        ds=ds,
        short_name="temp",
        spec=spec,
        fhr=15,
    )
    assert np.array_equal(uppdata_obj.field_column_max(), ds.t.max(axis=0))
    assert uppdata_obj.field_column_max().shape == (1059, 1799)


def test_uppdata_field_diff(uppdata_obj):
    summed_field = uppdata_obj.field_diff(values=uppdata_obj.field, variable2="temp", level2="sfc")
    assert np.array_equal(summed_field, uppdata_obj.ds.t * 0)


def test_uppdata_field_mean(prsfile, spec):
    ds = gribfile.GribFile(
        prsfile,
        var_config={
            "shortName": "t",
            "typeOfLevel": "isobaricInhPa",
        },
    ).contents
    uppdata_obj = ConcreteUPPData(
        ds=ds,
        short_name="temp",
        spec=spec,
        fhr=15,
    )
    levels = ["500mb", "800mb"]
    mean = uppdata_obj.field_mean(values=uppdata_obj.field, levels=levels)
    assert np.array_equal(mean, ds.t.sel(isobaricInhPa=[500, 800]).mean("isobaricInhPa"))
    assert mean.shape == (1059, 1799)


def test_uppdata_field_sum(uppdata_obj):
    summed_field = uppdata_obj.field_sum(values=uppdata_obj.field, variable2="temp", level2="sfc")
    assert np.array_equal(summed_field, uppdata_obj.ds.t * 2)
