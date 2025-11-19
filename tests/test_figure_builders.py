import gc
import tracemalloc
from argparse import Namespace
from datetime import datetime
from unittest.mock import call, patch

import numpy as np
from pytest import fixture
from uwtools.api.config import get_yaml_config

from adb_graphics import figure_builders, utils
from adb_graphics.datahandler import gribdata


@fixture
def fielddata_obj(prsfile, spec):
    spec = get_yaml_config(spec)
    spec.dereference(context={"file_type": "prs"})
    return gribdata.FieldData(
        model="hrrr",
        fhr=16,
        grib_paths=[prsfile],
        level="sfc",
        short_name="cref",
        spec=spec,
    )


@fixture
def parallel_maps_args(prsfile, spec, tmp_path):
    spec = get_yaml_config(spec)
    spec.dereference(context={"file_type": "prs"})
    cla = Namespace(
        **{  # noqa: PIE804
            "ens_size": 0,
            "graphic_type": "maps",
            "img_res": 72,
            "model_name": "hrrr",
            "specs": spec,
            "images": ["hrrr", []],
        }
    )
    return {
        "cla": cla,
        "fhr": 16,
        "grib_paths": [prsfile],
        "level": "sfc",
        "variable": "temp",
        "workdir": tmp_path,
    }


@fixture
def parallel_skewt_args(natfile, spec, tmp_path):
    spec = get_yaml_config(spec)
    spec.dereference(context={"file_type": "nat"})
    cla = Namespace(
        **{  # noqa: PIE804
            "file_type": "nat",
            "img_res": 72,
            "max_plev": 100,
            "model_name": "hrrr",
            "start_time": datetime(2025, 10, 6, 0),
            "specs": spec,
            "images": ["hrrr", []],
        }
    )
    return {
        "cla": cla,
        "fhr": 16,
        "grib_paths": [natfile],
        "site": " DNR  23062 72469  39.77 104.88 1611 Denver, CO",
        "workdir": tmp_path,
    }


@fixture(scope="module")
def spec(spec_file):
    spec = utils.load_yaml(spec_file)
    spec.dereference(context={"fhr": 16})
    return spec


def test_add_obs_panel(fielddata_obj, spec):
    spec = get_yaml_config(spec)
    spec.dereference(context={"file_type": "nat"})
    fig, ax = figure_builders.set_figure("hrrr", "enspanel", "full")
    # Overwriting this explicitly since the cfgrib should indefinitely come from the model data.
    spec["cref"]["obs"]["cfgrib"] = spec["1ref"]["1000m"]["cfgrib"]["hrrr"]
    args = {
        "ax": ax[8],
        "model_name": "hrrr",
        "obs_file": fielddata_obj.grib_paths[0],  # fake it with model data
        "proj_info": fielddata_obj.grid_info(),
        "spec": spec,
        "short_name": "cref",
        "tile": "full",
    }
    dm = figure_builders.add_obs_panel(**args)
    assert dm.figure == fig
    assert np.array_equal(dm.levels, np.arange(5, 76, 5))


def test_parallel_maps(parallel_maps_args, tmp_path):
    figure_builders.parallel_maps(**parallel_maps_args)
    assert (tmp_path / "temp_full_sfc_f016.png").is_file()


def test_parallel_maps_enspanel(parallel_maps_args, tmp_path):
    parallel_maps_args["cla"].ens_size = 9
    parallel_maps_args["cla"].graphic_type = "enspanel"
    parallel_maps_args["cla"].obs_file_path = tmp_path
    parallel_maps_args["cla"].specs["temp"]["sfc"]["include_obs"] = True

    with (
        patch.object(figure_builders, "MapFields") as fields,
        patch.object(figure_builders, "Map") as m,
        patch.object(figure_builders, "MultiPanelDataMap") as mpdm,
        patch.object(figure_builders, "add_obs_panel") as aop,
    ):
        mpdm_calls = [
            call(
                **{  # noqa: PIE804
                    "map_fields": fields(),
                    "map_": m(),
                    "member": mem,
                    "model_name": "hrrr",
                    "last_panel": mem == 9,
                }
            )
            for mem in [0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9]
        ]
        figure_builders.parallel_maps(**parallel_maps_args)
        assert mpdm.call_args_list == mpdm_calls
        call.title().assert_called_once()
        call.add_logo().assert_called_once()
        aop.assert_called_once()
        assert (tmp_path / "temp_full_sfc_f016.png").is_file()


def test_parallel_maps_mem_leak(parallel_maps_args):
    gc.collect()
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    figure_builders.parallel_maps(**parallel_maps_args)
    snapshot_after = tracemalloc.take_snapshot()
    gc.collect()
    tracemalloc.stop()
    # Compare memory usage
    stats_diff = snapshot_after.compare_to(snapshot_before, "lineno")
    total_diff_mb = sum(stat.size_diff for stat in stats_diff) / (1024 * 1024)
    assert total_diff_mb < 92  # Appropriate size when test was written


def test_parallel_skewt(parallel_skewt_args, tmp_path):
    figure_builders.parallel_skewt(**parallel_skewt_args)
    assert (tmp_path / "DNR_72469_skewt_f016.png").is_file()
    assert (tmp_path / "DNR.72469.skewt.2025100600_f016.csv").is_file()


def test_set_figure_enspanel_full():
    fig, ax = figure_builders.set_figure("hrrr", "enspanel", "full")
    assert len(ax) == 12
    assert list(fig.get_size_inches()) == [20.0, 10.0]


def test_set_figure_enspanel_other():
    fig, ax = figure_builders.set_figure("hrrr", "enspanel", "other")
    assert len(ax) == 12
    assert list(fig.get_size_inches()) == [20.0, 16.0]


def test_set_figure_enspanel_se():
    fig, ax = figure_builders.set_figure("hrrr", "enspanel", "SE")
    assert len(ax) == 12
    assert list(fig.get_size_inches()) == [20.0, 19.0]


def test_set_figure_maps_full():
    fig, ax = figure_builders.set_figure("hrrr", "maps", "full")
    assert len(ax) == 1
    assert list(fig.get_size_inches()) == [10.0, 10.0]
