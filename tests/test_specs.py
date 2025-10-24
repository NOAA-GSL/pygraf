from pathlib import Path

import numpy as np
import yaml
from pytest import fixture, mark

from adb_graphics import specs, utils

yaml.add_constructor("!join_ranges", utils.join_ranges, Loader=yaml.Loader)


class Spec(specs.VarSpec):
    """
    Concrete class for the VarSpec abstract class.
    """

    with Path("adb_graphics/default_specs.yml").open() as c:
        cfg = yaml.load(c, Loader=yaml.Loader)

    @property
    def clevs(self):
        return np.asarray(range(15))

    @property
    def vspec(self):
        return {"cmap": "rainbow"}


@fixture
def spec():
    return Spec()


def test_aod_colors(spec):
    colors = spec.aod_colors
    assert len(colors) == 15


@mark.parametrize(("levels", "expected"), [(3, 3), (4, 4), (None, 16)])
def test_centered_diff(levels, expected, spec):
    colors = spec.centered_diff(nlev=levels)
    assert len(colors) == expected


@mark.parametrize(
    ("func", "expected"),
    [
        ("aod_colors", 15),
        ("cin_colors", 12),
        ("ceil_colors", 14),
        ("cldcov_colors", 13),
        ("cref_colors", 16),
        ("fire_power_colors", 7),
        ("flru_colors", 4),
        ("frzn_colors", 12),
        ("goes_colors", 151),
        ("graupel_colors", 19),
        ("hail_colors", 10),
        ("heat_flux_colors", 22),
        ("heat_flux_colors_g", 13),
        ("heat_flux_colors_l", 17),
        ("heat_flux_colors_s", 17),
        ("icprb_colors", 10),
        ("icsev_colors", 6),
        ("lcl_colors", 19),
        ("lifted_index_colors", 31),
        ("mdn_colors", 18),
        ("mean_vvel_colors", 19),
        ("mup_colors", 18),
        ("pbl_colors", 15),
        ("pcp_colors", 10),
        ("pcp_colors_high", 6),
        ("pmsl_colors", 15),
        ("ps_colors", 105),
        ("pw_colors", 21),
        ("radiation_colors", 27),
        ("radiation_bw_colors", 80),
        ("radiation_mix_colors", 130),
        ("rainbow11_colors", 12),
        ("rainbow12_colors", 13),
        ("rainbow12_reverse", 13),
        ("rainbow16_colors", 17),
        ("shear_colors", 10),
        ("slw_colors", 16),
        ("smoke_colors", 15),
        ("smoke_emissions_colors", 14),
        ("snow_colors", 13),
        ("soilm_colors", 12),
        ("soilw_colors", 13),
        ("t_colors", 15),
        ("tsfc_colors", 21),
        ("terrain_colors", 18),
        ("ua_temp_colors", 32),
        ("vis_colors", 142),
        ("vvel_colors", 12),
        ("vort_colors", 13),
        ("wind_colors", 19),
        ("wind_colors_high", 14),
    ],
)
def test_colors(expected, func, spec):
    colors = spec.__getattribute__(func)
    assert len(colors) == expected
