# pylint: disable=invalid-name

"""
Pytests for the common utilities included in this package.

Includes:

    - conversions.py
    - specs.py
    - utils.py

To run the tests, type the following in the top level repo directory:

    python -m pytest --nat-file [path/to/gribfile] --prs-file [path/to/gribfile]

"""

import warnings
from inspect import getfullargspec
from string import ascii_letters, digits

import numpy as np
from matplotlib import cm
from matplotlib import colors as mcolors
from metpy.plots import ctables

from adb_graphics import specs, utils
from adb_graphics.datahandler import gribdata


class MockSpecs(specs.VarSpec):
    """Mock class for the VarSpec abstract class."""

    cfg = utils.load_yaml("adb_graphics/default_specs.yml")

    @property
    def clevs(self):
        return np.asarray(range(15))

    @property
    def vspec(self):
        return {}


def test_utils():
    """Test that utils works appropriately."""

    assert callable(utils.get_func("conversions.k_to_c"))


def test_join_ranges_constructor(tmp_path):
    """Test that the join_ranges constructor works as expected."""

    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("""
    foo: !join_ranges [[0, 15, 0.1], [20, 61, 20]]
    foo2: !join_ranges [[0, 15, 0.1]]
    foo3: !join_ranges [[0, 15, 0.1], [20, 40, 10], [40, 61, 20]]
    """)
    cfg = utils.load_yaml(cfg_file)

    expected = np.concatenate((np.arange(0, 15, 0.1), np.arange(20, 61, 20)), axis=0)
    expected2 = np.arange(0, 15, 0.1)
    expected3 = np.concatenate(
        (np.arange(0, 15, 0.1), np.arange(20, 40, 10), np.arange(40, 61, 20)), axis=0
    )

    assert np.array_equal(expected, cfg["foo"])
    assert np.array_equal(expected2, cfg["foo2"])
    assert np.array_equal(expected3, cfg["foo3"])


class TestDefaultSpecs:
    """Test contents of default_specs.yml."""

    config = "adb_graphics/default_specs.yml"
    varspec = MockSpecs()
    cfg = utils.load_yaml("adb_graphics/default_specs.yml")

    @property
    def allowable(self):
        """
        Each entry in the dict names a function that tests a key in
        default_specs.yml.
        """

        return {
            "accumulate": self.is_bool,
            "annotate": self.is_bool,
            "annotate_decimal": self.is_int,
            "clevs": self.is_a_clev,
            "cmap": self.is_a_cmap,
            "cfgrib": self.is_dict,
            "colors": self.is_a_color,
            "contours": self.is_a_contour_dict,
            "include_obs": self.is_bool,
            "hatches": self.is_a_contourf_dict,
            "labels": self.is_a_contourf_dict,
            "level": self.is_number,
            "ncl_name": True,
            "plot_airports": self.is_bool,
            "plot_scatter": self.is_bool,
            "print_units": True,
            "split": self.is_bool,
            "ticks": self.is_number,
            "title": self.is_string,
            "transform": self.check_transform,
            "unit": self.is_string,
            "vertical_index": self.is_int,
            "vertical_level_name": self.is_string,
            "wind": self.is_wind,
        }

    def check_kwargs(self, accepted_args, kwargs):
        """Ensure a dictionary entry matches the kwargs accepted by a function."""

        assert isinstance(kwargs, dict)

        for key, args in kwargs.items():
            lev = None
            if "_" in key:
                short_name, lev = key.split("_")
            else:
                short_name = key

            assert self.is_a_key(short_name)

            if lev:
                assert self.cfg.get(short_name, {}).get(lev) is not None

            for arg in args:
                assert arg in accepted_args

        return True

    def check_transform(self, entry):
        """
        Check structure of transform entry.

        The transform entry should be either a single transformation function, a list of
        transformation functions, or a dictionary containing the functions list and the kwargs list
        like so:

            transform:
              funcs: [list, of, functions]
              kwargs:
                first_arg: value
                sec_arg: value

        The functions listed under functions MUST be methods, not attributes!
        """

        kwargs = dict()

        # Check that each item listed is callable
        if isinstance(entry, (list, str)):
            assert self.is_callable(entry)

        # If the transform entry is a dictionary, check that it has the
        # appropriate contents
        elif isinstance(entry, dict):
            funcs = entry.get("funcs")
            assert funcs is not None

            # Make sure funcs is a list
            funcs = funcs if isinstance(funcs, list) else [funcs]

            # Key word arguments may not be present.
            kwargs = entry.get("kwargs", {})

            transforms = []
            for func in funcs:
                callables = self.get_callable(func)
                callables = callables if isinstance(callables, list) else [callables]
                transforms.extend(callables)

            # The argspecs bit gives us a list of all the accepted arguments
            # for the functions listed in the variable all_params. Test fails
            # when provided arguments don't appear in all_params.
            # arguments not in that list, we fail.
            if kwargs:
                argspecs = [getfullargspec(fx) for fx in transforms if callable(fx)]

                all_params: list = []
                for argspec in argspecs:
                    # Make sure all functions accept key word arguments
                    assert argspec.varkw is not None

                    parameters: list = []
                    for argtype in [argspec.args, argspec.varargs, argspec.varkw]:
                        if argtype is not None:
                            parameters.extend(argtype)
                    all_params.extend(parameters)

                for key in kwargs:
                    if key not in all_params:
                        msg = (
                            f"Function key {key} is not an explicit parameter"
                            f"in any of the transforms: {funcs}!"
                        )
                        warnings.warn(msg, UserWarning, stacklevel=2)

        return True

    # pylint: disable=inconsistent-return-statements
    def get_callable(self, func):
        """Return the callable function given a function name."""

        if func in dir(self.varspec):
            return self.varspec.__getattribute__(func)

        # Check datahandler.gribdata objects if a single word is provided
        if len(func.split(".")) == 1:
            # Check all the classes in the gribdata module
            for attr in dir(gribdata):
                # pylint: disable=no-member
                # Check the methods in each class
                if func in dir(gribdata.__getattribute__(attr)):
                    method = gribdata.__getattribute__(attr).__dict__.get(func)
                    if method is not None:
                        if isinstance(method, staticmethod):
                            return method.__func__
                        return method

        if callable(utils.get_func(func)):
            return utils.get_func(func)

        msg = f"{func} is not a known callable function!"
        raise ValueError(msg)

    @staticmethod
    def is_a_clev(clev):
        """Returns true for a clev that is a list, a range, or a callable function."""

        if isinstance(clev, (list, np.ndarray)):
            return True

        if "range" in clev.split("[")[0]:
            clean = lambda x: x.strip().split("-")[-1].replace(".", "1")
            items = clev.split(" ", 1)[1].strip("[").strip("]").split(",")
            nums = [clean(i).isnumeric() for i in items]
            return all(nums)

        return callable(utils.get_func(clev))

    @staticmethod
    def is_a_cmap(cmap):
        """Returns true for a cmap that is a Colormap object."""
        return cmap in dir(cm) + list(ctables.colortables.keys())

    def is_a_contour_dict(self, entry):
        """
        Set up the accepted arguments for plt.contour, and check the given
        arguments.
        """

        args = [
            "X",
            "Y",
            "Z",
            "levels",
            "corner_mask",
            "colors",
            "alpha",
            "cmap",
            "norm",
            "vmin",
            "vmax",
            "origin",
            "extent",
            "locator",
            "extend",
            "xunits",
            "yunits",
            "antialiased",
            "nchunk",
            "linewidths",
            "linestyles",
        ]

        if entry is None:
            return True

        return self.check_kwargs(args, entry)

    def is_a_contourf_dict(self, entry):
        """
        Set up the accepted arguments for plt.contourf, and check the given
        arguments.
        """

        args = [
            "X",
            "Y",
            "Z",
            "levels",
            "corner_mask",
            "colors",
            "alpha",
            "cmap",
            "labels",
            "norm",
            "vmin",
            "vmax",
            "origin",
            "extent",
            "locator",
            "extend",
            "xunits",
            "yunits",
            "antialiased",
            "nchunk",
            "linewidths",
            "hatches",
        ]

        if entry is None:
            return True

        return self.check_kwargs(args, entry)

    def is_a_color(self, color):
        """Returns true if color is contained in the list of recognized colors."""

        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS, **ctables.colortables)

        if color in colors:
            return True

        return color in dir(self.varspec)

    def is_a_dict(self, cfgrib):
        return isinstance(cfgrib, dict)

    @staticmethod
    def is_a_level(key):
        """
        Returns true if the key fits one of the level descriptor formats.

        Allowable formats include:

            [str_descriptor]     e.g. sfc, max, mup
            [numeric][lev_type]  e.g. 500mb, or 2m
            [stat][numeric]      e.g. mn02, mx25

        """

        allowed_levels = [
            "agl",  # above ground level
            "best",  # Best
            "bndylay",  # boundary layer cld cover
            "esbl",  # ???
            "esblmn",  # ???
            "high",  # high clouds
            "int",  # vertical integral
            "low",  # low clouds
            "max",  # maximum in column
            "maxsfc",  # max surface value
            "mdn",  # maximum downward
            "mid",  # mid-level clouds
            "mnsfc",  # min surface value
            "msl",  # mean sea level
            "mu",  # most unstable
            "mul",  # most unstable layer
            "mup",  # maximum upward
            "mu",  # most unstable
            "obs",  # observations
            "pw",  # wrt precipitable water
            "sat",  # satellite
            "sfc",  # surface
            "sfclt",  # surface (less than)
            "top",  # nominal top of atmosphere
            "total",  # total clouds
            "ua",  # upper air
            "uanat",  # upper air native file
            "uaprs",  # upper air prs file
        ]

        allowed_lev_type = [
            "cm",  # centimeters
            "ds",  # difference
            "ft",  # feet
            "km",  # kilometers
            "m",  # meters
            "mm",  # millimeters
            "mb",  # milibars
            "sr",  # storm relative
        ]

        allowed_stat = [
            "in",  # ???
            "ens",  # ensemble
            "m",  # ???
            "maxm",  # ???
            "mn",  # minimum
            "mx",  # maximum
        ]

        # Easy check first -- it is in the allowed_levels list
        if key in allowed_levels:
            return True

        # Check for [numeric][lev_type] or [lev_type][numeric] pattern

        # Numbers come at beginning or end, only
        numeric = "".join([c for c in key if c in digits + "."]) in key

        # The level is allowed
        level_str = [c for c in key if c in ascii_letters]
        allowed = "".join(level_str) in allowed_lev_type + allowed_stat

        # Check the other direction - level string contains one of the allowed
        # types.
        if not allowed:
            for lev in allowed_lev_type + allowed_stat:
                if lev in level_str:
                    allowed = True
                    break

        return numeric and allowed

    def is_a_key(self, key):
        """Returns true if key exists as a key in the config file."""

        return self.cfg.get(key) is not None

    @staticmethod
    def is_bool(k):
        """Returns true if k is a boolean variable."""

        return isinstance(k, bool)

    def is_callable(self, funcs):
        """Returns true if func in funcs list is the name of a callable function."""

        funcs = funcs if isinstance(funcs, list) else [funcs]

        callables = []
        for func in funcs:
            callable_ = self.get_callable(func)
            callable_ = callable_ if isinstance(callable_, list) else [callable_]

            for clbl in callable_:
                if isinstance(clbl, np.ndarray) or callable(clbl):
                    callables.append(True)
                else:
                    callables.append(False)

        return all(callables)

    @staticmethod
    def is_dict(d):
        """Returns true if d is a dictionary."""

        return isinstance(d, dict)

    @staticmethod
    def is_int(i):
        """Returns true if i is an integer."""

        if isinstance(i, int):
            return True
        return i.isnumeric() and len(i.split(".")) == 1

    @staticmethod
    def is_number(i):
        """Returns true if i is a number."""

        if isinstance(i, (int, float)):
            return True
        return i.isnumeric() and len(i.split(".")) <= 2

    @staticmethod
    def is_string(s):
        """Returns true if s is a string."""

        return isinstance(s, str)

    def is_wind(self, wind):
        """Returns true if wind is a bool or is_a_level."""

        return isinstance(wind, bool) or self.is_a_level(wind)

    def check_keys(self, d, depth=0):
        """
        Helper function that recursively checks the keys in the dictionary by calling the
        function defined in allowable.
        """

        max_depth = 2

        # Only proceed if d is a dictionary
        if not isinstance(d, dict):
            return

        # Proceed only up to max depth.
        if depth >= max_depth:
            return

        level = depth + 1

        for k, v in d.items():
            # Check that the key is allowable
            assert (k in self.allowable) or self.is_a_level(k)

            # Call a checker if one exists for the key, otherwise descend into
            # next level of dict
            checker = self.allowable.get(k)
            if checker:
                if isinstance(checker, bool):
                    assert checker
                else:
                    assert checker(v)
            elif isinstance(v, dict):
                self.check_keys(v, depth=level)

    def test_keys(self):
        """Tests each of top-level variables in the config file by calling the helper function."""

        for short_name, spec in self.cfg.items():
            assert "_" not in short_name
            self.check_keys(spec)
