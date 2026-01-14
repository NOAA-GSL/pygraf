"""
Classes that load grib files.
"""

import warnings
from functools import cached_property
from pathlib import Path

import cfgrib
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")


class GribFiles:  # pragma: no cover
    """
    Class for loading in a set of grib files and combining them over
    forecast hours.
    """

    def __init__(
        self,
        filenames: list[Path],
        cfgrib_config: dict,
    ):
        """
        Initialize GribFiles object.

          coord_dims  dict containing the name of the dimension to
                      concat (key), and a list of its values (value).
                        Ex: {'fhr': [2, 3, 4]}
          filenames   dict containing list of files names for the 0h and 1h
                      forecast lead times ('01fcst'), and all the free forecast
                      hours after that ('free_fcst').
          filetype    key to use for dict when setting variable_names
          model       string describing the model type
        """

        self.filenames = filenames
        self.cfgrib_config = cfgrib_config

    @cached_property
    def datasets(self):
        """Load the set of files into a single Xarray structure."""
        ds = xr.open_mfdataset(
            self.filenames,
            engine="cfgrib",
            concat_dim="time",
            combine="nested",
            compat="override",
            coords="minimal",
            backend_kwargs=(
                {
                    "filter_by_keys": self.cfgrib_config,
                    "indexpath": "",
                    "read_keys": ["orientationOfTheGridInDegrees"],
                }
            ),
        )
        return {_var_id(ds, list(ds.data_vars)[0]): ds}


class WholeGribFile:
    """
    Class for loading a whole gribfile into a dictionary for different categories of data, mostly
    separated by vertical coordinate and bucket type (avg, max, etc.).
    """

    def __init__(
        self,
        filename: Path,
    ):
        self.filename = filename

    @cached_property
    def datasets(self):
        datasets = cfgrib.open_datasets(
            str(self.filename),
            read_keys=["orientationOfTheGridInDegrees", "parameterNumber"],
            backend_kwargs=(
                {
                    "indexpath": "",
                    "read_keys": ["orientationOfTheGridInDegrees"],
                }
            ),
        )

        all_fields: dict = {}
        for ds in datasets:
            for var in ds.data_vars:
                var_id = _var_id(ds, str(var))
                if all_fields.get(var_id) is None:
                    all_fields[var_id] = ds
                else:  # pragma: no cover
                    msg = f"Multiple entries for {var_id} when opening {self.filename}"
                    raise ValueError(msg)
        return all_fields


def _var_id(ds: xr.Dataset, var: str):
    vertical_dim = ds[list(ds.data_vars)[0]].attrs.get("GRIB_typeOfLevel", "unknown")
    var_name = ds[var].attrs.get("GRIB_shortName")
    step_type = ds[var].attrs.get("GRIB_stepType", "nostepType")
    return f"{var_name}_{vertical_dim}_{step_type}"
