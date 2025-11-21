# pylint: disable=invalid-name,too-few-public-methods,too-many-locals,too-many-branches,too-many-statements

"""
Classes that load grib files.
"""

from pathlib import Path

import cfgrib
import xarray as xr


class GribFile:
    """Wrappers and helper functions for interfacing with cfgrib."""

    def __init__(self, filename: Path | str, cfgrib_config: dict):
        # pylint: disable=unused-argument

        self.filename = filename
        self.cfgrib_config = cfgrib_config
        self.contents = self._load()

    def _load(self) -> xr.Dataset:
        """
        Internal method that opens the grib file. Returns a grib message
        iterator.
        """

        return xr.open_dataset(
            self.filename,
            engine="cfgrib",
            lock=False,
            backend_kwargs=(
                {
                    "filter_by_keys": self.cfgrib_config,
                    "read_keys": ["orientationOfTheGridInDegrees"],
                }
            ),
        )


class GribFiles:
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
        self.contents = self._load()

    def _load(self, filenames: list[Path] | None = None):
        """Load the set of files into a single XArray structure."""
        filenames = self.filenames if filenames is None else filenames
        ds = xr.open_mfdataset(
            filenames,
            engine="cfgrib",
            concat_dim="time",
            combine="nested",
            backend_kwargs=(
                {
                    "filter_by_keys": self.cfgrib_config,
                    "indexpath": "",  # create a temp file here or pyfakefs to hold it in mem. check
                    # unittests in wxvx
                    "read_keys": ["orientationOfTheGridInDegrees"],
                }
            ),
        )
        return {_var_id(ds, list(ds.data_vars)[0]): ds}


class WholeGribFile:
    def __init__(
        self,
        filename: Path,
    ):
        self.filename = filename
        self.contents = self._load(filename)

    def _load(self, filename: Path):
        datasets = cfgrib.open_datasets(
            str(filename), read_keys=["orientationOfTheGridInDegrees", "parameterNumber"]
        )

        all_fields: dict = {}
        for ds in datasets:
            for var in ds.data_vars:
                # var_name = var if var != "unknown" else ds[var].attrs.get("GRIB_cfName",
                # ds[var].attrs.get("GRIB_shortName"))
                var_id = _var_id(ds, str(var))
                if all_fields.get(var_id) is None:
                    all_fields[var_id] = ds
                else:
                    msg = f"Multiple entries for {var_id} when opening {filename}"
                    raise ValueError(msg)
        return all_fields


def _var_id(ds: xr.Dataset, var: str):
    vertical_dim = ds[list(ds.data_vars)[0]].attrs.get("GRIB_typeOfLevel")
    var_name = ds[var].attrs.get("GRIB_shortName")
    step_type = ds[var].attrs.get("GRIB_stepType", "nostepType")
    return f"{var_name}_{vertical_dim}_{step_type}"
