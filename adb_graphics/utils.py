"""
A set of generic utilities available to all the adb_graphics components.
"""

import functools
import glob
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import yaml
from uwtools.api.config import YAMLConfig
from uwtools.config.support import uw_yaml_loader


def cfgrib_spec(config: dict, model: str) -> dict:
    """
    Given a cfgrib block and a model, return the appropriate sub-block, if it exists.
    """
    spec: dict = config.get(model, {})
    if spec and isinstance(spec, dict):
        return spec
    return config


def _write_zip(files_to_zip: list[Path], zipf: Path | str):
    """Write the zip file, overwriting existing files that have a newer modification timestamp."""
    print(f"Writing to zip file {zipf} for files like: {files_to_zip[0].name}")
    overwrite = {}
    with ZipFile(zipf, "a") as zf:
        arcfiles = zf.namelist()
        for file in files_to_zip:
            if file.name in arcfiles:
                arcinfo = zf.getinfo(file.name)
                arc_mod_time = datetime(*arcinfo.date_time)
                file_mod_time = datetime.fromtimestamp(file.stat().st_mtime)
                if file_mod_time > arc_mod_time:
                    overwrite[file.name] = file
            else:
                zf.write(file, arcname=Path(file).name)
        if overwrite:
            tmp_path = Path(f"{zipf}.tmp")
            with ZipFile(tmp_path, "w") as tmp:
                for item in zf.namelist():
                    if (arcfile := zf.getinfo(item).filename) not in overwrite:
                        tmp.write(arcfile, str(zf.read(item)))
                for arcname, file in overwrite.items():
                    tmp.write(file, arcname=arcname)

            tmp_path.rename(zipf)


def create_zip(files_to_zip: list[Path], zipf: Path | str):
    """Create a zip file. Use a locking mechanism -- write a lock file to disk."""

    lock_file = Path(f"{zipf}._lock")
    retry = 2
    count = 0
    while True:
        if not lock_file.exists():
            # Create the lock
            lock_file.touch()
            try:
                _write_zip(files_to_zip, zipf)
            except Exception as e:
                count += 1
                if count >= retry:
                    msg = "Error writing zip file!"
                    raise RuntimeError(msg) from e
            else:
                # Zipping was successful. Remove files that were zipped
                for file_to_zip in files_to_zip:
                    file_to_zip.unlink(missing_ok=True)
                break
            finally:
                lock_file.unlink(missing_ok=True)

        # Wait before trying to obtain the lock on the file
        time.sleep(1)


def fhr_list(args: list[int]) -> list[int]:
    """
    Given an argparse list argument, return the sequence of forecast hours to
    process.

    The length of the list will determine what forecast hours are returned:

      Length = 1:   A single fhr is to be processed
      Length = 2:   A sequence of start, stop with increment 1
      Length = 3:   A sequence of start, stop, increment
      Length > 3:   List as is

    argparse should provide a list of at least one item (nargs='+').
    """

    args = args if isinstance(args, list) else [args]
    arg_len = len(args)
    if arg_len in (2, 3):
        args[1] += 1
        return list(range(*args))

    return args


def from_datetime(date: datetime) -> str:
    """Return a string like YYYYMMDDHH given a datetime object."""
    return datetime.strftime(date, "%Y%m%d%H")


def get_func(val: str):
    """
    Gets a callable function.

    Given an input string, val, returns the corresponding callable function.
    This function is borrowed from stackoverflow.com response to "Python: YAML
    dictionary of functions: how to load without converting to strings."
    """

    module_name, fun_name = val.rsplit(".", 1)

    mod_spec = find_spec(module_name, package="adb_graphics")
    if mod_spec is None:
        mod_spec = find_spec("." + module_name, package="adb_graphics")
    if mod_spec is None:
        msg = "Could not find {module_name} in current environment."
        raise ValueError(msg)

    import_module(mod_spec.name)
    module = sys.modules[mod_spec.name]
    return getattr(module, fun_name)


def join_ranges(loader: yaml.SafeLoader, node: yaml.Node) -> Any:  # noqa: ARG001
    """
    Merge two or more different ranges into a single array for color bar clevs.

    e.g.: in default_specs.yml, clevs for visibility can be assigned as

        clevs: !join_ranges [[0, 10, 0.1], [10, 51, 1.0]]

    The join_ranges method concatenates these ranges into a single array of levels.
    This can be useful for plots where one part of the color ranges requires higher
    resolution than the rest, while keeping the colorbar from looking squished.

    Note that a "yaml.add_constructor" is required, as shown after the method.
    """

    list_ = []
    for seq_node in node.value:
        range_args = [float(scalar_node.value) for scalar_node in seq_node.value]

        list_.append(np.arange(*range_args))

    return np.concatenate(list_, axis=0)


def arange_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> Any:  # noqa: ARG001
    return np.arange(*[float(n.value) for n in node.value])


def load_yaml(config: Path | str) -> YAMLConfig:
    yaml.add_constructor("!join_ranges", join_ranges, Loader=uw_yaml_loader())
    yaml.add_constructor("!arange", arange_constructor, Loader=uw_yaml_loader())
    return YAMLConfig(config)


def load_sites(arg: str | Path) -> list[str]:
    """Return the contents of the sites file, if it exists."""
    path = Path(arg)
    with path.open() as sites_file:
        sites: list[str] = sites_file.readlines()
    return sites


def load_specs(arg: str | Path) -> YAMLConfig:
    """Check to make sure arg file exists. Return its contents."""

    spec_file = Path(arg)
    if not spec_file.exists():
        msg = f"The spec file {spec_file} does not exist."
        raise FileNotFoundError(msg)
    specs = load_yaml(spec_file)
    specs["file"] = spec_file

    return specs


def numeric_level(level: str | None = None) -> tuple[float | int | str, str]:
    """
    Split the numeric level and unit associated with the level key.

    A blank string is returned for lev_val for levels that do not contain a
    numeric, e.g., 'sfc' or 'ua'.
    """

    level = level if level is not None else ""
    if m := re.match(r"^([0-9.]+)?([a-z]+)?([0-9.]+)?$", level):
        groups = m.groups()
        units = groups[1]
        value = groups[0] or groups[2]
        for convert in (int, float):
            try:
                return convert(value), units
            except (TypeError, ValueError):  # noqa: PERF203
                pass
    return "", ""


def old_enough(age: int, file_path: Path | str):
    """
    Helper function to test the age of a file.

    Input:

      age         desired age in minutes
      file_path   full path to file to check

    Output:

      bool    whether the file is at least age minutes old
    """

    file_path = Path(file_path) if isinstance(file_path, str) else file_path

    file_time = datetime.fromtimestamp(file_path.stat().st_ctime)
    max_age = datetime.now() - timedelta(minutes=age)

    return file_time < max_age


def path_exists(path: Path | str):
    """Checks whether a file exists, and returns the path if it does."""

    ret_path = Path(path)
    if not ret_path.exists():
        msg = f"{path} does not exist!"
        raise FileNotFoundError(msg)

    return ret_path


def set_level(level: str, model: str, spec: dict):
    """
    Given the default_specs level string, extract and set a numeric level in the cfgrib block.
    """
    nlevel, _ = numeric_level(level=level)
    level_info = any(
        key
        for keys in cfgrib_spec(spec["cfgrib"], model)
        for key in ("level", "top", "bottom", "Surface")
        if key in keys
    )
    if nlevel and not level_info:
        if spec["cfgrib"].get(model) is not None:
            spec["cfgrib"][model]["level"] = nlevel
        else:
            spec["cfgrib"]["level"] = nlevel


def timer(func: Callable):
    """Decorator function that provides an elapsed time for a method."""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"{func.__name__} Elapsed time: {elapsed_time:0.4f} seconds")
        return value

    return wrapper_timer


def to_datetime(string: str):
    """Return a datetime object give a string like YYYYMMDDHH."""

    return datetime.strptime(string, "%Y%m%d%H")


def uniq_wgrib2_list(inlist: list[str]):
    """
    Given a list of wgrib2 output fields, returns a uniq list of fields for
    simplifying a grib2 dataset. Uniqueness is defined by the wgrib output from
    field 3 (colon delimted) onward, although the original full grib record must
    be included in the wgrib2 command below.
    """

    uniq_field_set = set()
    uniq_list = []
    for infield in inlist:
        infield_info = infield.split(":")
        if len(infield_info) <= 3:  # noqa: PLR2004
            continue
        infield_str = ":".join(infield_info[3:])
        if infield_str not in uniq_field_set:
            uniq_list.append(infield)
        uniq_field_set.add(infield_str)

    return uniq_list


@timer
def zip_products(fhr: int, workdir: Path, zipfiles: dict) -> None:  # pragma: no cover
    """
    Spin up a subprocess to zip all the product files into the staged zip files.

    Input:

        fhr         integer forecast hour
        workdir     path to the product files
        zipfiles    dictionary of tile keys, and zip directory values.

    Output:
        None
    """

    for tile, zipf in zipfiles.items():
        if tile == "skewt_csv":
            file_tmpl = f"*.skewt.*_f{fhr:03d}.csv"
        else:
            file_tmpl = f"*_{tile}_*{fhr:02d}.png"
        product_files = [Path(f) for f in glob.glob(str(workdir / file_tmpl))]
        if product_files:
            create_zip(product_files, zipf)
