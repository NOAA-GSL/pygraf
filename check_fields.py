import glob
import sys

from adb_graphics import utils
from adb_graphics.datahandler import gribfile, gribdata
from create_graphics import load_images


def main(args):

    grib_file, image_file = args

    model, var_levels = load_images([image_file, "hourly"])

    ds = gribfile.WholeGribFile(grib_file).datasets
    first_item = next(iter(ds.keys()))
    fhr = int(ds[first_item].step.dt.total_seconds() // 3600)

    specs = utils.load_yaml("adb_graphics/default_specs.yml")
    specs.dereference(context={"fhr": int(fhr), "file_type": "prs"})


    for variable, levels in var_levels.items():
        for level in levels:
            spec = specs.get(variable, {}).get(level)
            if spec is None:
                print(f"No spec for {variable} at {level}")
                continue
            vspec = utils.cfgrib_spec(spec["cfgrib"], model)
            args = {
                "fhr": fhr,
                "level": level,
                "model": model,
                "short_name": variable,
                "spec": specs,
                "ds": ds,
            }
            try:
                field = gribdata.FieldData(**args).data
            except Exception as e:
                print(str(e))
                continue

if __name__ == "__main__":
    main(sys.argv[1:])



