"""
This module is where pieces of the figures are put together. Data is
combined with maps and skewts to provide the final product.
"""

import gc
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import axes
from xarray import Dataset

from adb_graphics.datahandler import gribfile
from adb_graphics.figures import skewt
from adb_graphics.figures.maps import DataMap, DiffMap, Map, MapFields, MultiPanelDataMap

AIRPORTS = Path(__file__).resolve().parent.parent / "static" / "Airports_locs.txt"


def add_obs_panel(
    ax: axes.Axes,
    model_name: str,
    dataset: dict[str, Dataset],
    proj_info: dict,
    spec: dict,
    short_name: str,
    tile: str,
):
    """
    Plot observation data provided by the dataset
    path using the assigned projection.
    """

    ax.axis("on")
    map_fields = MapFields(
        fhr=0,
        fields_spec=spec,
        ds=dataset,
        level="obs",
        model="obs",
        name=short_name,
        map_type="maps",
    )
    m = Map(
        airport_fn=AIRPORTS,
        ax=ax,
        grid_info=proj_info,
        model="obs",
        tile=tile,
    )
    dm = MultiPanelDataMap(
        map_fields=map_fields,
        map_=m,
        member="obs",
        model_name=model_name,
    )

    # Draw the map
    return dm.draw()


def parallel_maps(  # noqa: PLR0915, PLR0912
    cla: Namespace,
    fhr: int,
    dataset: dict[str, Dataset],
    level: str,
    variable: str,
    workdir: Path,
    tile: str = "full",
    ds2: Path | None = None,
):
    """
    Function that creates plan-view maps, either a single panel, or
    multipanel for a forecast ensemble. Can be used in parallel.

    Input:

      fhr        forecast hour
      dataset    loaded data
      level      the vertical level of the variable to be plotted
                 corresponding to a key in the specs file
      variable   the name of the variable section in the specs file
      workdir    output directory

    Optional:
      tile       the label of the tile being plotted
      ds2        second dataset
    """

    fig, axes = set_figure(cla.images[0], cla.graphic_type, tile)
    spec = cla.specs[variable][level]
    # set last_panel to send into DataMap for colorbar control
    last_panel = False

    # Declare the type of object depending on graphic type
    map_classes = {
        "enspanel": MultiPanelDataMap,
        "diff": DiffMap,
    }
    map_class = map_classes.get(cla.graphic_type, DataMap)

    top_left = 0
    center_left = 4
    lower_left = 8
    for index, current_ax in enumerate(axes):
        if current_ax is axes[-1]:
            last_panel = True
        mem = None
        if cla.graphic_type == "enspanel":
            # Don't put data in the top left or bottom left panels.
            if index in (top_left, lower_left):
                current_ax.axis("off")

            # Shenanigans to match ensemble member to panel index. Here's where the ensemble members
            # should go:
            # ----------------
            # |   | 1 | 2 | 3 |
            # ----------------
            # | 0 | 4 | 5 | 6 |
            # ----------------
            # | o | 7 | 8 | 9 |
            # ----------------
            match index:
                case x if x in (top_left, center_left, lower_left):
                    mem = 0
                case x if x > lower_left:
                    mem = index - 2
                case x if x > center_left:
                    mem = index - 1
                case x if x < center_left:
                    mem = index

        # Create an object that holds all the fields for this map
        map_fields = MapFields(
            ds=dataset,
            ds2=ds2,
            fhr=fhr,
            fields_spec=cla.specs,
            level=level,
            name=variable,
            map_type=cla.graphic_type,
            model=cla.images[0],
            tile=tile,
        )
        # Generate a map object
        m = Map(
            airport_fn=AIRPORTS,
            ax=current_ax,
            grid_info=map_fields.shaded.grid_info(),
            model=cla.images[0],
            plot_airports=spec.get("plot_airports", True),
            tile=tile,
        )

        # Send all objects (map_field, contours, hatches) to a DataMap object
        dm = map_class(
            map_fields=map_fields,
            map_=m,
            member=mem,
            model_name=cla.model_name,
            last_panel=last_panel,
        )

        # Draw the map
        if cla.graphic_type == "enspanel":
            if index == 0:
                dm.title()
                dm.add_logo(current_ax)
            elif index == lower_left:
                if spec.get("include_obs", False) and cla.obs_file_path:
                    # Add observation panel to lower left. Currently only
                    # supported for composite reflectivity.
                    obs_ds = gribfile.WholeGribFile(cla.obs_file_path).contents
                    add_obs_panel(
                        ax=axes[8],
                        model_name=cla.model_name,
                        dataset=obs_ds,
                        proj_info=map_fields.shaded.grid_info(),
                        short_name=variable,
                        spec=cla.specs,
                        tile=tile,
                    )
            else:
                dm.draw(show=True)
        else:
            dm.draw(show=True)

    # Build the output path
    png_file = f"{variable}_{tile}_{level}_f{fhr:03d}.png"
    png_file = png_file.replace("__", "_")
    png_path = workdir / png_file

    print("*" * 120)
    print(f"Creating image file: {png_path}")
    print("*" * 120)

    # Save the png file to disk
    plt.savefig(
        png_path,
        bbox_inches="tight",
        dpi=cla.img_res,
        format="png",
        orientation="landscape",
        pil_kwargs={"optimize": True},
    )

    fig.clear()
    # Clear the current axes.
    plt.cla()
    # Clear the current figure.
    plt.clf()
    # Closes all the figure windows.
    plt.close("all")
    gc.collect()


def parallel_skewt(cla: Namespace, fhr: int, dataset: dict[str, Dataset], site: str, workdir: Path):
    """
    Function that creates a single SkewT plot.

    Can be used in parallel.
    Input:

      cla        command line arguments Namespace object
      ds         the XArray dataset
      fhr        the forecast hour integer
      site       the string representation of the site from the sites file
      workdir    output directory
    """
    # deduce model name
    possible_namers = (cla.model_name, cla.data_root, cla.file_tmpl)
    model = ""
    for name in ("global", "hrrr", "rrfs"):
        if any(
            name in namer.lower() if isinstance(namer, str) else name in str(namer[0]).lower()
            for namer in possible_namers
        ):
            model = name
            break
    skew = skewt.SkewTDiagram(
        fhr=fhr,
        ds=dataset,
        loc=site,
        model=model,
        spec=cla.specs,
        max_plev=cla.max_plev,
        model_name=cla.model_name,
    )
    skew.create_diagram()
    outfile = f"{skew.site_code}_{skew.site_num}_skewt_f{fhr:03d}.png"
    png_path = workdir / outfile
    print("*" * 80)
    print(f"Creating image file: {png_path}")
    print("*" * 80)

    plt.savefig(
        png_path,
        bbox_inches="tight",
        dpi=cla.img_res,
        format="png",
        orientation="landscape",
    )

    start_time = cla.start_time.strftime("%Y%m%d%H")
    csvfile = f"{skew.site_code}.{skew.site_num}.skewt.{start_time}_f{fhr:03d}.csv"
    csv_path = workdir / csvfile
    print("*" * 80)
    print(f"Creating csv file: {csv_path}")
    print("*" * 80)
    skew.create_csv(csv_path)

    plt.close()


def set_figure(model_name: str, graphic_type: str, tile: str):
    """
    Create the figure and subplots appropriate for the model and
    graphics type. Return the figure handle and list of axes.
    """

    inches = 12.2 if model_name == "HRRR-HI" else 10

    # Settings for a default single map
    x_aspect = 1.0
    y_aspect = 1.0
    nrows = 1
    ncols = 1

    if graphic_type == "enspanel":
        nrows = 3
        ncols = 4
        inches = 20
        # Most rough-square subdomains can use the 0.8 y_aspect
        y_aspect = 0.8
        x_aspect = 1
        if tile in ["full", "NW"]:
            # Horizontal rectangle subdomains, and CONUS need more
            # squashed horizontal rectangles
            y_aspect = 0.5
        if tile in ["SE"]:
            # Vertical rectangle subdomains can use a bit more height
            # than the others
            y_aspect = 0.95

    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(x_aspect * inches, y_aspect * inches),
        sharex=True,
        sharey=True,
    )
    # Flatten the 2D array and number panel axes from top left to bottom right
    # sequentially
    ax = ax.flatten() if isinstance(ax, np.ndarray) else [ax]

    return fig, ax
