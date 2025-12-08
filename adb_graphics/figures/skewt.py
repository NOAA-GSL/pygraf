# pylint: disable=invalid-name
"""
The module the contains the SkewTDiagram class responsible for creating a Skew-T
Log-P diagram using MetPy.
"""

from functools import cached_property
from math import atan2, degrees
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
from metpy.plots import Hodograph, SkewT
from metpy.units import units
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from uwtools.api.config import YAMLConfig
from xarray import DataArray, Dataset, where

from adb_graphics import errors
from adb_graphics.datahandler import gribdata

if TYPE_CHECKING:
    from pint import UnitRegistry


class HydroPlotSettings(TypedDict):
    color: str
    label: str
    marker: str
    scale: float
    units: str


class SkewTDiagram(gribdata.ProfileData):
    """
    The class responsible for gathering all data needed from a grib file to
    produce a Skew-T Log-P diagram.

    Input:

      ds               xarray dataset from grib file
      loc              the entire line entry of the sites file.

    Key word arguments:

      fhr              forecast hour
      max_plev         maximum pressure level to plot in mb
      model_name       model name to use for plotting

    Additional keyword arguments for the gribdata.ProfileData base class should also
    be included.
    """

    def __init__(
        self,
        fhr: int,
        ds: dict[str, Dataset],
        loc: str,
        model: str,
        spec: dict | YAMLConfig,
        max_plev: int | None = 0,
        model_name: str | None = None,
    ):
        # Initialize on the temperature field since we need to gather
        # field-specific data from this object, e.g. dates, lat, lon, etc.

        super().__init__(fhr=fhr, ds=ds, loc=loc, model=model, short_name="temp", spec=spec)

        self.max_plev = max_plev
        self.model_name = model_name or "Analysis"

    def _add_hydrometeors(self, hydro_subplot: Axes):
        mixing_ratios: dict[str, HydroPlotSettings] = {
            "clwmr": {
                "color": "blue",
                "label": "CWAT",
                "marker": "s",
                "scale": 1.0,
                "units": "g/m2",
            },
            "icmr": {
                "color": "red",
                "label": "CICE",
                "marker": "^",
                "scale": 10.0,
                "units": "g/m2",
            },
            "rwmr": {
                "color": "cyan",
                "label": "RAIN",
                "marker": "o",
                "scale": 1.0,
                "units": "g/m2",
            },
            "snmr": {
                "color": "purple",
                "label": "SNOW",
                "marker": "*",
                "scale": 1.0,
                "units": "g/m2",
            },
            "grle": {
                "color": "orange",
                "label": "GRPL",
                "marker": "D",
                "scale": 1.0,
                "units": "g/m2",
            },
        }

        pres = self.atmo_profiles["pres"]["data"]
        temp = self.atmo_profiles["temp"]["data"]
        handles = []
        gravity = 9.81  # m/s^2

        lines = [
            "Vert. Integrated Amt\n(Resolved, Total)\n"
            "(supercool layers shaded,\nwith filled markers)"
        ]
        freezing_f = 32.0

        for mixr, settings in mixing_ratios.items():
            # Get the profile values
            scale = settings.get("scale", 1.0)
            try:
                profile = self.values(name=mixr) * 1000.0 * scale
            except (errors.NoGraphicsDefinitionForVariableError, IndexError, ValueError):
                try:
                    profile = self.values(name=mixr, level="uanat") * 1000.0 * scale
                except errors.NoGraphicsDefinitionForVariableError:
                    print(f"missing {mixr} for hydrometeor plot, skipping that field.")
                    continue
            mixr_total: units = 0.0
            if profile.any():
                for n in range(len(pres)):
                    if n == 0:
                        pres_sigma = pres[0]
                    else:
                        pres_layer = 2 * (pres_sigma - pres[n])  # layer depth
                        pres_sigma = pres_sigma - pres_layer  # pressure at next sigma level
                        mixr_total = mixr_total + pres_layer / gravity * profile[n]
                mixr_total = mixr_total.to_numpy()

            # limit values to upper and lower values of plotting range
            profile = where((profile > 0.0) & (profile < 1.0e-4), 1.0e-4, profile)  # noqa: PLR2004
            profile = where((profile > 10.0), 10.0, profile)  # noqa: PLR2004

            # plot line
            profile = profile[: pres.shape[0]]
            hydro_subplot.plot(
                profile,
                pres,
                settings.get("color", ""),
                fillstyle="none",
                linewidth=0.5,
                marker=settings.get("marker"),
                markersize=6,
            )
            if mixr in ["clwmr", "rwmr"]:
                freezing_levs = profile.where(
                    (profile > 0.0) & (temp.magnitude < freezing_f), profile, 0
                ).to_numpy()
                if freezing_levs.any():
                    hydro_subplot.plot(
                        profile[temp.magnitude < freezing_f],
                        pres[temp.magnitude < freezing_f],
                        settings.get("color", ""),
                        fillstyle="full",
                        linewidth=0.5,
                        marker=settings.get("marker"),
                        markersize=6,
                    )
                    pres_levs = pres[freezing_levs > 0].magnitude
                    rect = plt.Rectangle(
                        (0, pres_levs[-1]),
                        100,
                        (pres_levs[0] - pres_levs[-1]),
                        facecolor=settings.get("color"),
                        alpha=0.1,
                    )
                    hydro_subplot.add_patch(rect)

            # compute vertically integrated amount and add legend line
            label = settings.get("label")
            line = f"{label:<7s} {mixr_total:>10.3f} {settings.get('units')}"
            if scale != 1.0:
                line = f"{label:<5s}(x{scale}) {mixr_total:.3f} {settings.get('units')}"
            lines.append(line)

            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=settings.get("color"),
                    fillstyle="none",
                    label=f"{label:<5s}(x{scale})" if scale != 1.0 else f"{label:<7s}",
                    linewidth=1.0,
                    marker=settings.get("marker"),
                    markersize=8,
                )
            )

        hydro_subplot.legend(handles=handles, loc=[0.05, 0.65])

        contents = "\n".join(lines)
        # Draw the vertically integrated amounts box
        hydro_subplot.text(
            0.02,
            0.98,
            contents,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.7),
            fontproperties=fm.FontProperties(family="monospace"),
            size=8,
            transform=hydro_subplot.transAxes,
            verticalalignment="top",
        )

    def _add_thermo_inset(self, skew: SkewT):
        # Build up the text that goes in the thermo-dyniamics box
        lines = []
        for name, items in self.thermo_variables.items():
            # Magic to get the desired number of decimals to appear.
            decimals = items.get("decimals", 0)
            data = items["data"]
            value = int(data) if decimals == 0 else data.round(decimals=decimals).to_numpy()

            # Sure would have been nice to use a variable in the f string to
            # denote the format per variable.
            line = f"{name.upper():<7s} {value!s:>6} {items['units']}"
            lines.append(line)

        contents = "\n".join(lines)

        # Draw the text box
        skew.ax.text(
            0.75,
            0.98,
            contents,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.7),
            fontproperties=fm.FontProperties(family="monospace"),
            size=8,
            transform=skew.ax.transAxes,
            verticalalignment="top",
        )

    @cached_property
    def atmo_profiles(self):
        """
        Return a dictionary of atmospheric data profiles for each variable
        needed by the skewT.

        Each of these variables must have units set appropriately for use
        with MetPy SkewT. Handle those units and conversions here since it
        differs from the requirements of other graphics units/transforms.
        """

        # We need to get pressure profile first. Entries in
        # the dict are as follows:
        #
        #   Variable short name:   consistent with default_specs.yml
        #      transform:          units string to pass to MetPy's to() function
        #      units:              the end unit of the field (after transform,
        #                          if applicable).
        atmo_vars = {
            "pres": {
                "transform": "hectoPa",
                "units": units.Pa,
            },
            "gh": {
                "units": units.gpm,
            },
            "sphum": {
                "units": units.dimensionless,
            },
            "temp": {
                "transform": "degF",
                "units": units.degK,
            },
            "u": {
                "transform": "knots",
                "units": units.meter_per_second,
            },
            "v": {
                "transform": "knots",
                "units": units.meter_per_second,
            },
        }

        for var, items in atmo_vars.items():
            # Get the profile values and attach MetPy units
            vals = self.values(name=var).to_numpy() * items["units"]

            # Apply any needed transformations
            transform = items.get("transform")
            atmo_vars[var]["data"] = vals.to(transform) if transform else vals

        return atmo_vars

    def create_diagram(self):
        """
        Calls the private methods for creating each component of the SkewT
        Diagram.
        """

        skew, hydro_subplot = self._setup_diagram()
        self._title()
        self._plot_profile(skew)
        self._plot_wind_barbs(skew)
        self._plot_labels(skew)

        self._plot_hodograph(skew)
        self._add_thermo_inset(skew)
        self._add_hydrometeors(hydro_subplot)

    def create_csv(self, csv_path: Path | str):
        """Calls the private methods for writing each of the SkewT Data."""

        self._write_profile(csv_path)

    def _plot_hodograph(self, skew: SkewT):
        # Create an array that indicates which layer (10-3, 3-1, 0-1 km) the
        # wind belongs to. The array, agl, will be set to the height
        # corresponding to the top of the layer. The resulting array will look
        # something like this:
        #
        #   agl = [1.0 1.0 1.0 3.0 3.0 3.0 10.0 10.0 10.0 10.87 ]
        #
        # Where the values above 10 km are unchanged, and there are three levels
        # in each of the 3 layers of interest.
        #
        data_copy: units = np.copy(self.atmo_profiles["gh"]["data"])
        agl = data_copy.to("km")

        # Retrieve the wind data profiles
        u_wind = self.atmo_profiles["u"]["data"]
        v_wind = self.atmo_profiles["v"]["data"]

        # Create an inset axes object that is 28% width and height of the
        # figure and put it in the upper left hand corner.
        ax = inset_axes(skew.ax, "25%", "25%", loc=2)
        h = Hodograph(ax, component_range=80.0)
        h.add_grid(increment=20, linewidth=0.5)

        intervals: UnitRegistry = np.array([0, 1, 3, 10]) * agl.units
        colors = ["xkcd:salmon", "xkcd:aquamarine", "xkcd:navy blue"]
        line_width = 1.5

        # Plot the line colored by height AGL only up to the 10km level
        lines = h.plot_colormapped(
            u_wind,
            v_wind,
            agl,
            colors=colors,
            intervals=intervals,
            linewidth=line_width,
        )

        # Local function to create a proxy line object for creating a legend on
        # a LineCollection returned from plot_colormapped. Using lines and
        # colors from outside scope.
        def make_proxy(zval: int, idx: int):
            color = colors[idx] if idx < len(colors) else lines.cmap(zval - 1)
            return Line2D([0, 1], [0, 1], color=color, linewidth=line_width)

        # Make a list of proxies
        proxies = [make_proxy(item, i) for i, item in enumerate(np.asarray(intervals.magnitude))]

        # Draw the legend
        ax.legend(
            proxies[:-1],
            ["0-1 km", "1-3 km", "3-10 km", ""],
            fontsize="small",
            loc="lower left",
        )

    @staticmethod
    def _plot_labels(skew: SkewT):
        skew.ax.set_xlabel("Temperature (F)")
        skew.ax.set_ylabel("Pressure (hPa)")

    def _write_profile(self, csv_path: str | Path):
        profiles = self.atmo_profiles  # dictionary
        pres = profiles["pres"]["data"]
        u = profiles["u"]["data"]
        v = profiles["v"]["data"]
        temp = profiles["temp"]["data"].to("degC")
        sphum = profiles["sphum"]["data"]

        dewpt = mpcalc.dewpoint_from_specific_humidity(pressure=pres, specific_humidity=sphum).to(
            "degC"
        )
        wspd = mpcalc.wind_speed(u, v)
        wdir = mpcalc.wind_direction(u, v)

        profile = pd.DataFrame(
            {
                "LEVEL": pres.magnitude,
                "TEMP": temp.magnitude,
                "DWPT": dewpt.magnitude,
                "WDIR": wdir.magnitude,
                "WSPD": wspd.magnitude,
            }
        )

        profile.to_csv(csv_path, index=False, float_format="%10.2f")

    def _plot_profile(self, skew: SkewT):
        profiles = self.atmo_profiles  # dictionary
        pres = profiles.get("pres").get("data")
        temp = profiles.get("temp").get("data")
        sphum = profiles.get("sphum").get("data")

        dewpt = mpcalc.dewpoint_from_specific_humidity(pressure=pres, specific_humidity=sphum).to(
            "degF"
        )

        # Pressure vs temperature
        skew.plot(pres, temp, "r", linewidth=1.5)

        # Pressure vs dew point temperature
        skew.plot(pres, dewpt, "blue", linewidth=1.5)

        # Compute parcel profile and plot it
        parcel_profile = mpcalc.parcel_profile(pres, temp[0], dewpt[0]).to("degC")
        skew.plot(
            pres,
            parcel_profile,
            "orange",
            linestyle="dashed",
            linewidth=1.2,
        )

    def _plot_wind_barbs(self, skew: SkewT):
        # Pressure vs wind
        skew.plot_barbs(
            self.atmo_profiles["pres"]["data"],
            self.atmo_profiles["u"]["data"],
            self.atmo_profiles["v"]["data"],
            color="blue",
            linewidth=0.2,
            y_clip_radius=0,
        )

    def _setup_diagram(self):
        # Create a new figure. The dimensions here give a good aspect ratio.
        fig = plt.figure(figsize=(12, 12))
        gs = plt.GridSpec(4, 5)

        skew = SkewT(fig, rotation=45, aspect=85, subplot=gs[:, :-1])

        # Set the range covered by the x and y axes.
        skew.ax.set_ylim(1050, self.max_plev)
        skew.ax.set_xlim(-35, 50)

        # The upper air grid is in Celcius, but we want ticks at the surface to
        # display in Fahrenheit.

        # Fahrenheit tick labels that will display
        labels_f = list(range(-20, 125, 20)) * units.degF

        # Celsius VALUES for those tick marks. These put the ticks in the right
        # spot.
        labels = labels_f.to("degC").magnitude

        # Set the MINOR tick values to the CELSIUS values.
        skew.ax.xaxis.set_minor_locator(FixedLocator(labels))

        # Set the MINOR tick labels to the FAHRENHEIT values.
        skew.ax.set_xticklabels(labels_f.magnitude, minor=True)
        skew.ax.tick_params(which="minor", length=8)

        # Turn off the MAJOR (celsius) tick marks, label the grid lines inside
        # the axes.
        skew.ax.tick_params(
            axis="x",
            labelbottom=True,
            labelcolor="gray",
            labelright=True,
            labelrotation=45,
            labeltop=True,
            length=0,
            pad=-25,
            which="major",
        )

        # Add the relevant special lines with their labels
        dry_adiabats = np.arange(-40, 210, 10) * units.degC
        skew.plot_dry_adiabats(
            dry_adiabats,
            colors="tan",
            linestyles="solid",
            linewidth=0.7,
        )
        label_lines(
            ax=skew.ax,
            lines=skew.dry_adiabats,
            labels=dry_adiabats.magnitude,
            end="top",
            offset=1,
        )

        moist_adiabats = np.arange(8, 36, 4) * units.degC
        moist_pr = np.arange(1001, 220, -10) * units.hPa
        skew.plot_moist_adiabats(
            moist_adiabats,
            moist_pr,
            colors="green",
            linestyles="solid",
            linewidth=0.7,
        )
        label_lines(
            ax=skew.ax,
            lines=skew.moist_adiabats,
            labels=moist_adiabats.magnitude,
            end="top",
        )

        mixing_lines = np.array([1, 2, 3, 5, 8, 12, 16, 20]).reshape(-1, 1) / 1000
        mix_pr = np.arange(1001, 400, -50) * units.hPa
        skew.plot_mixing_lines(
            mixing_ratio=mixing_lines,
            pressure=mix_pr,
            colors="green",
            linestyles=(0, (5, 10)),
            linewidth=0.7,
        )
        label_lines(
            ax=skew.ax,
            lines=skew.mixing_lines,
            labels=mixing_lines * 1000,
        )

        hydro_subplot = fig.add_subplot(gs[:, -1], sharey=skew.ax)
        hydro_subplot.set_xlim(0.0001, 10.0)
        hydro_subplot.set_xscale("log")
        hydro_subplot.yaxis.tick_right()
        hydro_subplot.set_aspect(23)  # completely arbitrary

        plt.grid(which="major", axis="both")
        plt.xlabel("hydrometeors")
        plt.ylabel("")

        return skew, hydro_subplot

    @cached_property
    def thermo_variables(self):
        """
        Return a dictionary of thermodynamic variables needed for the skewT.
        Ensure it's ordered because we want to print these values in this order on the SkewT
        diagram.  The return dictionary contains a 'data' entry for each variable that includes the
        value of the metric.

        Variables' transforms and units are handled by default specs in much the same way as in
        FieldData class since these are not used by MetPy explictly.
        """

        # We want the thermodynamic variables printed in the same order every time in the resulting
        # SkewT inset. The fields include:
        #
        #    Variable short name:     can be consistent with default_specs.yml.
        #                             If not, must provide level and variable
        #                             entries
        #       level:                (optional) level to choose in
        #                             default_specs.yml. Default is 'ua'
        #       variable:             (optional) top-level variable to choose
        #                             from default_specs.yml.
        #       decimals:             (optional) number of decimal places to
        #                             include when formatting output. Defaults
        #                             to 0 (integer).
        thermo: dict = {
            "cape": {  # Convective available potential energy
                "level": "sfc",
            },
            "cin": {  # Convective inhibition
                "level": "sfc",
            },
            "mucape": {  # Most Unstable CAPE
                "level": "mu",
                "variable": "cape",
            },
            "mucin": {  # CIN from MUCAPE level
                "level": "mu",
                "variable": "cin",
            },
            "li": {  # Lifted Index
                "decimals": 1,
                "level": "sfc",
            },
            "bli": {  # Best Lifted Index
                "decimals": 1,
                "level": "best",
                "variable": "li",
            },
            "lcl": {  # Lifted Condensation Level
            },
            "lpl": {  # Lifted Parcel Level
            },
            "srh03": {  # 0-3 km Storm relative helicity
                "level": "sr03",
                "variable": "hlcy",
            },
            "srh01": {  # 0-1 km Storm relative helicity
                "level": "sr01",
                "variable": "hlcy",
            },
            "shr06": {  # 0-6 km Shear
                "level": "06km",
                "variable": "shear",
            },
            "shr01": {  # 0-1 km Shear
                "level": "01km",
                "variable": "shear",
            },
            "cell": {  # Cell motion
            },
            "pwtr": {  # Precipitable water
                "decimals": 1,
                "level": "sfc",
            },
        }

        for var, items in thermo.items():
            varname = items.get("variable", var)
            lev = items.get("level", "ua")
            spec = self.spec.get(varname, {}).get(lev)

            if not spec:
                raise errors.NoGraphicsDefinitionForVariableError(varname, lev)

            try:
                vals = self.values(level=lev, name=varname)
                transforms = spec.get("transform")
                if transforms:
                    vals = self.get_transform(transforms, vals)

            except errors.GribReadError:
                vals = DataArray([])
            thermo[var]["data"] = vals
            thermo[var]["units"] = spec.get("unit")

        return thermo

    def _title(self):
        """Creates standard annotation for a skew-T."""

        atime = self.date_to_str(self.anl_dt)
        vtime = self.date_to_str(self.valid_dt)

        # Top Left
        plt.title(
            f"{self.model_name}: {atime}\nFcst Hr: {self.fhr}",
            fontsize=16,
            loc="left",
            x=-4.8,
            y=1.03,
        )

        # Top Right
        plt.title(
            f"Valid: {vtime}",
            fontsize=16,
            loc="right",
            x=-0.20,
            y=1.03,
        )

        # Center
        site = f"{self.site_code} {self.site_num} {self.site_name}"
        site_loc = f"{self.site_lat},  {self.site_lon}"
        site_title = f"{site} at nearest grid pt over land {site_loc}"
        plt.title(
            site_title,
            fontsize=12,
            loc="center",
            x=-2.5,
            y=1.0,
        )


def label_line(ax: Axes, label: str, segment: np.ndarray, **kwargs):
    """
    Label a single line with line2D label data.

    Input:

      ax        the SkewT object axis
      label     label to be used for the current line
      segment   a list (array) of values for the current line

    Key Word Arguments

      align     optional bool to enable the rotation of the label to line angle
      end       the end of the line at which to put the label. 'bottom' or 'top'
      offset    index to use for the "end" of the array

      Any kwargs accepted by matplotlib's text box.
    """

    # Strip non-text-box key word arguments and set default if they don't exist
    align = kwargs.pop("align", True)
    end = kwargs.pop("end", "bottom")
    offset = kwargs.pop("offset", 0)

    # Label location
    if end == "bottom":
        x, y = segment[0 + offset, :]
        ip = 1 + offset
    elif end == "top":
        x, y = segment[-1 - offset, :]
        ip = -1 - offset

    if align:
        # Compute the slope
        dx = segment[ip, 0] - segment[ip - 1, 0]
        dy = segment[ip, 1] - segment[ip - 1, 1]
        ang = degrees(atan2(dy, dx))

        # Transform to screen co-ordinates
        pt = np.array([x, y]).reshape((1, 2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)), pt)[0]

        if end == "top":
            trans_angle -= 180

    else:
        trans_angle = 0

    # Set a bunch of keyword arguments
    if ("horizontalalignment" not in kwargs) and ("ha" not in kwargs):
        kwargs["ha"] = "center"

    if ("verticalalignment" not in kwargs) and ("va" not in kwargs):
        kwargs["va"] = "center"

    if "backgroundcolor" not in kwargs:
        kwargs["backgroundcolor"] = ax.get_facecolor()

    if "clip_on" not in kwargs:
        kwargs["clip_on"] = True

    if "fontsize" not in kwargs:
        kwargs["fontsize"] = "larger"

    if "fontweight" not in kwargs:
        kwargs["fontweight"] = "bold"

    # Larger value (e.g., 2.0) to move box in front of other diagram elements
    if "zorder" not in kwargs:
        kwargs["zorder"] = 1.50

    # Place the text box label on the line.
    ax.text(x, y, label, rotation=trans_angle, **kwargs)


def label_lines(ax: Axes, lines: Any, labels: np.ndarray, offset: float = 0, **kwargs):
    """
    Plots labels on a set of lines from SkewT.

    Input:

      ax      the SkewT object axis
      lines   the SkewT object special lines
      labels  list of labels to be used
      offset  index to use for the "end" of the array

    Key Word Arguments

      color   line color

      Along with any other kwargs accepted by matplotlib's text box.
    """

    if "color" not in kwargs:
        kwargs["color"] = lines.get_color()[0]

    for i, line in enumerate(lines.get_segments()):
        assert not labels[i].ndim > 1
        label = int(labels[i])
        label_line(ax, str(label), line, align=True, offset=offset, **kwargs)
