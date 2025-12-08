"""
Classes that handle the specifics of grib files from UPP.
"""

import abc
from copy import deepcopy
from datetime import datetime, timedelta
from functools import cached_property

import numpy as np
from matplotlib.pyplot import get_cmap
from pandas import to_datetime
from uwtools.api.config import YAMLConfig
from xarray import DataArray, Dataset, ufuncs, where

from adb_graphics import conversions, errors, specs, utils


class UPPData(specs.VarSpec):
    """
    Class provides interface for accessing field  data from UPP in
    Grib2 format.

    Input:
        ds:          xarray dataset from grib file
        model:       name of the model from the image list
        short_name:  name of variable corresponding to entry in specs configuration
        spec:        full specs dictionary
    """

    def __init__(
        self,
        fhr: int,
        ds: dict[str, Dataset],
        model: str,
        short_name: str,
        spec: dict | YAMLConfig,
        level: str | None = None,
    ):
        self.model = model
        self.spec = spec
        self.short_name = short_name
        self.level = level or "ua"

        self.fhr = fhr
        cf = deepcopy(self.vspec)
        utils.set_level(level=str(level), model=self.model, spec=cf)
        cf = utils.cfgrib_spec(cf["cfgrib"], self.model)
        key = "typeOfLevel"
        try:
            self.vertical_coord = cf[key]
        except KeyError:  # pragma: no cover
            msg = f"{key} is not a key for {short_name} at {level}. cf: {cf}"
            raise KeyError(msg) from None
        self.ds = ds

    @property
    def anl_dt(self) -> datetime:
        """
        Returns the initial time of the GRIB file as a datetime object from
        the GRIB file.
        """
        ret: datetime = to_datetime(self.field.time.values)
        return ret

    @property
    def clevs(self) -> np.ndarray:
        """
        Uses the information contained in the yaml config file to determine
        the set of levels to be contoured. Returns the list of levels.

        The yaml file "clevs" key may contain a list or a range.
        """

        return np.asarray(self.vspec.get("clevs", []))

    @staticmethod
    def date_to_str(date: datetime) -> str:
        """
        Returns a formatted string (for graphic title) from a datetime object.
        """

        return date.strftime("%Y%m%d %H UTC")

    @cached_property
    def field(self) -> DataArray:
        """
        Get the first DataArray out of the Dataset.
        """
        return self._get_field(self.vspec["cfgrib"].get(self.model, self.vspec["cfgrib"]))

    def _get_data_levels(self, vertical_coord: str):
        """
        Values of the vertical dimension.

        Arg:
          vertical_coord   the name of the vertical dimension
        """
        dim = [str(coord) for coord in self.field.coords if vertical_coord in str(coord)][0]
        return self.field.coords[dim].to_numpy()

    def _get_field(self, cfgribspec: dict) -> DataArray:
        """
        Given a cfgrib block, return the DataArray.

        Arg:
          cfgribspec the specifications dictionary to use for the variable in
                     question
        """

        def _find_var():
            if ds.get(short_name) is not None:
                return short_name

            for var in ds:
                if ds[var].attrs["GRIB_shortName"] == short_name:
                    return var
            return None  # pragma: no cover

        short_name = cfgribspec.get("shortName", "unknown")
        vertical_coord = cfgribspec["typeOfLevel"]
        step_type = cfgribspec.get("stepType", "instant")
        var_id = f"{short_name}_{vertical_coord}_{step_type}"
        ds: Dataset | dict = self.ds.get(var_id, {})
        if ds == {}:
            msg = f"{var_id} is not a valid key for the dataset"
            raise ValueError(msg)
        var = _find_var()
        if var is not None:
            field = ds[var]
            top = cfgribspec.get("topLevel", cfgribspec.get("scaledValueOfFirstFixedSurface"))
            bottom = cfgribspec.get(
                "bottomLevel", cfgribspec.get("scaledValueOfSecondFixedSurface")
            )
            layered = top is not None or bottom is not None
            level = top if top in field.coords[vertical_coord] else bottom
            if level is None:
                level = cfgribspec.get("level", utils.numeric_level(self.level)[0])
            level = None if level == "" else level
            leveled = level is not None and vertical_coord != "hybrid"
            if len(field.coords[vertical_coord].shape) > 0 and (layered or leveled):
                if vertical_coord == "depthBelowLandLayer" and level:
                    level = level / 100.0  # pragma: no cover
                field = field.sel(**{vertical_coord: level})
            return DataArray(field)
        msg = f"Variable {short_name} not found in dataset."  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover

    def get_transform(self, transforms: dict | list | str, val: DataArray) -> DataArray:
        """
        Applies a set of one or more transforms to an np.array of
        data values.

        Input:

          transforms:    the transform section of a variable spec
          val:           a value, list, or array of values to be
                         transformed

        Return:
          val:           updated values after transforms have been
                         applied

        """

        transform_kwargs: dict = {}
        if isinstance(transforms, dict):
            transform_list = transforms.get("funcs")
            if not isinstance(transform_list, list):
                transform_list = [transform_list]
            transform_kwargs = transforms.get("kwargs", {})
        elif isinstance(transforms, str):
            transform_list = [transforms]
        else:
            transform_list = transforms

        for transform in transform_list:
            if len(transform.split(".")) == 1:
                val = self.__getattribute__(transform)(val, **transform_kwargs)
            else:
                val = utils.get_func(transform)(val, **transform_kwargs)
        return val

    def get_xypoint(self, site_lat: float, site_lon: float) -> tuple:
        """
        Return the X, Y grid point corresponding to the site location. No
        interpolation is used.
        """

        lats, lons = self.latlons()
        adjust = 360 if np.any(lons < 0) else 0
        lons = lons + adjust

        msg = f"site location is outside your domain! {site_lat} {site_lon}"
        if not lats.min() < site_lat < lats.max() or not lons.min() < site_lon < lons.max():
            print(msg)
            return (-1, -1)

        # Numpy magic to grab the X, Y grid point nearest the profile site
        x, y = np.unravel_index(
            (np.abs(lats - site_lat) + np.abs(lons - site_lon)).argmin(), lats.shape
        )

        return (x, y)

    def latlons(self) -> list[np.ndarray]:
        """Returns the set of latitudes and longitudes."""

        coords = sorted(
            [
                str(c)
                for c in list(self.field.coords)
                if any(ele in str(c) for ele in ["lat", "lon"])
            ]
        )
        lat = self.field.coords[coords[0]].to_numpy()
        if len(lat.shape) == 1 and lat[-1] < lat[0]:
            lat = lat[::-1]
        lon = self.field.coords[coords[-1]].to_numpy()
        return [lat, lon]

    @staticmethod
    def opposite(values: DataArray, **_kwargs) -> DataArray:
        """Returns the opposite of input values."""

        return -values

    @property
    def valid_dt(self) -> datetime:
        """
        Returns a datetime object corresponding to the forecast hour's valid
        time as set in the Grib file.
        """

        fh = timedelta(hours=int(self.fhr))
        return self.anl_dt + fh

    @abc.abstractmethod
    def values(
        self, level: str | None = None, name: str | None = None, do_transform: bool = True
    ) -> DataArray:
        """Returns the values of a given variable."""

    def vector_magnitude(
        self,
        field1: DataArray,
        field2_id: str,
        **_kwargs,
    ):
        """
        Returns the vector magnitude of two component vector fields.

        The second field can be specified by either a dict of cfkeys or a default_specs identifier
        in the form <name>_<level>.

        """
        var, lev = field2_id.split("_") if "_" in field2_id else (field2_id, self.level)
        field2 = self.values(level=lev, name=var, do_transform=False)
        mag = conversions.magnitude(field1, field2)
        field1.close()
        field2.close()

        return mag

    @property
    def vspec(self):
        """Return the graphics specification for a given level."""

        vspec = self.spec.get(self.short_name, {}).get(self.level)
        if not vspec:
            raise errors.NoGraphicsDefinitionForVariableError(self.short_name, self.level)
        return vspec


class FieldData(UPPData):
    """
    Class provides interface for accessing field (2D plan view) data from UPP in
    Grib2 format.

    Input:
        ds:          xarray dataset from grib file
        level:       level corresponding to entry in specs configuration
        name:        name of variable corresponding to entry in specs configuration

    Keyword Arguments:
        config:      path to a user-specified configuration file
        member:      integer describing the ensemble member number to
                     grab data for

    """

    def __init__(
        self,
        fhr: int,
        ds: dict[str, Dataset],
        level: str,
        model: str,
        short_name: str,
        spec: dict | YAMLConfig,
        member: str | None = None,
        contour_kwargs: dict | None = None,
    ):
        super().__init__(
            fhr=fhr,
            ds=ds,
            level=level,
            model=model,
            short_name=short_name,
            spec=spec,
        )
        self.level = level
        self.contour_kwargs = {} if contour_kwargs is None else contour_kwargs
        self.mem = member

    def aviation_flight_rules(self, values: DataArray, **_kwargs):
        """
        Generates a field of Aviation Flight Rules from Ceil and Vis.
        """

        ceil = values
        vis = self.values(name="vis", level="sfc")

        flru = where((ceil > 1.0) & (ceil < 3.0), 1.01, 0.0)
        flru = where((vis > 3.0) & (vis < 5.0), 1.01, flru)
        flru = where((ceil > 0.5) & (ceil < 1.0), 2.01, flru)
        flru = where((vis > 1.0) & (vis < 3.0), 2.01, flru)
        flru = where((ceil > 0.0) & (ceil < 0.5), 3.01, flru)
        flru = where((vis < 1.0), 3.01, flru)

        vis.close()

        return DataArray(flru)

    @property
    def cmap(self):
        """
        The LinearSegmentedColormap specified by the config key 'cmap'.
        """

        return get_cmap(self.vspec["cmap"])

    @property
    def colors(self) -> np.ndarray:
        """
        Returns an array of colors, specified by the config key "colors".
        """

        color_spec = self.vspec.get("colors", "")
        if not color_spec:
            msg = f"No colors definition found for {self.short_name} at {self.level}"
            raise errors.NoGraphicsDefinitionForVariableError(msg)
        try:
            ret = self.__getattribute__(color_spec)
        except AttributeError as e:
            msg = f"There is no color definition named {color_spec}"
            raise AttributeError(msg) from e
        if callable(ret):
            return np.asarray(ret())  # pragma: no cover
        return np.asarray(ret)

    @property
    def corners(self) -> list:
        """

        Returns lat and lon of lower left (ll) and upper right (ur) corners.

        Order:
               ll_lat, ur_lat, ll_lon, ur_lon
        """

        lat, lon = self.latlons()
        if len(lat.shape) == 2:
            return [
                np.round(x, decimals=6) for x in [lat[0, 0], lat[-1, -1], lon[0, 0], lon[-1, -1]]
            ]
        return [np.round(x, decimals=6) for x in [lat[0], lat[-1], lon[0], lon[-1]]]

    @property
    def data(self) -> DataArray:
        """
        Sets the data property on the object for use when we need to update
        the values associated with a given object -- helpful for differences.
        """
        if not hasattr(self, "_data"):
            self._data = self.values()
        return self._data

    @data.setter
    def data(self, value: DataArray):
        self._data = value

    def field_column_max(self, values: DataArray, **_kwargs):
        """Returns the column max of the values."""

        return values.max(dim=self.vertical_coord)

    def field_diff(self, values: DataArray, variable2: str, level2: str, **kwargs):
        """Subtracts the values from variable2 from self.field."""

        value2 = self.values(
            name=variable2, level=level2, do_transform=kwargs.get("do_transform", True)
        )
        diff = values - value2
        value2.close()

        return diff

    def field_mean(
        self,
        values: DataArray,
        levels: list,
        **kwargs,
    ):
        """Returns the mean of the values over the vertical dimension."""

        levels = kwargs["global_levels"] if "global" in self.model else levels
        levs = [int(x[:-2]) for x in levels]
        return values.sel(isobaricInhPa=levs).mean("isobaricInhPa")

    def field_sum(self, values: DataArray, variable2: str, level2: str, **kwargs):
        """Return the sum of the values."""

        value2 = self.values(
            name=variable2, level=level2, do_transform=kwargs.get("do_transform", True)
        )
        sum2 = values + value2
        value2.close()

        return sum2

    def fire_weather_index(self, values: DataArray, **_kwargs):
        """
        Generates a field of Fire Weather Index.

        This method uses wrfprs data to find regions where
        weather conditions are most likely to lead to wildfires.

        """

        # Gather fields from the input
        veg = values

        temp = self.values(level="2m", name="temp", do_transform=False)
        dewpt = self.values(level="2m", name="dewp", do_transform=False)
        weasd = self.values(level="sfc", name="weasd", do_transform=False)
        gust = self.values(level="10m", name="gust", do_transform=False)
        soilm = self.values(level="sfc", name="soilm", do_transform=False)

        # A few derived fields
        dewpt_depression = temp - dewpt
        dewpt_depression = where(dewpt_depression < 0, 0, dewpt_depression)
        dewpt_depression = ufuncs.maximum(15.0, dewpt_depression)
        gust_max = np.maximum(3.0, gust)

        snowc = (25.0 - weasd) / 25.0
        snowc = where(snowc > 0.0, snowc, 0.0)

        mois = 0.01 * (100.0 - soilm)

        # Set urban (13), snow/ice (15), barren (16), and water (17) to 0.
        for vegtype in [13, 15, 16, 17]:
            veg = where(veg == vegtype, 0, veg)

        # Set all others vegetation types to 1
        veg = where(veg > 0, 1, veg)

        fwi = veg * (2.37 * (gust_max**1.11) * (dewpt_depression**0.92) * (mois**6.95) * snowc)

        fwi = fwi / 10.0

        temp.close()
        dewpt.close()
        weasd.close()
        gust.close()
        soilm.close()

        return fwi

    def grid_info(self) -> dict:
        """Returns a dict that includes the grid info for the full grid."""

        # Keys are grib names, values are Basemap argument names
        keys_to_basemap = dict(
            CenterLon="lon_0",
            CenterLat="lat_0",
            GRIB_Latin2InDegrees="lat_1",
            GRIB_Latin1InDegrees="lat_2",
            GRIB_LoVInDegrees="lon_0",
            GRIB_orientationOfTheGridInDegrees="lon_0",
            Latin2="lat_1",
            Latin1="lat_2",
            Lov="lon_0",
            La1="lat_0",
            La2="lat_2",
            Lo1="lon_1",
            Lo2="lon_2",
        )

        grid_info: dict[str, str | float | int | list] = {}
        var_info = self.field
        grid_def = var_info.attrs["GRIB_gridDefinitionDescription"].lower()
        match grid_def:  # pragma: no cover
            case x if "lambert" in x:
                attrs = [
                    "GRIB_Latin1InDegrees",
                    "GRIB_Latin2InDegrees",
                    "GRIB_LoVInDegrees",
                ]
                grid_info["projection"] = "lcc"
                grid_info["lat_0"] = 39.0
            case x if "polar stereographic" in x:
                attrs = ["GRIB_orientationOfTheGridInDegrees"]
                grid_info["projection"] = "stere"
                grid_info["lat_0"] = 90
            case "rotated latitude/longitude":  # RRFS NA
                attrs = []
                grid_info["projection"] = "rotpole"
                lon_0: float = var_info.attrs["GRIB_longitudeOfSouthernPoleInDegrees"]
                grid_info["lon_0"] = lon_0 - 360
                center_lat: float = var_info.attrs["GRIB_latitudeOfSouthernPoleInDegrees"]
                grid_info["o_lat_p"] = -center_lat if center_lat < 0 else 90 - center_lat
                grid_info["o_lon_p"] = 180

            case x if "equidistant cylindrical" in x:  # GFS
                attrs = []
                grid_info["projection"] = "cyl"
            case _:
                msg = f"Can't define grid for {grid_def}"
                raise ValueError(msg)
        if self.model != "hrrrhi":
            if not grid_info.get("corners"):
                grid_info["corners"] = self.corners

            for attr in attrs:
                bm_arg = keys_to_basemap[attr]
                val = var_info.attrs[attr]
                val = val[0] if isinstance(val, np.ndarray) else val
                grid_info[bm_arg] = val
                del val

        else:
            grid_info["lat_0"] = 20.44
            grid_info["lon_0"] = 202.54
            grid_info["width"] = 2000000
            grid_info["height"] = 2000000

        return grid_info

    @staticmethod
    def icing_adjust_trace(values: DataArray, **_kwargs):
        """Changes the value of ICSEV trace from 4.0 to 0.5, to maintain ascending order."""

        return where(values == 4.0, 0.5, values)

    @staticmethod
    def run_max(values: DataArray, **_kwargs):
        """Finds the max hourly value over all the forecast lead times available."""

        return values.max(dim="time")  # pragma: no cover

    @staticmethod
    def run_min(values: DataArray, **_kwargs):
        """Finds the min hourly value over all the forecast lead times available."""

        return values.min(dim="time")  # pragma: no cover

    @staticmethod
    def run_total(values: DataArray, **_kwargs):
        """Sums over all the forecast lead times available."""

        return values.sum(dim="time")  # pragma: no cover

    def supercooled_liquid_water(self, **_kwargs):
        """
        Generates a field of Supercooled Liquid Water.

        This method uses wrfnat data to find regions where
        cloud and rain moisture are in below-freezing temps.

        Because pressures represent mid-layer values, the calculation
        works from the surface and (1) computes the depth of a pressure layer,
        and (2) computes supercooled liquid water for the layer and sums the
        columns, and (3) uses the layer depth to find the pressure at the
        next sigma level.

        The process is iterative to the topof the atmosphere.
        """
        pres_sfc = self.values(name="pres", level="sfc") * 100.0  # convert back to Pa
        pres_nat_lev = self.values(name="pres", level="ua")
        temp = self.values(name="temp", level="ua")
        cloud_mixing_ratio = self.values(name="clwmr", level="ua")
        rain_mixing_ratio = self.values(name="rwmr", level="ua")

        gravity = 9.81
        slw = pres_sfc * 0.0  # start with array of zero values

        nlevs = np.shape(pres_nat_lev)[0]  # determine number of vertical levels
        for n in range(nlevs):
            if n == 0:
                pres_layer = 2 * (pres_sfc[:, :] - pres_nat_lev[n, :, :])  # layer depth
                pres_sigma = pres_sfc - pres_layer  # pressure at next sigma level
            else:
                pres_layer = 2 * (pres_sigma[:, :] - pres_nat_lev[n, :, :])  # layer depth
                pres_sigma = pres_sigma - pres_layer  # pressure at next sigma level
            # compute supercooled water in layer and add to previous values
            supercool_locs = where(
                (temp[n, ::] < 0.0),
                cloud_mixing_ratio[n, ::] + rain_mixing_ratio[n, ::],
                0.0,
            )
            slw = slw + pres_layer / gravity * supercool_locs

        pres_sfc.close()
        pres_nat_lev.close()
        temp.close()
        cloud_mixing_ratio.close()
        rain_mixing_ratio.close()
        return slw

    @property
    def ticks(self) -> int:
        """
        Returns the number of color bar tick marks from the yaml config
        settings.
        """

        return int(self.vspec.get("ticks", 10))

    @property
    def units(self) -> str:
        """
        Returns the variable unit from the yaml config, if available. If not
        specified in the yaml file, returns the value set in the Grib file.
        """

        return str(self.vspec.get("unit", self.field.units))

    def values(
        self, level: str | None = None, name: str | None = None, do_transform: bool = True
    ) -> DataArray:
        """
        Returns the FieldData array of values at the requested level for the
        variable after applying any unit conversion to the original data.

        Optional Input:
            level         the desired level of the named field
            name          the name of a field other than defined in self
            do_transform  apply a standard transformation of units, etc.?

        """

        level = str(level or self.level)
        vals = self.field
        spec = self.vspec

        if name is not None:
            # Get the spec dict and ncl_name for the given variable name
            spec = deepcopy(self.spec.get(name, {}).get(level, {}))
            if not spec:
                raise errors.NoGraphicsDefinitionForVariableError(name, level)
            utils.set_level(level=level, model=self.model, spec=spec)
        vals = self._get_field(spec["cfgrib"].get(self.model, spec["cfgrib"]))

        transforms = spec.get("transform")
        if transforms and do_transform:
            vals = self.get_transform(transforms, vals)

        return vals if "global" not in self.model else vals[::-1, :]


class ProfileData(UPPData):
    """
    Class provides methods for getting profiles from a specific lat/lon location
    from a grib file.

    Input:

      ds           xarray dataset from grib file
      loc          single entry from sites file. Use the first 31 spaces to get
                   site_code, site_num, lat, lon. Past 31 spaces is the site's
                   long name.
      short_name

    Key word arguments:

      Only used for base classes.

    """

    def __init__(
        self,
        fhr: int,
        ds: dict[str, Dataset],
        model: str,
        loc: str,
        short_name: str,
        spec: dict | YAMLConfig,
        level: str | None = None,
    ):
        super().__init__(
            fhr=fhr,
            ds=ds,
            level=level or "ua",
            model=model,
            short_name=short_name,
            spec=spec,
        )

        self.loc = loc
        # The first 31 columns are space delimted
        self.site_code, _, self.site_num, lat, lon = loc[:31].split()

        # The variable lenght site name is included past column 37
        self.site_name = loc[37:].rstrip()

        # Convert the string to a number. Longitude should be positive for all
        # these sites.
        # The conus_raobs file uses -180 to 180, but leaves off the minus sign,
        # i.e., the values are in degrees West. So, first we need to add the
        # minus sign to convert the longitude to deg East, and then need to
        # adjust to the 0 to 360 system.
        self.site_lat = float(lat)
        self.site_lon = -float(lon)  # lons are -180 but without minus sign in input file
        if self.site_lon < 0:
            self.site_lon = self.site_lon + 360.0

    def values(
        self,
        level: str | None = None,
        name: str | None = None,
        do_transform: bool = False,
    ) -> DataArray:
        """
        Returns the numpy array of values at the object's x, y location for the
        requested variable.

        Optional Input:
            name       the short name of a field other than defined in self
            level      the level of the alternate field to use, default='ua' for
                       upper air

        """

        assert do_transform is False  # not supported by this class

        # Set the defaults here since this is an instance of an abstract method
        # level refers to the level key in the specs file.
        level = level if level is not None else "ua"

        if name is not None:
            # Get the spec dict and ncl_name for the given variable name
            spec = deepcopy(self.spec.get(name, {}).get(level, {}))
            if not spec:
                raise errors.NoGraphicsDefinitionForVariableError(name, level)
            utils.set_level(level=level, model=self.model, spec=spec)
            profile = self._get_field(spec["cfgrib"].get(self.model, spec["cfgrib"])).squeeze()
        else:
            profile = self.field.squeeze()
        # Retrive the location for the profile
        x, y = self.get_xypoint(self.site_lat, self.site_lon)

        # 2D
        if len(profile.shape) == 2:
            profile = profile[x, y]
        # 3D
        elif len(profile.shape) == 3:
            profile = profile[:, x, y]
        return profile
