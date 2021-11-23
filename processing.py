import numpy as np
import abc
import xarray as xr
import pylaeoclim_leeds.util_hadcm3 as util
import cftime
import time


# @xr.register_dataset_accessor("geo")
class GeoDA:

    def __init__(self):
        pass




























import matplotlib.colors
class GeoDS:
    """
    Mother class to treat all files (proxies, model outputs...).
    Should fill it after treating an example of proxy file..

    ...

    Attributes
    ----------
    verbose : bool
          Determine whether to print the outputs in a logfile or in directly on the console.
          True is console - debug mode.  (default is False)

    Methods
    -------
    """
    
    def __init__(self, verbose, debug, logger):
        """
        Parameters
        ----------
        verbose : bool, optional
              Determine whether to print the outputs in a logfile or in directly on the console.
              True is console - debug mode.  (default is False)
        """
        
        self.verbose = verbose
        self.debug = debug
        self.logger = logger


class ModelDS(GeoDS):
    
    def __init__(self, verbose, debug, logger):
        """
        Parameters
        ----------
        verbose : bool, optional
            whether or not to display details about the computation.
            Outputs are printed.  (default is False)
        """
        
        super(ModelDS, self).__init__(verbose, debug, logger)
        self.lon, self.lat, self.z = None, None, None
        self.lonb, self.latb, self.zb = None, None, None
        self.lons, self.lats, self.zs = None, None, None
        self.lon_p, self.lat_p, self.z_p = None, None, None
        self.lonb_p, self.latb_p, self.zb_p = None, None, None
        self.lons_p, self.lats_p, self.zs_p = None, None, None
        self.start_year, self.end_year = None, None
        self.t = None
    
    @abc.abstractmethod
    def import_data(self):
        pass
    
    def to_ncdf(self):
        """
        Save the dataset as a netcdf file
        :return:
        """
        
        pass
    
    def to_csv(self):
        """
        Save the dataset as a netcdf file
        :return:
        """
        
        pass
    
    @staticmethod
    def filter_months(data_array, month_list):
        # To define in GeoDataArray !!!!and GeoDS!!!!
        if month_list is not None:
            condition = xr.zeros_like(data_array.t)
            for i in range(len(data_array.t)):
                condition[i] = data_array.t[i].values[()].month in util.months_to_number(month_list)
            data_array = data_array.where(condition, drop=True)
        return data_array
    
    def guess_bounds(self):
        self.lonb = util.guess_bounds(self.lon)
        self.latb = util.guess_bounds(self.lat)
        self.zb = util.guess_bounds(self.z)


class GeoDataArray:
    
    def __init__(self, data_input, ds=None, coords=None, dims=None, name=None, attrs=None, indexes=None,
                 fastpath=False, process=None):
        
        if isinstance(data_input, xr.DataArray):
            self.data = data_input
        else:
            self.data = xr.DataArray(data_input, coords=coords, dims=dims, name=name, attrs=attrs,
                                     indexes=indexes, fastpath=fastpath)
        
        self.sort_data()
        
        # Weird tests
        self.lon = ds.lon if ds is not None else np.sort(self.data.longitude)
        self.lat = ds.lat if ds is not None else np.sort(self.data.latitude)
        self.z = ds.z if ds is not None else None
        self.lonb, self.latb, self.zb = ds.lonb if ds is not None else None, \
                                        ds.latb if ds is not None else None, \
                                        ds.zb if ds is not None else None
        self.lons, self.lats, self.zs = ds.lons if ds is not None else None, \
                                        ds.lats if ds is not None else None, \
                                        ds.zs if ds is not None else None
        self.lon_p, self.lat_p, self.z_p = ds.lon_p if ds is not None else None, \
                                           ds.lat_p if ds is not None else None, \
                                           ds.z_p if ds is not None else None
        self.lonb_p, self.latb_p, self.zb_p = ds.lonb_p if ds is not None else None, \
                                              ds.latb_p if ds is not None else None, \
                                              ds.zb_p if ds is not None else None
        self.lons_p, self.lats_p, self.zs_p = ds.lons_p if ds is not None else None, \
                                              ds.lats_p if ds is not None else None, \
                                              ds.zs_p if ds is not None else None
        self.t = ds.t if ds is not None else None
        self.process = process
        self.proc_lon, self.proc_lat, self.proc_z = True, True, True
        self.start_year = ds.start_year if ds is not None else None
        self.end_year = ds.end_year if ds is not None else None
        self.months = ds.months if ds is not None else None
        self.verbose = ds.verbose if ds is not None else None
        self.debug = ds.verbose if ds is not None else None
        self.logger = ds.verbose if ds is not None else None
        
        print("____ Coordinates imported in the GeoDataArray instance.")
    
    def __repr__(self):
        return f"{util.print_coordinates('lon', self.lon)}; {util.print_coordinates('lon_p', self.lon_p)}\n" \
               f"{util.print_coordinates('lonb', self.lonb)}; {util.print_coordinates('lonb_p', self.lonb_p)}\n" \
               f"{util.print_coordinates('lons', self.lons)}; {util.print_coordinates('lons_p', self.lons_p)}\n" \
               f"{util.print_coordinates('lat', self.lat)}; {util.print_coordinates('lat_p', self.lat_p)}\n" \
               f"{util.print_coordinates('latb', self.latb)}; {util.print_coordinates('latb_p', self.latb_p)}\n" \
               f"{util.print_coordinates('lats', self.lats)}; {util.print_coordinates('lats_p', self.lats_p)}\n" \
               f"{util.print_coordinates('z', self.z)}; {util.print_coordinates('z_p', self.z_p)}\n" \
               f"{util.print_coordinates('zb', self.zb)}; {util.print_coordinates('zb_p', self.zb_p)}\n" \
               f"{util.print_coordinates('zs', self.zs)}; {util.print_coordinates('zs_p', self.zs_p)}\n" \
               f"{util.print_coordinates('t', self.t)}\n" \
               f"DATA: {self.data}"
    
    def values(self, processing=True):
        if self.debug:
            data = self.process(self.data.where(self.data.values != 0), self.proc_lon, self.proc_lat,
                                self.proc_z).values if processing else self.data.where(self.data.values != 0).values
            return data
        else:
            return self.process(self.data.where(self.data.values != 0), self.proc_lon, self.proc_lat,
                                self.proc_z).values if processing else self.data.where(self.data.values != 0).values
    
    def processed_time(self, new_start_year=None):
        return np.linspace(0, self.end_year - self.start_year, len(self.t)) + \
               (new_start_year if new_start_year is not None else self.start_year)
    
    @staticmethod
    def filter_months(data_array, month_list):
        # To define in GeoDataArray !!!!and GeoDS!!!!
        if month_list is not None:
            condition = xr.zeros_like(data_array.t)
            for i in range(len(data_array.t)):
                condition[i] = data_array.t[i].values[()].month in util.months_to_number(month_list)
            data_array = data_array.where(condition, drop=True)
        return data_array
    
    def sort_data(self):
        """
        Sort all dimensions of the data.
        :return:
        """
        for dim in self.data.dims:
            # if self.debug:print(f"____ Sorting data along dimension : {dim}")
            self.data = self.data.sortby(dim, ascending=True)
    
    def get_lon(self, mode_lon, value_lon, offset_lon=1):
        
        try:
            if mode_lon is None:
                pass
            elif 'longitude' in self.data.dims:
                if mode_lon == "index":
                    if value_lon is None:
                        raise ValueError("!!!! To use the index mode, please indicate a value_lon.")
                    print(f"____ New longitude value : {self.lon[int(value_lon)]}")
                    self.data = self.data.isel(longitude=value_lon)
                elif mode_lon == "value":
                    if value_lon is None:
                        raise ValueError("!!!! To use the value mode, please indicate a value_lon.")
                    new_lon = self.lon[util.lon_to_index(self.lon, value_lon)]
                    print(
                        f"____ New longitude value : {new_lon}")
                    self.data = self.data.isel(longitude=util.lon_to_index(self.lon, value_lon))
                elif mode_lon == "mean":
                    print("____ Processing longitude: mean")
                    self.data = self.data.mean(dim="longitude", skipna=True)
                elif mode_lon == "weighted_mean":
                    # No weights for longitude.
                    print("____ Processing longitude: weighted_mean")
                    self.data = self.data.mean(dim="longitude", skipna=True)
                elif mode_lon == "min":
                    print("____ Processing longitude: min")
                    self.data = self.data.min(dim="longitude", skipna=True)
                elif mode_lon == "max":
                    print("____ Processing longitude: max")
                    self.data = self.data.max(dim="longitude", skipna=True)
                elif mode_lon == "median":
                    print("____ Processing longitude: median")
                    self.data = self.data.median(dim="longitude", skipna=True)
                elif mode_lon == "sum":
                    print("____ Processing longitude: sum")
                    self.data = self.data.sum(dim="longitude", skipna=True)
                else:
                    print("!!!! Mode wasn't recognized. The data_array was not changed.")
                self.update_lon(mode_lon, value_lon)
            
            elif 'longitudeb' in self.data.dims:
                if mode_lon == "index":
                    if value_lon is None:
                        raise ValueError("!!!! To use the index mode, please indicate a value_lon.")
                    print(f"____ New longitudeb value : {self.lonb[int(value_lon)]}")
                    self.data = self.data.isel(longitudeb=value_lon)
                elif mode_lon == "value":
                    if value_lon is None:
                        raise ValueError("!!!! To use the value mode, please indicate a value_lon.")
                    new_lon = self.lonb[util.lon_to_index(self.lonb, value_lon)]
                    print(
                        f"____ New longitudeb value : {new_lon}")
                    self.data = self.data.isel(longitudeb=util.lon_to_index(self.lonb, value_lon))
                elif mode_lon == "mean":
                    print("____ Processing longitudeb: mean")
                    self.data = self.data.mean(dim="longitudeb", skipna=True)
                elif mode_lon == "weighted_mean":
                    # No weights for longitude.
                    print("____ Processing longitudeb: weighted_mean")
                    self.data = self.data.mean(dim="longitudeb", skipna=True)
                elif mode_lon == "min":
                    print("____ Processing longitudeb: min")
                    self.data = self.data.min(dim="longitudeb", skipna=True)
                elif mode_lon == "max":
                    print("____ Processing longitudeb: max")
                    self.data = self.data.max(dim="longitudeb", skipna=True)
                elif mode_lon == "median":
                    print("____ Processing longitudeb: median")
                    self.data = self.data.median(dim="longitudeb", skipna=True)
                elif mode_lon == "sum":
                    print("____ Processing longitude: sum")
                    self.data = self.data.sum(dim="longitudeb", skipna=True)
                else:
                    print("!!!! Mode wasn't recognized. The data_array was not changed.")
                self.update_lon(mode_lon, value_lon)
            
            elif 'row_index' in self.data.dims:
                print("!!!! Impossible to use get_lon method for the moment. The data_array was not changed.")
                if mode_lon == "value":
                    if value_lon is None:
                        raise ValueError("!!!! To use the value mode, please indicate a value_lon.")
                    print(f"____ New longitude value : {value_lon}")
                    self.data.where(value_lon - offset_lon <= self.lon <= value_lon + offset_lon)
        
        except ValueError as error:
            print(error)
            print("____ The DataArray was not changed.")
        except IndexError as error:
            print(error)
            print("!!!! The longitude index was out of bound, the DataArray was not changed")
        finally:
            return self
    
    def update_lon(self, mode_lon, value_lon):
        if mode_lon is None:
            pass
        elif mode_lon == "index":
            if value_lon is None:
                raise ValueError("!!!! To use the index mode, please indicate a value_lon.")
            self.lon = self.lon[int(value_lon)] if 'longitude' in self.data.dims else self.lonb[int(value_lon)]
            self.lonb, self.lons = self.lon, None
            self.lon_p, self.lonb_p, self.lons_p = self.lon, self.lon, None
        elif mode_lon == "value":
            if value_lon is None:
                raise ValueError("!!!! To use the value mode, please indicate a value_lon.")
            # Take the closest longitude
            new_lon = self.lon[util.lon_to_index(self.lon, value_lon)] if 'longitude' in self.data.dims else self.lonb[
                util.lon_to_index(self.lonb, value_lon)]
            self.lon = new_lon
            self.lonb, self.lons = self.lon, None
            self.lon_p, self.lonb_p, self.lons_p = self.lon, self.lon, None
        elif mode_lon in ["weighted_mean", "mean", "min", "max", "median", "sum"]:
            self.lon, self.lonb, self.lons = None, None, None
            self.lon_p, self.lonb_p, self.lons_p = None, None, None
        else:
            print("!!!! Mode wasn't recognized. The data_array was not changed.")
    
    def get_lat(self, mode_lat, value_lat, latitude=None):
        
        try:
            if mode_lat is None:
                pass
            elif 'latitude' in self.data.dims:
                if mode_lat == "index":
                    if value_lat is None:
                        raise ValueError("!!!! To use the index mode, please indicate a value_lat.")
                    print(f"____ New latitude value : {self.lat[int(value_lat)]}")
                    self.data = self.data.isel(latitude=value_lat)
                elif mode_lat == "value":
                    if value_lat is None:
                        raise ValueError("!!!! To use the value mode, please indicate a value_lat.")
                    new_lat = self.lat[util.lat_to_index(self.lat, value_lat)]
                    print(
                        f"____ New latitude value : {new_lat}")
                    self.data = self.data.isel(latitude=util.lat_to_index(self.lat, value_lat))
                elif mode_lat == "mean":
                    print("____ Processing latitude: mean")
                    self.data = self.data.mean(dim="latitude", skipna=True)
                elif mode_lat == "weighted_mean":
                    # proportionnal to cosinus
                    print("____ Processing latitude: weighted_mean")
                    lat_weights = np.cos(np.deg2rad(self.data.latitude))
                    weights = self.data.weighted(lat_weights)
                    self.data = weights.mean("latitude")
                elif mode_lat == "min":
                    print("____ Processing latitude: min")
                    self.data = self.data.min(dim="latitude", skipna=True)
                elif mode_lat == "max":
                    print("____ Processing latitude: max")
                    self.data = self.data.max(dim="latitude", skipna=True)
                elif mode_lat == "median":
                    print("____ Processing latitude: median")
                    self.data = self.data.median(dim="latitude", skipna=True)
                elif mode_lat == "sum":
                    print("____ Processing latitude: sum")
                    self.data = self.data.sum(dim="latitude", skipna=True)
                else:
                    print("!!!! Mode wasn't recognized. The data_array was not changed.")
                self.update_lat(mode_lat, value_lat)
            
            elif 'latitudeb' in self.data.dims:
                if mode_lat == "index":
                    if value_lat is None:
                        raise ValueError("!!!! To use the index mode, please indicate a value_lat.")
                    print(f"____ New latitudeb value : {self.latb[int(value_lat)]}")
                    self.data = self.data.isel(latitudeb=value_lat)
                elif mode_lat == "value":
                    if value_lat is None:
                        raise ValueError("!!!! To use the value mode, please indicate a value_lat.")
                    new_lat = self.latb[util.lat_to_index(self.latb, value_lat)]
                    print(f"____ New latitudeb value : {new_lat}")
                    self.data = self.data.isel(latitudeb=util.lat_to_index(self.latb, value_lat))
                elif mode_lat == "mean":
                    print("____ Processing latitudeb: mean")
                    self.data = self.data.mean(dim="latitudeb", skipna=True)
                elif mode_lat == "weighted_mean":
                    # proportionnal to cosinus
                    print("____ Processing latitudeb: weighted_mean")
                    lat_weights = np.cos(np.deg2rad(self.data.latitudeb))
                    weights = self.data.weighted(lat_weights)
                    self.data = weights.mean("latitudeb")
                elif mode_lat == "min":
                    print("____ Processing latitudeb: min")
                    self.data = self.data.min(dim="latitudeb", skipna=True)
                elif mode_lat == "max":
                    print("____ Processing latitudeb: max")
                    self.data = self.data.max(dim="latitudeb", skipna=True)
                elif mode_lat == "median":
                    print("____ Processing latitudeb: median")
                    self.data = self.data.median(dim="latitudeb", skipna=True)
                elif mode_lat == "sum":
                    print("____ Processing latitudeb: sum")
                    self.data = self.data.sum(dim="latitudeb", skipna=True)
                else:
                    print("!!!! Mode wasn't recognized. The data_array was not changed.")
                self.update_lat(mode_lat, value_lat)
            
            elif 'col_index' in self.data.dims or latitude is None:
                print("!!!! Impossible to use get_lat method for the moment. The data_array was not changed.")
        
        except ValueError as error:
            print(error)
            print("____ The DataArray was not changed.")
        except IndexError as error:
            print(error)
            print("!!!! The latitude index was out of bound, the DataArray was not changed")
        finally:
            return self
    
    def update_lat(self, mode_lat, value_lat):
        if mode_lat is None:
            pass
        elif mode_lat == "index":
            if value_lat is None:
                raise ValueError("!!!! To use the index mode, please indicate a value_lat.")
            print(f"____ New latitude value : {self.lat[int(value_lat)]}")
            self.lat = self.lat[int(value_lat)] if 'latitude' in self.data.dims else self.latb[int(value_lat)]
            self.latb, self.lats = self.lat, None
            self.lat_p, self.latb_p, self.lats_p = self.lat, self.lat, None
        elif mode_lat == "value":
            if value_lat is None:
                raise ValueError("!!!! To use the value mode, please indicate a value_lat.")
            # Take the closest latitude
            new_lat = self.lat[util.lat_to_index(self.lat, value_lat)] if 'latitudeb' in self.data.dims else self.latb[
                util.lat_to_index(self.latb, value_lat)]
            self.lat = new_lat
            self.latb, self.lats = self.lat, None
            self.lat_p, self.latb_p, self.lats_p = self.lat, self.lat, None
        elif mode_lat in ["weighted_mean", "mean", "min", "max", "median", "sum"]:
            self.lat, self.latb, self.lats = None, None, None
            self.lat_p, self.latb_p, self.lats_p = None, None, None
        else:
            print("!!!! Mode wasn't recognized. The data_array was not changed.")
    
    def get_z(self, mode_z, value_z):
        
        try:
            if mode_z is None:
                pass
            
            elif 'z' in self.data.dims:
                if mode_z == "index":
                    if value_z is None:
                        raise ValueError("!!!! To use the index mode, please indicate a value_z.")
                    print(f"____ New z value : {self.z[int(value_z)]}")
                    self.data = self.data.isel(z=value_z)
                elif mode_z == "value":
                    if value_z is None:
                        raise ValueError("!!!! To use the value mode, please indicate a value_z.")
                    new_z = self.z[util.z_to_index(self.z, value_z)]
                    print(
                        f"New z value : {new_z}")
                    self.data = self.data.isel(z=util.z_to_index(self.z, value_z))
                elif mode_z == "mean":
                    print("____ Processing z: mean")
                    self.data = self.data.mean(dim="z", skipna=True)
                elif mode_z == "weighted_mean":
                    # proportionnal to steps.
                    print("____ Processing z: weighted_mean")
                    if len(self.zs) == len(self.z):
                        self.data = self.data.weighted(xr.DataArray(self.zs, dims=["z"])).mean("z")
                    else:
                        zs = util.compute_steps(util.guess_bounds(self.z))
                        self.data = self.data.weighted(xr.DataArray(zs, dims=["z"])).mean("z")
                elif mode_z == "min":
                    print("____ Processing z: min")
                    self.data = self.data.min(dim="z", skipna=True)
                elif mode_z == "max":
                    print("____ Processing z: max")
                    self.data = self.data.max(dim="z", skipna=True)
                elif mode_z == "median":
                    print("____ Processing z: median")
                    self.data = self.data.median(dim="z", skipna=True)
                elif mode_z == "sum":
                    print("____ Processing z: sum")
                    self.data = self.data.sum(dim="z", skipna=True)
                else:
                    print("!!!! Mode wasn't recognized. The data_array was not changed.")
                self.update_z(mode_z, value_z)
            
            elif 'zb' in self.data.dims:
                if mode_z == "index":
                    if value_z is None:
                        raise ValueError("!!!! To use the index mode, please indicate a value_z.")
                    print(f"____ New zb value : {self.zb[int(value_z)]}")
                    self.data = self.data.isel(zb=value_z)
                elif mode_z == "value":
                    if value_z is None:
                        raise ValueError("!!!! To use the value mode, please indicate a value_z.")
                    new_z = self.zb[util.z_to_index(self.zb, value_z)]
                    print(
                        f"New zb value : {new_z}")
                    self.data = self.data.isel(zb=util.z_to_index(self.zb, value_z))
                elif mode_z == "mean":
                    print("____ Processing zb: mean")
                    self.data = self.data.mean(dim="zb", skipna=True)
                elif mode_z == "weighted_mean":
                    # proportionnal to steps.
                    print("____ Processing zb: weighted_mean")
                    if len(self.zs) == len(self.zb):
                        self.data = self.data.weighted(xr.DataArray(self.zs, dims=["zb"])).mean("zb")
                    else:
                        zs = util.compute_steps(util.guess_bounds(self.zb))
                        self.data = self.data.weighted(xr.DataArray(zs, dims=["zb"])).mean("zb")
                elif mode_z == "min":
                    print("____ Processing zb: min")
                    self.data = self.data.min(dim="zb", skipna=True)
                elif mode_z == "max":
                    print("____ Processing zb: max")
                    self.data = self.data.max(dim="zb", skipna=True)
                elif mode_z == "median":
                    print("____ Processing zb: median")
                    self.data = self.data.median(dim="zb", skipna=True)
                elif mode_z == "sum":
                    print("____ Processing zb: sum")
                    self.data = self.data.sum(dim="zb", skipna=True)
                else:
                    print("!!!! Mode wasn't recognized. The data_array was not changed.")
                self.update_z(mode_z, value_z)
        
        except ValueError as error:
            print(error)
            print("____ The DataArray was not changed.")
        except IndexError as error:
            print(error)
            print("!!!! The z index was out of bound, the DataArray was not changed")
        finally:
            return self
    
    def update_z(self, mode_z, value_z):
        if mode_z is None:
            pass
        elif mode_z == "index":
            if value_z is None:
                raise ValueError("!!!! To use the index mode, please indicate a value_z.")
            self.z = self.z[int(value_z)] if 'z' in self.data.dims else self.zb[int(value_z)]
            self.zb, self.zs = self.z, None
            self.z_p, self.zb_p, self.zs_p = self.z, self.z, None
        elif mode_z == "value":
            if value_z is None:
                raise ValueError("!!!! To use the value mode, please indicate a value_z.")
            # Take the closest z
            new_z = self.z[util.z_to_index(self.z, value_z)] if 'z' in self.data.dims else self.zb[
                util.z_to_index(self.zb, value_z)]
            self.z = new_z
            self.zb, self.zs = self.z, None
            self.z_p, self.zb_p, self.zs_p = self.z, self.z, None
        elif mode_z in ["weighted_mean", "mean", "min", "max", "median", "sum"]:
            self.z, self.zb, self.zs = None, None, None
            self.z_p, self.zb_p, self.zs_p = None, None, None
        else:
            print("!!!! Mode wasn't recognized. The data_array was not changed.")
    
    def get_t(self, mode_t, value_t):
        
        try:
            if mode_t is None:
                pass
            elif mode_t == "index":
                if value_t is None:
                    raise ValueError("!!!! To use the index mode, please indicate a value_t.")
                print(f"____ New t value : {self.t[int(value_t)]}")
                self.data = self.data.isel(t=value_t)
            elif mode_t == "value":
                if value_t is None:
                    raise ValueError("!!!! To use the value mode, please indicate a value_t.")
                new_t = self.t[util.t_to_index(self.t, value_t)]
                print(
                    f"____ New t value : {new_t}")
                self.data = self.data.isel(t=util.t_to_index(self.t, value_t))
            elif mode_t == "mean":
                print("____ Processing t: mean")
                self.data = self.data.mean(dim="t", skipna=True)
            elif mode_t == "min":
                print("____ Processing t: min")
                self.data = self.data.min(dim="t", skipna=True)
            elif mode_t == "max":
                print("____ Processing t: max")
                self.data = self.data.max(dim="t", skipna=True)
            elif mode_t == "median":
                print("____ Processing t: median")
                self.data = self.data.median(dim="t", skipna=True)
            elif mode_t == "sum":
                print("____ Processing t: sum")
                self.data = self.data.sum(dim="t", skipna=True)
            else:
                print("!!!! Mode wasn't recognited. The data_array was not changed.")
            self.update_t(mode_t, value_t)
        except ValueError as error:
            print(error)
            print("____ The DataArray was not changed.")
        except IndexError as error:
            print(error)
            print("!!!! The t index was out of bound, the DataArray was not changed.")
        finally:
            return self
    
    def update_t(self, mode_t, value_t):
        if mode_t is None:
            pass
        elif mode_t == "index":
            if value_t is None:
                raise ValueError("!!!! To use the index mode, please indicate a value_t.")
            self.t = self.t[int(value_t)]
        elif mode_t == "value":
            if value_t is None:
                raise ValueError("!!!! To use the value mode, please indicate a value_t.")
            new_t = self.t[util.t_to_index(self.t, value_t)]
            self.z = new_t
        elif mode_t in ["mean", "min", "max", "median", "sum"]:
            self.t = None
        else:
            print("!!!! Mode wasn't recognized. The data_array was not changed.")
    
    def crop_months(self, new_month_list):
        condition = xr.zeros_like(self.data.t)
        for i in range(len(self.data.t)):
            condition[i] = self.data.t[i].values[()].month in util.months_to_number(new_month_list)
        self.data = self.data.where(condition, drop=True)
        self.months = new_month_list
        print("____ Data cropped to the new month list.")
        return self
    
    def crop_years(self, new_start_year, new_end_year):
        if new_start_year is not None:
            self.data = self.data.where(self.data.t >= cftime.Datetime360Day(new_start_year, 1, 1),
                                        drop=True)
            self.start_year = new_start_year
        if new_end_year is not None:
            self.data = self.data.where(self.data.t <= cftime.Datetime360Day(new_end_year, 12, 30),
                                        drop=True)
            self.end_year = new_end_year
        print("____ Data cropped to the new start and end years.")
        return self


class LSM:
    
    def __init__(self):
        self.lon = None
        self.lat = None
        self.z = None
        self.depth = None
        self.level = None
        self.lsm2d = None
        self.mask2d = None
        self.lsm3d = None
        self.mask3d = None
    
    @classmethod
    def default_lsm(cls, lon, lat, z):
        """
        Global function.
        :param lon:
        :param lat:
        :param z:
        :return:
        """
        return np.ones((len(lon), len(lat), len(z)))
    
    @classmethod
    def default_mask(cls, lon, lat, z):
        """
        Global function.
        :param lon:
        :param lat:
        :param z:
        :return:
        """
        return np.zeros((len(lon), len(lat), len(z)))


class Grid:
    
    def __init__(self):
        pass
