import numpy as np
import abc
import xarray as xr
import paleoclim_leeds.util_hadcm3 as util
import cftime


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
    
    def __init__(self, verbose, logger):
        """
        Parameters
        ----------
        verbose : bool, optional
              Determine whether to print the outputs in a logfile or in directly on the console.
              True is console - debug mode.  (default is False)
        """
        self.logger = logger
        self.verbose = verbose


class ModelDS(GeoDS):
    
    def __init__(self, verbose, logger):
        """
        Parameters
        ----------
        verbose : bool, optional
            whether or not to display details about the computation.
            Outputs are printed.  (default is False)
        """
        
        super(ModelDS, self).__init__(verbose, logger)
        self.lon, self.lat, self.z = None, None, None
        self.lonb, self.latb, self.zb = None, None, None
        self.lons, self.lats, self.zs = None, None, None
        self.lon_p, self.lat_p, self.z_p = None, None, None
        self.lonb_p, self.latb_p, self.zb_p = None, None, None
        self.lons_p, self.lats_p, self.zs_p = None, None, None
        self.start_year = None
        self.end_year = None
    
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
    
    def guess_bounds(self):
        self.lonb = util.guess_bounds(self.lon)
        self.latb = util.guess_bounds(self.lat)
        self.zb = util.guess_bounds(self.z)


def filter_months(data_array, month_list):
    # To define in GeoDataArray !!!!and GeoDS!!!!
    if month_list is not None:
        condition = xr.zeros_like(data_array.t)
        for i in range(len(data_array.t)):
            condition[i] = data_array.t[i].values[()].month in util.months_to_number(month_list)
        data_array = data_array.where(condition, drop=True)
    return data_array


class GeoDataArray:
    
    def __init__(self, data_input, ds=None, coords=None, dims=None, name=None, attrs=None, indexes=None,
                 fastpath=False, transform=None):
        if isinstance(data_input, xr.DataArray):
            self.data = xr.DataArray(data_input.values, dims=data_input.dims, name=data_input.name,
                                     attrs=data_input.attrs, coords=[data_input[dim].values for dim in data_input.dims])
        else:
            self.data = xr.DataArray(data_input, coords=coords, dims=dims, name=name, attrs=attrs,
                                     indexes=indexes, fastpath=fastpath)
        self.sort_data()
        print("____ Data imported in the GeoDataArray instance.")
        
        self.lon = np.sort(ds.lon) if ds is not None and ds.lon is not None else np.sort(self.data.longitude)
        self.lat = np.sort(ds.lat) if ds is not None and ds.lat is not None else np.sort(self.data.latitude)
        self.z = np.sort(ds.z) if ds is not None and ds.z is not None else None
        self.lonb, self.latb, self.zb = np.sort(ds.lonb) if ds is not None and ds.lonb is not None else None, \
                                        np.sort(ds.latb) if ds is not None and ds.latb is not None else None, \
                                        np.sort(ds.zb) if ds is not None and ds.zb is not None else None
        self.lons, self.lats, self.zs = np.sort(ds.lons) if ds is not None and ds.lons is not None else None, \
                                        np.sort(ds.lats) if ds is not None and ds.lats is not None else None, \
                                        np.sort(ds.zs) if ds is not None and ds.zs is not None else None
        self.lon_p, self.lat_p, self.z_p = np.sort(ds.lon_p) if ds is not None and ds.lon_p is not None else None, \
                                           np.sort(ds.lat_p) if ds is not None and ds.lat_p is not None else None, \
                                           np.sort(ds.z_p) if ds is not None and ds.z_p is not None else None
        self.lonb_p, self.latb_p, self.zb_p = np.sort(ds.lonb_p) if ds is not None and ds.lonb_p is not None else None, \
                                              np.sort(ds.latb_p) if ds is not None and ds.latb_p is not None else None, \
                                              np.sort(ds.zb_p) if ds is not None and ds.zb_p is not None else None
        self.lons_p, self.lats_p, self.zs_p = np.sort(ds.lons_p) if ds is not None and ds.lons_p is not None else None, \
                                              np.sort(ds.lats_p) if ds is not None and ds.lats_p is not None else None, \
                                              np.sort(ds.zs_p) if ds is not None and ds.zs_p is not None else None
        self.t = np.sort(ds.t) if ds is not None else None
        self.transform = transform
        
        print("____ Coordinate imported in the GeoDataArray instance.")
    
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
        return self.transform(self.data.where(self.data.values != 0)).values if processing \
            else self.data.where(self.data.values != 0).values
    
    def sort_data(self):
        """
        Sort all dimensions of the data.
        :return:
        """
        for dim in self.data.dims:
            self.data = self.data.sortby(dim, ascending=True)
    
    def get_lon(self, mode_lon, value_lon):
        
        try:
            if mode_lon is None:
                pass
            elif 'longitude' in self.data.dims:
                if mode_lon == "index":
                    if value_lon is None:
                        raise ValueError("**** To use the index mode, please indicate a value_lon.")
                    print(f"____ New longitude value : {self.lon[int(value_lon)]}")
                    self.data = self.data.isel(longitude=value_lon)
                elif mode_lon == "value":
                    if value_lon is None:
                        raise ValueError("**** To use the value mode, please indicate a value_lon.")
                    new_lon = self.lon[util.lon_to_index(self.lon, value_lon)]
                    print(
                        f"____ New longitude value : {new_lon}")
                    self.data = self.data.isel(longitude=util.lon_to_index(self.lon, value_lon))
                elif mode_lon == "mean":
                    self.data = self.data.mean(dim="longitude", skipna=True)
                elif mode_lon == "min":
                    self.data = self.data.min(dim="longitude", skipna=True)
                elif mode_lon == "max":
                    self.data = self.data.max(dim="longitude", skipna=True)
                elif mode_lon == "median":
                    self.data = self.data.median(dim="longitude", skipna=True)
                elif mode_lon == "sum":
                    self.data = self.data.sum(dim="longitude", skipna=True)
                else:
                    print("**** Mode wasn't recognized. The data_array was not changed.")
                self.update_lon(mode_lon, value_lon)
            
            elif 'longitudeb' in self.data.dims:
                if mode_lon == "index":
                    if value_lon is None:
                        raise ValueError("**** To use the index mode, please indicate a value_lon.")
                    print(f"____ New longitude value : {self.lonb[int(value_lon)]}")
                    self.data = self.data.isel(longitudeb=value_lon)
                elif mode_lon == "value":
                    if value_lon is None:
                        raise ValueError("**** To use the value mode, please indicate a value_lon.")
                    new_lon = self.lonb[util.lon_to_index(self.lonb, value_lon)]
                    print(
                        f"____ New longitude value : {new_lon}")
                    self.data = self.data.isel(longitudeb=util.lon_to_index(self.lonb, value_lon))
                elif mode_lon == "mean":
                    self.data = self.data.mean(dim="longitudeb", skipna=True)
                elif mode_lon == "min":
                    self.data = self.data.min(dim="longitudeb", skipna=True)
                elif mode_lon == "max":
                    self.data = self.data.max(dim="longitudeb", skipna=True)
                elif mode_lon == "median":
                    self.data = self.data.median(dim="longitudeb", skipna=True)
                elif mode_lon == "sum":
                    self.data = self.data.sum(dim="longitudeb", skipna=True)
                else:
                    print("**** Mode wasn't recognized. The data_array was not changed.")
                self.update_lon(mode_lon, value_lon)
        
        except ValueError as error:
            print(error)
            print("____ The DataArray was not changed.")
        except IndexError as error:
            print(error)
            print("**** The longitude index was out of bound, the DataArray was not changed")
        finally:
            return self
    
    def update_lon(self, mode_lon, value_lon):
        if mode_lon is None:
            pass
        elif mode_lon == "index":
            if value_lon is None:
                raise ValueError("**** To use the index mode, please indicate a value_lon.")
            self.lon = self.lon[int(value_lon)] if 'longitude' in self.data.dims else self.lonb[int(value_lon)]
            self.lonb, self.lons = self.lon, None
            self.lon_p, self.lonb_p, self.lons_p = self.lon, self.lon, None
        elif mode_lon == "value":
            if value_lon is None:
                raise ValueError("**** To use the value mode, please indicate a value_lon.")
            # Take the closest longitude
            new_lon = self.lon[util.lon_to_index(self.lon, value_lon)] if 'longitude' in self.data.dims else self.lonb[
                util.lon_to_index(self.lonb, value_lon)]
            self.lon = new_lon
            self.lonb, self.lons = self.lon, None
            self.lon_p, self.lonb_p, self.lons_p = self.lon, self.lon, None
        elif mode_lon in ["mean", "min", "max", "median", "sum"]:
            self.lon, self.lonb, self.lons = None, None, None
            self.lon_p, self.lonb_p, self.lons_p = None, None, None
        else:
            print("**** Mode wasn't recognized. The data_array was not changed.")
    
    def get_lat(self, mode_lat, value_lat):
        try:
            if mode_lat is None:
                pass
            elif 'latitude' in self.data.dims:
                if mode_lat == "index":
                    if value_lat is None:
                        raise ValueError("**** To use the index mode, please indicate a value_lat.")
                    print(f"____ New latitude value : {self.lat[int(value_lat)]}")
                    self.data = self.data.isel(latitude=value_lat)
                elif mode_lat == "value":
                    if value_lat is None:
                        raise ValueError("**** To use the value mode, please indicate a value_lat.")
                    new_lat = self.lat[util.lat_to_index(self.lat, value_lat)]
                    print(
                        f"____ New latitude value : {new_lat}")
                    self.data = self.data.isel(latitude=util.lat_to_index(self.lat, value_lat))
                elif mode_lat == "mean":
                    self.data = self.data.mean(dim="latitude", skipna=True)
                elif mode_lat == "min":
                    self.data = self.data.min(dim="latitude", skipna=True)
                elif mode_lat == "max":
                    self.data = self.data.max(dim="latitude", skipna=True)
                elif mode_lat == "median":
                    self.data = self.data.median(dim="latitude", skipna=True)
                elif mode_lat == "sum":
                    self.data = self.data.sum(dim="latitude", skipna=True)
                else:
                    print("**** Mode wasn't recognized. The data_array was not changed.")
                self.update_lat(mode_lat, value_lat)
            
            elif 'latitudeb' in self.data.dims:
                if mode_lat == "index":
                    if value_lat is None:
                        raise ValueError("**** To use the index mode, please indicate a value_lat.")
                    print(f"____ New latitude value : {self.latb[int(value_lat)]}")
                    self.data = self.data.isel(latitudeb=value_lat)
                elif mode_lat == "value":
                    if value_lat is None:
                        raise ValueError("**** To use the value mode, please indicate a value_lat.")
                    new_lat = self.latb[util.lat_to_index(self.latb, value_lat)]
                    print(
                        f"____ New latitude value : {new_lat}")
                    self.data = self.data.isel(latitudeb=util.lat_to_index(self.latb, value_lat))
                elif mode_lat == "mean":
                    self.data = self.data.mean(dim="latitudeb", skipna=True)
                elif mode_lat == "min":
                    self.data = self.data.min(dim="latitudeb", skipna=True)
                elif mode_lat == "max":
                    self.data = self.data.max(dim="latitudeb", skipna=True)
                elif mode_lat == "median":
                    self.data = self.data.median(dim="latitudeb", skipna=True)
                elif mode_lat == "sum":
                    self.data = self.data.sum(dim="latitudeb", skipna=True)
                else:
                    print("**** Mode wasn't recognized. The data_array was not changed.")
                self.update_lat(mode_lat, value_lat)
        
        except ValueError as error:
            print(error)
            print("____ The DataArray was not changed.")
        except IndexError as error:
            print(error)
            print("**** The latitude index was out of bound, the DataArray was not changed")
        finally:
            return self
    
    def update_lat(self, mode_lat, value_lat):
        if mode_lat is None:
            pass
        elif mode_lat == "index":
            if value_lat is None:
                raise ValueError("**** To use the index mode, please indicate a value_lat.")
            print(f"____ New latitude value : {self.lat[int(value_lat)]}")
            self.lat = self.lat[int(value_lat)] if 'latitude' in self.data.dims else self.latb[int(value_lat)]
            self.latb, self.lats = self.lat, None
            self.lat_p, self.latb_p, self.lats_p = self.lat, self.lat, None
        elif mode_lat == "value":
            if value_lat is None:
                raise ValueError("**** To use the value mode, please indicate a value_lat.")
            # Take the closest latitude
            new_lat = self.lat[util.lat_to_index(self.lat, value_lat)] if 'latitudeb' in self.data.dims else self.latb[
                util.lat_to_index(self.latb, value_lat)]
            self.lat = new_lat
            self.latb, self.lats = self.lat, None
            self.lat_p, self.latb_p, self.lats_p = self.lat, self.lat, None
        elif mode_lat in ["mean", "min", "max", "median", "sum"]:
            self.lat, self.latb, self.lats = None, None, None
            self.lat_p, self.latb_p, self.lats_p = None, None, None
        else:
            print("**** Mode wasn't recognized. The data_array was not changed.")
    
    def get_z(self, mode_z, value_z):
        try:
            if mode_z is None:
                pass
            
            elif 'z' in self.data.dims:
                if mode_z == "index":
                    if value_z is None:
                        raise ValueError("**** To use the index mode, please indicate a value_z.")
                    print(f"____ New z value : {self.z[int(value_z)]}")
                    self.data = self.data.isel(z=value_z)
                elif mode_z == "value":
                    if value_z is None:
                        raise ValueError("**** To use the value mode, please indicate a value_z.")
                    new_z = self.z[util.z_to_index(self.z, value_z)]
                    print(
                        f"New z value : {new_z}")
                    self.data = self.data.isel(z=util.z_to_index(self.z, value_z))
                elif mode_z == "mean":
                    self.data = self.data.mean(dim="z", skipna=True)
                elif mode_z == "min":
                    self.data = self.data.min(dim="z", skipna=True)
                elif mode_z == "max":
                    self.data = self.data.max(dim="z", skipna=True)
                elif mode_z == "median":
                    self.data = self.data.median(dim="z", skipna=True)
                elif mode_z == "sum":
                    self.data = self.data.sum(dim="z", skipna=True)
                else:
                    print("**** Mode wasn't recognized. The data_array was not changed.")
                self.update_z(mode_z, value_z)
            
            elif 'zb' in self.data.dims:
                if mode_z == "index":
                    if value_z is None:
                        raise ValueError("**** To use the index mode, please indicate a value_z.")
                    print(f"____ New z value : {self.zb[int(value_z)]}")
                    self.data = self.data.isel(zb=value_z)
                elif mode_z == "value":
                    if value_z is None:
                        raise ValueError("**** To use the value mode, please indicate a value_z.")
                    new_z = self.zb[util.z_to_index(self.zb, value_z)]
                    print(
                        f"New z value : {new_z}")
                    self.data = self.data.isel(zb=util.z_to_index(self.zb, value_z))
                elif mode_z == "mean":
                    self.data = self.data.mean(dim="zb", skipna=True)
                elif mode_z == "min":
                    self.data = self.data.min(dim="zb", skipna=True)
                elif mode_z == "max":
                    self.data = self.data.max(dim="zb", skipna=True)
                elif mode_z == "median":
                    self.data = self.data.median(dim="zb", skipna=True)
                elif mode_z == "sum":
                    self.data = self.data.sum(dim="zb", skipna=True)
                else:
                    print("**** Mode wasn't recognized. The data_array was not changed.")
                self.update_z(mode_z, value_z)
        
        except ValueError as error:
            print(error)
            print("____ The DataArray was not changed.")
        except IndexError as error:
            print(error)
            print("**** The z index was out of bound, the DataArray was not changed")
        finally:
            return self
    
    def update_z(self, mode_z, value_z):
        if mode_z is None:
            pass
        elif mode_z == "index":
            if value_z is None:
                raise ValueError("**** To use the index mode, please indicate a value_z.")
            self.z = self.z[int(value_z)] if 'z' in self.data.dims else self.zb[int(value_z)]
            self.zb, self.zs = self.z, None
            self.z_p, self.zb_p, self.zs_p = self.z, self.z, None
        elif mode_z == "value":
            if value_z is None:
                raise ValueError("**** To use the value mode, please indicate a value_z.")
            # Take the closest z
            new_z = self.z[util.z_to_index(self.z, value_z)] if 'z' in self.data.dims else self.zb[
                util.z_to_index(self.zb, value_z)]
            self.z = new_z
            self.zb, self.zs = self.z, None
            self.z_p, self.zb_p, self.zs_p = self.z, self.z, None
        elif mode_z in ["mean", "min", "max", "median", "sum"]:
            self.z, self.zb, self.zs = None, None, None
            self.z_p, self.zb_p, self.zs_p = None, None, None
        else:
            print("**** Mode wasn't recognized. The data_array was not changed.")
    
    def get_t(self, mode_t, value_t):
        try:
            if mode_t is None:
                pass
            elif mode_t == "index":
                if value_t is None:
                    raise ValueError("**** To use the index mode, please indicate a value_t.")
                print(f"____ New t value : {self.t[int(value_t)]}")
                self.data = self.data.isel(t=value_t)
            elif mode_t == "value":
                if value_t is None:
                    raise ValueError("**** To use the value mode, please indicate a value_t.")
                new_t = self.t[util.t_to_index(self.t, value_t)]
                print(
                    f"____ New t value : {new_t}")
                self.data = self.data.isel(t=util.t_to_index(self.t, value_t))
            elif mode_t == "mean":
                self.data = self.data.mean(dim="t", skipna=True)
            elif mode_t == "min":
                self.data = self.data.min(dim="t", skipna=True)
            elif mode_t == "max":
                self.data = self.data.max(dim="t", skipna=True)
            elif mode_t == "median":
                self.data = self.data.median(dim="t", skipna=True)
            elif mode_t == "sum":
                self.data = self.data.sum(dim="t", skipna=True)
            else:
                print("**** Mode wasn't recognited. The data_array was not changed.")
        except ValueError as error:
            print(error)
            print("____ The DataArray was not changed.")
        except IndexError as error:
            print(error)
            print("**** The t index was out of bound, the DataArray was not changed.")
        finally:
            return self
    
    def crop_months(self, new_month_list):
        condition = xr.zeros_like(self.data.t)
        for i in range(len(self.data.t)):
            condition[i] = self.data.t[i].values[()].month in util.months_to_number(new_month_list)
        self.data = self.data.where(condition, drop=True)
        print("____ Data cropped to the new month list.")
        return self
    
    def crop_years(self, new_start_year, new_end_year):
        if new_start_year is not None:
            self.data = self.data.where(self.data.t >= cftime.Datetime360Day(new_start_year, 1, 1),
                                        drop=True)
        if new_end_year is not None:
            self.data = self.data.where(self.data.t <= cftime.Datetime360Day(new_end_year, 12, 30),
                                        drop=True)
        print("____ Data cropped to the new start and end years.")
        return self
    
    # def fit_coordinates_to_data(self):
    # DEPRECATED
    #     try:
    #         self.lon = self.data.longitude.values
    #     except AttributeError:
    #         self.lon = None
    #     try:
    #         self.lat = self.data.latitude.values
    #     except AttributeError:
    #         self.lat = None
    #     try:
    #         self.z = self.data.z.values
    #     except AttributeError:
    #         self.z = None
    #     try:
    #         self.t = self.data.t.values
    #     except AttributeError:
    #         self.t = None
    #     self.lonb = util.guess_bounds(self.lon, "lon")
    #     self.latb = util.guess_bounds(self.lat, "lat")
    #     self.zb = util.guess_bounds(self.z, "z")
    #
    #     print("____ Coordinates cropped to the new data.")


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
