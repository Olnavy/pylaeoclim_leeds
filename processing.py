import numpy as np
import abc
import xarray as xr
import util_hadcm3 as util
import cftime
import hadcm3_processing as hcm3


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
        self.lon = None
        self.lat = None
        self.z = None
        self.lon_b = None
        self.lat_b = None
        self.z_b = None
        self.lsm = None
        self.start_year = None
        self.end_year = None
    
    @abc.abstractmethod
    def import_data(self, experiment):
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


def filter_months(data_array, month_list):
    # To define in GeoDataArray !!!!and GeoDS!!!!
    if month_list is not None:
        condition = xr.zeros_like(data_array.t)
        for i in range(len(data_array.t)):
            condition[i] = data_array.t[i].values[()].month in util.months_to_number(month_list)
        data_array = data_array.where(condition, drop=True)
    return data_array


class GeoDataArray:
    
    def __init__(self, data_input, ds=None, coords=None, dims=None, name=None, attrs=None, encoding=None, indexes=None,
                 fastpath=False):
        if isinstance(data_input, xr.DataArray):
            self.data = xr.DataArray(data_input.values, dims=data_input.dims, name=data_input.name,
                                     attrs=data_input.attrs, coords=[data_input[dim].values for dim in data_input.dims])
        else:
            self.data = xr.DataArray(data_input, coords=coords, dims=dims, name=name, attrs=attrs, encoding=encoding,
                                     indexes=indexes, fastpath=fastpath)
        
        self.lon = None
        self.lon_b = None
        self.lat = None
        self.lat_b = None
        self.z = None
        self.z_b = None
        self.t = None
        
        if ds is not None:
            self.import_coordinates_from_data_set(ds)
        else:
            self.import_coordinates_from_data_array(self.data)
    
    def __repr__(self):
        return f"DATA: {self.data}"
    
    def values(self):
        return self.data.values
    
    def import_coordinates_from_data_set(self, ds):
        try:
            self.lon = ds.lon
        except AttributeError:
            print("in again")
            self.lon = None
        try:
            self.lon_b = ds.lon_b
        except AttributeError:
            self.lon_b = None
        try:
            self.lat = ds.lat
        except AttributeError:
            self.lat = None
        try:
            self.lat_b = ds.lat_b
        except AttributeError:
            self.lat_b = None
        try:
            self.z = ds.z
        except AttributeError:
            self.z = None
        try:
            self.z_b = ds.z_b
        except AttributeError:
            self.z_b = None
        try:
            self.t = ds.t
        except AttributeError:
            self.t = None
            
    def import_coordinates_from_data_array(self, da):
        try:
            self.lon = da.longitude.values
        except AttributeError:
            self.lon = None
        try:
            self.lat = da.latitude.values
        except AttributeError:
            self.lat = None
        try:
            self.z = da.z.values
        except AttributeError:
            self.z = None
        try:
            self.t = da.t.values
        except AttributeError:
            self.t = None

        self.lon_b = util.guess_bounds(self.lon)
        self.lat_b = util.guess_bounds(self.lat)
        self.z_b = util.guess_bounds(self.z)
    
    def get_lon(self, mode_lon, value_lon):
        try:
            if mode_lon is None:
                pass
            elif mode_lon == "index":
                if value_lon is None:
                    raise ValueError("To use the index mode, please indicate a value_lon.")
                print(f"New longitude value : {self.lon[int(value_lon)]}")
                self.data = self.data.isel(longitude=value_lon)
            elif mode_lon == "value":
                if value_lon is None:
                    raise ValueError("To use the value mode, please indicate a value_lon.")
                new_lon = self.lon[util.lon_to_index(self.lon, value_lon)]
                print(
                    f"New longitude value : {new_lon}")
                self.data = self.data.isel(longitude=util.lon_to_index(self.lon, value_lon))
            elif mode_lon == "mean":
                self.data = self.data.mean(dim="longitude")
            else:
                print("Mode wasn't recognized. The data_array was not changed.")
            self.update_lon()
        except ValueError as error:
            print(error)
            print("The DataArray was not changed.")
        except IndexError as error:
            print(error)
            print("The longitude index was out of bound, the DataArray was not changed")
        finally:
            return self
    
    def get_lat(self, mode_lat, value_lat):
        try:
            if mode_lat is None:
                pass
            elif mode_lat == "index":
                if value_lat is None:
                    raise ValueError("To use the index mode, please indicate a value_lat.")
                print(f"New latitude value : {self.lat[int(value_lat)]}")
                self.data = self.data.isel(latitude=value_lat)
            elif mode_lat == "value":
                if value_lat is None:
                    raise ValueError("To use the value mode, please indicate a value_lat.")
                new_lat = self.lat[util.lat_to_index(self.lat, value_lat)]
                print(
                    f"New latitude value : {new_lat}")
                self.data = self.data.isel(latitude=util.lat_to_index(self.lat, value_lat))
            elif mode_lat == "mean":
                self.data = self.data.mean(dim="latitude")
            else:
                print("Mode wasn't recognized. The data_array was not changed.")
            self.update_lat()
        except ValueError as error:
            print(error)
            print("The DataArray was not changed.")
        except IndexError as error:
            print(error)
            print("The latitude index was out of bound, the DataArray was not changed")
        finally:
            return self
    
    def get_z(self, mode_z, value_z):
        try:
            if mode_z is None:
                pass
            elif mode_z == "index":
                if value_z is None:
                    raise ValueError("To use the index mode, please indicate a value_z.")
                print(f"New z value : {self.z[int(value_z)]}")
                self.data = self.data.isel(z=value_z)
            elif mode_z == "value":
                if value_z is None:
                    raise ValueError("To use the value mode, please indicate a value_z.")
                new_z = self.z[util.z_to_index(self.z, value_z)]
                print(
                    f"New z value : {new_z}")
                self.data = self.data.isel(z=util.z_to_index(self.z, value_z))
            elif mode_z == "mean":
                self.data = self.data.mean(dim="z")
            else:
                print("Mode wasn't recognized. The data_array was not changed.")
            self.update_z()
        except ValueError as error:
            print(error)
            print("The DataArray was not changed.")
        except IndexError as error:
            print(error)
            print("The z index was out of bound, the DataArray was not changed")
        finally:
            return self

    def get_t(self, mode_t, value_t):
        try:
            if mode_t is None:
                pass
            elif mode_t == "index":
                if value_t is None:
                    raise ValueError("To use the index mode, please indicate a value_t.")
                print(f"New t value : {self.t[int(value_t)]}")
                self.data = self.data.isel(t=value_t)
            elif mode_t == "value":
                if value_t is None:
                    raise ValueError("To use the value mode, please indicate a value_t.")
                new_t = self.t[util.t_to_index(self.t, value_t)]
                print(
                    f"New t value : {new_t}")
                self.data = self.data.isel(t=util.t_to_index(self.t, value_t))
            elif mode_t == "mean":
                self.data = self.data.mean(dim="t")
            else:
                print("Mode wasn't recognited. The data_array was not changed.")
        except ValueError as error:
            print(error)
            print("The DataArray was not changed.")
        except IndexError as error:
            print(error)
            print("The t index was out of bound, the DataArray was not changed")
        finally:
            return self

    def truncate_months(self, new_month_list):
        condition = xr.zeros_like(self.data.t)
        for i in range(len(self.data.t)):
            condition[i] = self.data.t[i].values[()].month in util.months_to_number(new_month_list)
        self.data = self.data.where(condition, drop=True)
        return self
    
    def truncate_years(self, new_start_year, new_end_year):
        if new_start_year is not None:
            self.data = self.data.where(self.data.t >= cftime.Datetime360Day(new_start_year, 1, 1),
                                        drop=True)
        if new_end_year is not None:
            self.data = self.data.where(self.data.t <= cftime.Datetime360Day(new_end_year, 12, 30),
                                        drop=True)
        return self
    
    def fit_coordinates_to_data(self):
        try:
            self.lon = self.data.longitude.values
        except AttributeError:
            self.lon = None
        try:
            self.lat = self.data.latitude.values
        except AttributeError:
            self.lat = None
        try:
            self.z = self.data.z.values
        except AttributeError:
            self.z = None
        try:
            self.t = self.data.t.values
        except AttributeError:
            self.t = None
        self.lon_b = util.guess_bounds(self.lon)
        self.lat_b = util.guess_bounds(self.lat)
        self.z_b = util.guess_bounds(self.z)
        
        pass
        
    def update_lon(self):
        try:
            self.lon = self.data.longitude.values
            self.lon_b = util.guess_bounds(self.lon)
        except AttributeError:
            self.lon = None
            self.lon_b = None
    
    def update_lat(self):
        try:
            self.lat = self.data.latitude.values
            self.lat_b = util.guess_bounds(self.lon)
        except AttributeError:
            self.lat = None
            self.lat_b = None
    
    def update_z(self):
        try:
            self.z = self.data.z.values
            self.z_b = util.guess_bounds(self.z)
        except AttributeError:
            self.z = None
            self.z_b = None


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
