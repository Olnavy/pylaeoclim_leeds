import numpy as np
import abc
import xarray as xr
import util_hadcm3 as util
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
    condition = xr.zeros_like(data_array.t)
    for i in range(len(data_array.t)):
        condition[i] = data_array.t[i].values[()].month in util.months_to_number(month_list)
    data_array = data_array.where(condition, drop=True)
    return data_array


class GeoDataArray(xr.DataArray):
    
    def __init__(self, data, coords=None, dims=None, name=None, attrs=None, encoding=None, indexes=None,
                 fastpath=False):
        if isinstance(data, xr.DataArray):
            super(GeoDataArray, self).__init__(data.values, dims=data.dims, name=data.name, attrs=data.attrs,
                                               coords=[data[dim].values for dim in data.dims])
        else:
            super(GeoDataArray, self).__init__(data, coords=coords, dims=dims, name=name, attrs=attrs,
                                               encoding=encoding, indexes=indexes, fastpath=fastpath)
    
    def truncate_months(self, new_month_list):
        condition = xr.zeros_like(self.t)
        for i in range(len(self.t)):
            condition[i] = self.t[i].values[()].month in util.months_to_number(new_month_list)
        return GeoDataArray(self.where(condition, drop=True))
    
    def truncate_years(self, new_start_year, new_end_year):
        data_array = self
        if new_start_year is not None:
            data_array = self.where(self.t >= cftime.Datetime360Day(new_start_year, 1, 1), drop=True)
        if new_end_year is not None:
            data_array = self.where(self.t <= cftime.Datetime360Day(new_end_year, 12, 30), drop=True)
        return GeoDataArray(data_array)


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
