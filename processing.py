import numpy as np
import abc
import zones
import xarray as xr
import util_hadcm3 as util


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

    def __init__(self, verbose=False):
        """
        Parameters
        ----------
        verbose : bool, optional
              Determine whether to print the outputs in a logfile or in directly on the console.
              True is console - debug mode.  (default is False)
        """

        self.verbose = verbose


class ModelDS(GeoDS):

    def __init__(self, verbose=False):
        """
        Parameters
        ----------
        verbose : bool, optional
            whether or not to display details about the computation.
            Outputs are printed.  (default is False)
        """

        super(ModelDS, self).__init__(verbose)
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
    def import_data(self, path, experiment):
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
    # To define in GeoDataArray
    condition = xr.zeros_like(data_array.t)
    for i in range(len(data_array.t)):
        condition[i] = data_array.t[i].values[()].month in util.months_to_number(month_list)
    data_array = data_array.where(condition, drop=True)
    return data_array

class GeoDataArray(xr.DataArray):
    
    def __init__(self):
        super(GeoDataArray,self).__init__(self)


    def filter_months(self,month_list):
        condition = xr.zeros_like(self.t)
        for i in range(len(self.t)):
            condition[i] = self.t[i].values[()].month in util.months_to_number(month_list)
        
        self.where(condition, drop=True)

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
