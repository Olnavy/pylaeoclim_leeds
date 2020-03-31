import numpy as np
import abc
import zones


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
