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

    @staticmethod
    def get(cube, zone=zones.NoZone()):
        return zone.compact(cube) if zone is not None else cube

    @staticmethod
    def mean(cube, zone=zones.NoZone()):
        cube = cube.mean(dim="t")
        return zone.compact(cube) if not isinstance(zone, zones.NoZone) else cube

    @staticmethod
    def mean_lon(tcube, zone=zones.NoZone()):
        cube = np.mean(tcube, axis=(0, 2))
        return zone.compact(cube) if zone is not None else cube

    @staticmethod
    def mean_lat(tcube, zone=zones.NoZone()):
        cube = np.mean(tcube, axis=(0, 1))
        return zone.compact(cube) if zone is not None else cube

    @staticmethod
    def mean_z(tcube, zone=zones.NoZone()):
        cube = np.mean(tcube, axis=(0, 3))
        return zone.compact(cube) if zone is not None else cube

    @staticmethod
    def mean_lon_lat(tcube, zone=zones.NoZone()):
        cube = np.mean(tcube, axis=(0, 1, 2))
        return zone.compact(cube) if zone is not None else cube

    @staticmethod
    def mean_lon_z(tcube, zone=zones.NoZone()):
        cube = np.mean(tcube, axis=(0, 2, 3))
        return zone.compact(cube) if zone is not None else cube

    @staticmethod
    def mean_lat_z(tcube, zone=zones.NoZone()):
        cube = np.mean(tcube, axis=(0, 1, 3))
        return zone.compact(cube) if zone is not None else cube

    @staticmethod
    def serie(tcube, zone=zones.NoZone()):
        if zone is not None:
            tcube_av = zone.compact(tcube)
        else:
            tcube_av = zone.compact(tcube)
        return np.mean(tcube_av, axis=(1, 2, 3))

    @staticmethod
    def serie_lon(tcube, zone=zones.NoZone()):
        if zone is not None:
            tcube_av = zone.compact(tcube)
        else:
            tcube_av = zone.compact(tcube)
        return np.mean(tcube_av, axis=(1, 3))

    @staticmethod
    def serie_lat(tcube, zone=zones.NoZone()):
        if zone is not None:
            tcube_av = zone.compact(tcube)
        else:
            tcube_av = zone.compact(tcube)
        return np.mean(tcube_av, axis=(2, 3))

    @staticmethod
    def serie_z(tcube, zone=zones.NoZone()):
        if zone is not None:
            tcube_av = zone.compact(tcube)
        else:
            tcube_av = zone.compact(tcube)
        return np.mean(tcube_av, axis=(1, 2))

    @staticmethod
    def serie_lon_lat(tcube, zone=zones.NoZone()):
        if zone is not None:
            tcube_av = zone.compact(tcube)
        else:
            tcube_av = zone.compact(tcube)
        return np.mean(tcube_av, axis=3)

    @staticmethod
    def serie_lon_z(tcube, zone=zones.NoZone()):
        if zone is not None:
            tcube_av = zone.compact(tcube)
        else:
            tcube_av = zone.compact(tcube)
        return np.mean(tcube_av, axis=1)

    @staticmethod
    def serie_lat_z(tcube, zone=zones.NoZone()):
        if zone is not None:
            tcube_av = zone.compact(tcube)
        else:
            tcube_av = zone.compact(tcube)
        return np.mean(tcube_av, axis=2)

    @staticmethod
    def global_mean(tcube, zone=zones.NoZone()):
        if zone is not None:
            tcube_av = zone.compact(tcube)
        else:
            tcube_av = zone.compact(tcube)
        return np.mean(tcube_av)


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
