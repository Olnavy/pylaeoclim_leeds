import numpy as np
import numpy.ma as ma
import processing as prc
import abc


class Zone:

    def __init__(self, verbose):
        if verbose:
            self.pprint = print
        pass

    @abc.abstractmethod
    def compact(self, cube):
        return


class NoZone(Zone):

    def __init__(self, verbose=False):
        super(NoZone, self).__init__(verbose)

    def compact(self, cube):
        return cube


class Box(Zone):

    def __init__(self, lon_min, lon_max, lat_min, lat_max, z_min, z_max, lon=None, lat=None, z=None, lsm=None,
                 verbose=False):
        super(Box, self).__init__(verbose)
        self.lon = lon
        self.lat = lat
        self.z = z
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.z_min = z_min
        self.z_max = z_max
        if lsm is None:
            self.lsm = prc.LSM().default_lsm(lon, lat, z)
        else:
            self.lsm = lsm

    def create_box_from_ds(self, ds):
        self.lon = ds.lon
        self.lat = ds.lat
        self.z = ds.z

        if self.lsm is None:
            self.lsm = ds.lsm

    def get_indexes(self):

        if any([self.lon is None, self.lat is None, self.z is None]):
            self.pprint("Caution : Please import coordinates first")
            raise KeyError("Caution : Please import coordinates first")

        else:
            ilon_box = np.where(self.lon_min <= self.lon <= self.lon_max)
            ilat_box = np.where(self.lat_min <= self.lat <= self.lat_max)
            iz_box = np.where(self.z_min <= self.z <= self.z_max)
            return ilon_box, ilat_box, iz_box

    def compact(self, cube):
        # Test lon etc.
        ilon_box, ilat_box, iz_box = self.get_indexes()
        return ma.masked_array(cube[ilon_box[0]:ilon_box[-1] + 1, ilat_box[0]:ilat_box[-1] + 1,
                               iz_box[0]:iz_box[-1] + 1], mask=self.lsm.mask3d)


class Plane(Zone):
    """
    FOR THE MOMENT, x = lon and y = lat; should adapt that with differnet projection.
    """

    def __init__(self, x_min, x_max, y_min, y_max, x=None, y=None, lsm=None, verbose=False):
        super(Plane, self).__init__(verbose)
        self.x = x
        self.y = y
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        if lsm is None:
            self.lsm = prc.LSM().default_lsm(x, y, [0])
        else:
            self.lsm = lsm

    def get_indexes(self):

        if any([self.x is None, self.y is None]):
            self.pprint("Caution : Please import coordinates first")
            raise KeyError("Caution : Please import coordinates first")

        else:
            ix_box = np.where(self.x_min <= self.x <= self.x_max)
            iy_box = np.where(self.y_min <= self.y <= self.y_max)
            return ix_box, iy_box

    def compact(self, cube):
        # PROBLEM WITH THE LSM!!!! Should I define the different projection?
        # Test lon etc.
        ix_box, iy_box = self.get_indexes()
        return ma.masked_array(cube[ix_box[0]:ix_box[-1] + 1, iy_box[0]:iy_box[-1] + 1], mask=self.lsm.mask2d)


class Line(Zone):
    """
    FOR THE MOMENT NO LAND SEA MASK
    """

    def __init__(self, x_min, x_max, x=None, verbose=False):
        super(Line, self).__init__(verbose)
        self.x = x
        self.x_min = x_min
        self.x_max = x_max

    def get_indexes(self):

        if self.x is None:
            self.pprint("Caution : Please import coordinates first")
            raise KeyError("Caution : Please import coordinates first")

        else:
            ix_box = np.where(self.x_min <= self.x <= self.x_max)
            return ix_box

    def compact(self, cube):
        # PROBLEM WITH THE LSM!!!! Should I define the different projection?
        # Test lon etc.
        ix_box = self.get_indexes()
        return ma.masked_array(cube[ix_box[0]:ix_box[-1] + 1])
