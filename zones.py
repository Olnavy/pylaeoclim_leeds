import numpy as np
import numpy.ma as ma


class Zones:
    
    def __init__(self, verbose):
        if verbose:
            self.pprint = print
        pass


class Box(Zones):
    
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
            self.lsm = LSM().default_lsm(lon,lat,z)
        else :
            self.lsm = lsm
    
    def create_box_from_ds(self, ds):
        self.lon = ds.lon
        self.lat = ds.lat
        self.z = ds.z
        
        if self.lsm is None:
            self.lsm = ds.lsm
    
    def get_indexes(self):
        if any([self.lon is None, self.lat is None, self is None]):
            self.pprint("Caution : Please import coordinates first")
            raise KeyError("Caution : Please import coordinates first")
        
        else:
            ilon_box = np.where(self.lon >= self.lon_min and self.lon <= self.lon_max)
            ilat_box = np.where(self.lat >= self.lat_min and self.lat <= self.lat_max)
            iz_box = np.where(self.z >= self.z_min and self.z <= self.z_max)
            return ilon_box, ilat_box, iz_box
    
    def compact(self, cube):
        # Test lon etc.
        
        ilon_box, ilat_box, iz_box = self.get_indexes()
        return ma.masked_array(cube[ilon_box[0]:ilon_box[-1] + 1, ilat_box[0]:ilat_box[-1] + 1,
                               iz_box[0]:iz_box[-1] + 1], mask=self.lsm.mask)