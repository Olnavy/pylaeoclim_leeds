import numpy as np
# import numpy.ma as ma
import processing as proc
import abc


class Zone:
    
    def __init__(self, verbose):
        self.verbose = verbose
    
    @abc.abstractmethod
    def compact(self, cube):
        return

    @abc.abstractmethod
    def import_coordinates_from_data_array(self, data_array_source):
        pass


class NoZone(Zone):
    
    def __init__(self, verbose=False):
        super(NoZone, self).__init__(verbose)
    
    def compact(self, data_array):
        return data_array

    def import_coordinates_from_data_array(self, data_array_source):
        return  self


class Box(Zone):
    
    def __init__(self, lon_min=None, lon_max=None, lat_min=None, lat_max=None, z_min=None, z_max=None,
                 lon=None, lat=None, z=None, data_array_source=None, verbose=False):
        
        super(Box, self).__init__(verbose)
        
        if data_array_source is None:
            self.lon = lon
            self.lat = lat
            self.z = z
        else:
            self.import_coordinates_from_data_array(data_array_source)
        
        self.lon_min = np.min(self.lon) if lon_min is None and self.lon is not None else lon_min
        self.lon_max = np.max(self.lon) if lon_max is None and self.lon is not None else lon_max
        self.lat_min = np.min(self.lat) if lat_min is None and self.lat is not None else lat_min
        self.lat_max = np.max(self.lat) if lat_max is None and self.lat is not None else lat_max
        self.z_min = np.min(self.z) if z_min is None and self.z is not None else z_min
        self.z_max = np.max(self.z) if z_max is None and self.z is not None else z_max
        
        # if lsm is None:
        #     self.lsm = prc.LSM().default_lsm(lon, lat, z)
        # else:
        #     self.lsm = lsm
    
    def import_coordinates_from_data_set(self, ds):
        try:
            self.lon = ds.lon
        except AttributeError:
            self.lon = None
        try:
            self.lat = ds.lat
        except AttributeError:
            self.lat = None
        try:
            self.z = ds.z
        except AttributeError:
            self.z = None
        
        # if self.lsm is None:
        #     self.lsm = ds.lsm
    
    def get_indexes(self):
        
        if any([self.lon is None, self.lat is None, self.z is None]):
            print("Caution : Please import coordinates first")
            raise KeyError("Caution : Please import coordinates first")
        
        else:
            ilon_box = np.where(self.lon_min <= self.lon <= self.lon_max)
            ilat_box = np.where(self.lat_min <= self.lat <= self.lat_max)
            iz_box = np.where(self.z_min <= self.z <= self.z_max)
            return ilon_box, ilat_box, iz_box
    
    def compact(self, data_array):
        # Test lon etc.
        
        if self.lon is not None:
            data_array = data_array.where(data_array.longitude >= self.lon_min, drop=True) \
                .where(data_array.longitude <= self.lon_max, drop=True)
        if self.lat is not None:
            data_array = data_array.where(data_array.latitude >= self.lat_min, drop=True) \
                .where(data_array.latitude <= self.lat_max, drop=True)
        if self.z is not None:
            data_array = data_array.where(data_array.z >= self.z_min, drop=True) \
                .where(data_array.z <= self.z_max, drop=True)
        
        return proc.GeoDataArray(data_array)
    
    def import_coordinates_from_data_array(self, data_array_source):
        try:
            self.lon = data_array_source.longitude.values
        except AttributeError:
            self.lon = None
        try:
            self.lat = data_array_source.latitude.values
        except AttributeError:
            self.lat = None
        try:
            self.z = data_array_source.z.values
        except AttributeError:
            self.z = None
        
        self.update()
        
        return self
    
    def update(self):
        
        if self.lon is not None and self.lon_min is None:
            self.lon_min = np.min(self.lon)
        if self.lon is not None and self.lon_max is None:
            self.lon_max = np.max(self.lon)
        if self.lat is not None and self.lat_min is None:
            self.lat_min = np.min(self.lat)
        if self.lat is not None and self.lat_max is None:
            self.lat_max = np.max(self.lat)
        if self.z is not None and self.z_min is None:
            self.z_min = np.min(self.z)
        if self.z is not None and self.z_max is None:
            self.z_max = np.max(self.z)

        